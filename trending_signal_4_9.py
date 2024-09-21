import ccxt
import pandas as pd
import talib
import time
import numpy as np
import pytz
import requests
from datetime import datetime, timedelta
import signal
import sys
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from concurrent.futures import ThreadPoolExecutor
import ta
import requests




# Telegram Setup
bot_token = ''  # Replace with your bot's token
#channel_id = '@'  # Replace with your channel's username

def send_message(message):
    """Send a message to a Telegram channel."""
    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    payload = {
        'chat_id': '@',  # Replace with your bot's token
        'text': message,
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print('Message sent successfully.')
    else:
        print(f'Failed to send message. Status code: {response.status_code}. Response: {response.text}')


# Setup Exchange Connection
exchange = ccxt.binance({
    'apiKey': '',
    'secret': '',
    'enableRateLimit': True,
})
symbols = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 
    'DOGE/USDT', 'XRP/USDT', 'DOT/USDT', 'LTC/USDT', 'BCH/USDT',
    'LINK/USDT', 'MATIC/USDT', 'AVAX/USDT', 'XLM/USDT', 'TRX/USDT',
    'ATOM/USDT', 'VET/USDT', 'FIL/USDT', 'EOS/USDT', 'AAVE/USDT',
    'UNI/USDT', 'ALGO/USDT', 'XTZ/USDT', 'COMP/USDT',
    'YFI/USDT', 'MKR/USDT', 'SNX/USDT', 'ZIL/USDT', 'KSM/USDT',
    'LUNA/USDT', 'ENJ/USDT', 'BAT/USDT', 'CRV/USDT', 'MANA/USDT',
    '1INCH/USDT', 'GRT/USDT', 'ANKR/USDT', 'SAND/USDT', 'CHZ/USDT',
    'VITE/USDT', 'BNX/USDT', 'QKC/USDT', 'SUPER/USDT', 'RDNT/USDT',
    'CREAM/USDT', 'HARD/USDT', 'GFT/USDT', 'NULS/USDT', 'FIO/USDT',
    'AUDIO/USDT', 'KLAY/USDT', 'PAXG/USDT', 'DGB/USDT', 'AVA/USDT',
    'LDO/USDT', 'ARN/USDT', 'OOKI/USDT', 'WLD/USDT'
]


def fetch_ohlcv(symbol, timeframe, limit):
    """Fetch OHLCV data for a symbol and a given timeframe."""
    return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

def calculate_volatility(ohlcv):
    """Calculate volatility as the percentage amplitude of the price range."""
    highs = [candle[2] for candle in ohlcv]
    lows = [candle[3] for candle in ohlcv]
    amplitude = ((max(highs) - min(lows)) / min(lows)) * 100
    return amplitude

def calculate_volatility_for_symbol(symbol, timeframe, limit):
    """Fetch OHLCV data and calculate volatility for a single symbol."""
    try:
        # Fetch data using a consistent method
        ohlcv = fetch_ohlcv(symbol, timeframe, limit)
        
        # Ensure data is not empty and has sufficient rows
        if ohlcv is not None and len(ohlcv) >= limit:
            volatility = calculate_volatility(ohlcv)
            return symbol, volatility
        else:
            print(f"Not enough data for {symbol}. Expected at least {limit} rows.")
            return symbol, None

    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return symbol, None


def fetch_and_rank_volatilities(symbols, timeframe, limit):
    """Fetch and rank the most volatile pairs, returning only the symbol names."""
    df = []

    for symbol in symbols:
        try:
            # Fetch data using the same method as in score_cryptos
            df = get_data(symbol, timeframe, limit)

            if len(df) < limit:
                print(f"Not enough data for {symbol}. Expected {limit}, got {len(df)}")
                continue  # Skip this symbol if there's not enough data

            # Calculate volatility as the standard deviation of the close prices
            volatility = df['close'].pct_change().std() * 100  # Percent volatility

            if volatility is not None:
                df.append([symbol, volatility])

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            continue  # Skip to the next symbol if there's an error

    # Convert the data to a DataFrame
    df = pd.DataFrame(data, columns=['Symbol', 'Volatility (%)'])

    # Sort the DataFrame by Volatility in descending order
    df = df.sort_values(by='Volatility (%)', ascending=False)

    # Return a list of symbols
    return df['Symbol'].tolist()



# Fetch data from Binance
def get_data(symbol: str, timeframe: str, limit=100):
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Signal handling for clean exit
def signal_handler(sig, frame):
    send_message("Stopping signal bot...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


timeframe = '1h'  # Fetch data in a smaller timeframe (1 minute)
initial_capital = 100  # Starting with $100
current_capital = initial_capital
stop_loss = None
btc_trend_confirmation = False
last_btc_update = datetime.now() - timedelta(hours=1)
last_status_update = datetime.now() - timedelta(hours=1)
trade_history = []
position = None  # No position at the start
best_crypto = None

# DQN Parameters
state_size = 22  # Number of features in the state representation (based on indicators)
action_size = 3  # Three possible actions: buy, sell, hold
batch_size = 32

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.999  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.trade_count = 0
        self.last_action = None
        self.last_price = None
        self.cumulative_returns = []
        self.model_checkpoint = ModelCheckpoint('model1.keras', monitor='loss', save_best_only=True)

    def _build_model(self):
        model = Sequential()
        # Input layer
        model.add(Input(shape=(self.state_size,)))  # Explicitly define input shape
        # First block of layers
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        # Second block of layers
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        # Third block of layers
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        # Fourth block of layers
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        # Fifth block of layers
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        # Sixth block of layers (optional, for even more depth)
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        # Seventh block of layers (optional, for even more depth)
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        # Output layer
        model.add(Dense(self.action_size, activation='linear'))
        # Compile the model
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))    
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        global position
        state = np.array(state).reshape(1, -1)  # Ensure state is reshaped correctly
        df = fetch_data(best_crypto, '1m', limit=500)
        df = apply_technical_indicators(df)

        # Extract useful features from the state
        current_price = state[0][0]
        previous_price = state[0][3]
        price_movement = current_price - previous_price
        volatility_index = state[0][-1]  # Assuming the last feature is a volatility indicator (e.g., ATR)
        rsi = state[0][5]  # Assuming RSI is at index 5, adjust based on actual index
        moving_average_diff = state[0][2] - state[0][1]  # Example of using two moving averages, adjust indices

        # Set a default value for volatility_threshold at the start
        volatility_threshold = 0.04  # Adjust based on experimentation

        # Improved heuristic-based action selection
        if np.random.rand() <= self.epsilon:
            action = 2  # Hold in other cases
            message = f"Heuristic action: Hold"
            if position == 'long':
                if detect_peak_condition(df) and best_crypto != score_cryptos(symbols):
                    action = 1  # Sell if price is down, RSI is above oversold level, and volatility is high
                    message = f"Heuristic action: Buy"
            else :
                if detect_buy_condition(df) and best_crypto == score_cryptos(symbols):
                    action = 0  # Buy if price is up, RSI is below overbought level, and volatility is low
                    message = f"Heuristic action: Sell"
        else:
            # Use model to predict action values
            act_values = self.model.predict(state)

            # Volatility-based decision adjustment
            confidence_threshold = 0.7  # Lowered confidence threshold to be more cautious in volatile markets

            action_confidence = np.max(act_values[0])

            if volatility_index > volatility_threshold:
                if action_confidence < confidence_threshold and position == 'long':
                    action = 2  # Hold in high volatility with low confidence
                    message = f"Model action: Hold (high volatility, low confidence)"
                else:
                    action = np.argmax(act_values[0])
                    message = f"Model action: {action} with confidence {action_confidence:.2f}"
            else:
                # In low volatility, trust the model's prediction more
                action = np.argmax(act_values[0])
                message = f"Model action: {action} with values {act_values[0]}"

        self.last_action = action
        # Uncomment the print statement to log decisions
        # print(message)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        #send_message(f"action : {action}")

        return action


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
        
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[self.model_checkpoint])

            sharpe_ratio = self.calculate_sharpe_ratio(self.cumulative_returns)
            sortino_ratio = self.calculate_sortino_ratio(self.cumulative_returns)

            if self.trade_count > 0:
                penalty = -0.01 * self.trade_count
                reward += penalty

            if self.last_action == 1 and self.detect_sharp_drop(self.last_price, next_state):
                reward += 1.0

            if self.last_action == 1:
                current_return = (next_state[0][-1] - self.last_price) / self.last_price
                self.cumulative_returns.append(current_return)

            if sharpe_ratio < 1:
                reward *= 0.9
                # Bad decision detected: reset epsilon to encourage exploration
                self.epsilon = min(self.epsilon + 0.05, 1.0)  # Increase epsilon, max value 1.0
            elif sortino_ratio < 1:
                reward *= 0.9
                # Another bad decision indicator: reset epsilon
                self.epsilon = min(self.epsilon + 0.05, 1.0)
            else:
                reward *= 1.1
                # Good decisions can still decay epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

        self.trade_count += 1


    def calculate_sharpe_ratio(self, returns):
        if len(returns) == 0:
            return 0
        mean_return = np.mean(returns)
        std_dev = np.std(returns)
        sharpe_ratio = mean_return / std_dev if std_dev != 0 else 0
        return sharpe_ratio


    def calculate_sortino_ratio(self, returns):
        if len(returns) == 0:
            return 0
        mean_return = np.mean(returns)
        downside_deviation = np.std([r for r in returns if r < 0])
        sortino_ratio = mean_return / downside_deviation if downside_deviation != 0 else 0
        return sortino_ratio

    def detect_sharp_drop(self, last_price, next_state):
        # Implement logic to detect sharp drop
        current_price = next_state[0][-1]
        return (last_price - current_price) / last_price > 0.05  # Example: 5% drop as a threshold

    # Add other methods as needed, such as load, save, etc.




# Initialize the DQN agent
agent = DQNAgent(state_size, action_size)

# Fetch Historical Data
def fetch_data(symbol, timeframe, limit=400):
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Paris')
    return df



# Convert indicators to state representation for DQN
def get_state(df):
    # Calculate the peak signal (1 if peak detected, 0 otherwise)
    peak_signal = 1 if detect_peak_condition(df) else 0
    
    # Return the state as an array including the peak signal
    return np.array([
        df['SMA'].iloc[-1],
        df['EMA'].iloc[-1],
        df['WMA'].iloc[-1],
        df['EMA_12'].iloc[-1],
        df['EMA_26'].iloc[-1],
        df['RSI'].iloc[-1],
        df['Stochastic_K'].iloc[-1],
        df['Stochastic_D'].iloc[-1],
        df['CCI'].iloc[-1],
        df['MACD'].iloc[-1],
        df['MACD_Signal'].iloc[-1],
        df['MOM'].iloc[-1],
        df['ATR'].iloc[-1],
        df['Volatility'].iloc[-1],
        df['Upper_BB'].iloc[-1],
        df['Lower_BB'].iloc[-1],
        df['OBV'].iloc[-1],
        df['ADOSC'].iloc[-1],
        df['ADX'].iloc[-1],
        df['SAR'].iloc[-1],
        df['TRIX'].iloc[-1],
        peak_signal  # New feature: Peak signal as part of the state
    ])


# Initialize performance tracking for each indicator
indicator_performance = {
    'SMA': [],
    'EMA': [],
    'WMA': [],
    'EMA_12': [],
    'EMA_26': [],
    'RSI': [],
    'Stochastic_K': [],
    'Stochastic_D': [],
    'CCI': [],
    'MACD': [],
    'MACD_Signal': [],
    'MOM': [],
    'ATR': [],
    'Volatility': [],
    'Upper_BB': [],
    'Lower_BB': [],
    'OBV': [],
    'ADOSC': [],
    'ADX': [],
    'SAR': [],
    'ROC': [],
    'WilliamsR': [],
    'TRIX': [],
}

# Function to apply technical indicators
def apply_technical_indicators(df):
    # Moving Averages
    df['SMA'] = talib.SMA(df['close'], timeperiod=14)  # Lengthened from 40 to 100
    df['EMA'] = talib.EMA(df['close'], timeperiod=14)  # Lengthened from 40 to 100
    df['WMA'] = talib.WMA(df['close'], timeperiod=14)  # Lengthened from 40 to 100
    
    # Additional EMAs
    df['EMA_12'] = talib.EMA(df['close'], timeperiod=12)  # Lengthened from 12 to 50 for medium-term trend
    df['EMA_26'] = talib.EMA(df['close'], timeperiod=26)  # Lengthened from 26 to 200 for long-term trend
    
    # Oscillators
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)  # Lengthened from 40 to 70
    df['Stochastic_K'], df['Stochastic_D'] = talib.STOCH(df['high'], df['low'], df['close'],
                                                          fastk_period=21, slowk_period=7, slowd_period=7)  # Lengthened from 14-3-3 to 21-7-7
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=70)  # Lengthened from 40 to 70
    
    # Momentum Indicators
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['close'], fastperiod=50, slowperiod=200, signalperiod=20)  # Lengthened fast/slow periods
    df['MOM'] = talib.MOM(df['close'], timeperiod=30)  # Lengthened from 14 to 30
    
    # Volatility Indicators
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)  # Lengthened from 40 to 70
    df['Volatility'] = df['close'].pct_change().rolling(window=30).std() * np.sqrt(14)  # Lengthened window to 30, and period factor to 70
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['close'], timeperiod=14, nbdevup=2, nbdevdn=2, matype=0)  # Lengthened from 20 to 50
    
    # Volume Indicators
    df['OBV'] = talib.OBV(df['close'], df['volume'])  # On-Balance Volume (no period adjustment needed)
    df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=20, slowperiod=60)  # Lengthened fast/slow periods
    
    # Trend Indicators
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)  # Lengthened from 40 to 70
    df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0.01, maximum=0.2)  # Slowed down SAR by reducing acceleration
    
    # Additional Momentum Indicators
    df['ROC'] = talib.ROC(df['close'], timeperiod=30)  # Lengthened from 10 to 30
    df['WilliamsR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)  # Lengthened from 14 to 21
    
    # Trend Strength Indicator
    df['TRIX'] = talib.TRIX(df['close'], timeperiod=30)  # Lengthened from 15 to 30

    return df

def detect_buy_condition(df):
    """
    Detects if the current market conditions suggest a good buying opportunity,
    optimized for short-term investments in highly volatile assets.
    """
    
    # Get the latest values of the indicators
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    stochastic_k = df['Stochastic_K'].iloc[-1]
    stochastic_d = df['Stochastic_D'].iloc[-1]
    upper_bb = df['Upper_BB'].iloc[-1]
    lower_bb = df['Lower_BB'].iloc[-1]
    close_price = df['close'].iloc[-1]
    cci = df['CCI'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    williams_r = df['WilliamsR'].iloc[-1]
    sar = df['SAR'].iloc[-1]
    trix = df['TRIX'].iloc[-1]
    
    # Tighter Buy condition criteria
    rsi_buy = rsi < 30  # Lower threshold for oversold, making it stricter
    macd_buy = macd > macd_signal and macd > 0  # Bullish crossover with MACD above 0
    stochastic_buy = stochastic_k < 30 and stochastic_k > stochastic_d  # Stricter threshold
    bollinger_band_buy = close_price < lower_bb * 0.98  # Price well below lower BB, more conservative
    cci_buy = cci < -100  # Even more oversold
    atr_buy = atr > df['ATR'].rolling(window=14).mean().iloc[-1] * 1.2  # Higher volatility, stricter requirement
    williams_r_buy = williams_r < -45  # More oversold in Williams %R
    sar_buy = close_price > sar  # Price above SAR, no change needed
    trix_buy = trix > 0 and trix > df['TRIX'].rolling(window=5).mean().iloc[-1]  # TRIX bullish and rising
    
    # Require more indicators to signal a buy
    indicators_buy = [
        rsi_buy, macd_buy, stochastic_buy, 
        bollinger_band_buy, cci_buy, atr_buy, 
        williams_r_buy, sar_buy, trix_buy
    ]
    buy_signals = sum(indicators_buy)
    send_message(f"buy_signal = {buy_signals}")
    
    return buy_signals >= 0  # Require at least 6 indicators to agree



# Function to adjust entry criteria based on trade outcomes

def detect_peak_condition(df):
    """
    Detects if the current market conditions suggest that the price is near a peak,
    optimized for short-term trading in highly volatile assets.
    Uses a combination of RSI, MACD, Stochastic Oscillator, Bollinger Bands, and additional indicators like CCI, ATR, and Williams %R.
    """

    # Get the latest values of the indicators
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    stochastic_k = df['Stochastic_K'].iloc[-1]
    stochastic_d = df['Stochastic_D'].iloc[-1]
    upper_bb = df['Upper_BB'].iloc[-1]
    lower_bb = df['Lower_BB'].iloc[-1]
    close_price = df['close'].iloc[-1]
    cci = df['CCI'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    williams_r = df['WilliamsR'].iloc[-1]
    sar = df['SAR'].iloc[-1]
    trix = df['TRIX'].iloc[-1]

    # Optimized Peak detection criteria for short-term, highly volatile trading
    rsi_peak = rsi > 75  # Slightly lower threshold to catch peaks earlier
    macd_peak = macd < macd_signal  # MACD crossing below the signal line
    stochastic_peak = stochastic_k > 75 and stochastic_k > stochastic_d  # Stochastic K is over 75 and starting to drop
    bollinger_band_peak = close_price > upper_bb  # Price is above the upper Bollinger Band
    cci_peak = cci > 100  # CCI above 100 can indicate overbought conditions
    atr_peak = atr > df['ATR'].rolling(window=14).mean().iloc[-1]  # Use a 14-period ATR to capture more recent volatility
    williams_r_peak = williams_r > -20  # Williams %R indicating overbought conditions
    sar_peak = close_price < sar  # Price falling below SAR suggests a potential reversal
    trix_peak = trix < 0  # TRIX crossing below zero suggests a weakening trend

    # If three or more indicators suggest a peak, return True
    indicators_peak = [
        rsi_peak, macd_peak, stochastic_peak, 
        bollinger_band_peak, cci_peak, atr_peak, 
        williams_r_peak, sar_peak, trix_peak
    ]
    peak_signals = sum(indicators_peak)

    return peak_signals >= 8


# Function to score and select the best crypto to trade
def score_cryptos(symbols, period=15):
    global last_status_update
    best_crypto = None
    highest_score = -float('inf')
    highest_volatility = -float('inf')
    current_time = datetime.now()


    for symbol in symbols:
        try:
            score = 0
            # Fetch and process data
            data = get_data(symbol, '1h', limit=100)
            #data = apply_technical_indicators(df)

            data['SMA'] = ta.trend.sma_indicator(data['close'], window=period)
            data['EMA'] = ta.trend.ema_indicator(data['close'], window=period)
            data['RSI'] = ta.momentum.rsi(data['close'], window=period)
            data['MACD'] = ta.trend.macd_diff(data['close'])
            data['ADX'] = ta.trend.adx(data['high'], data['low'], data['close'], window=period)
    
            # Calculate the slope of SMA and EMA
            sma_slope = data['SMA'].iloc[-1] - data['SMA'].iloc[-2]
            ema_slope = data['EMA'].iloc[-1] - data['EMA'].iloc[-2]
    
            # Interpret SMA and EMA slopes
            if sma_slope > 0 :
                score += 1
            else :
                score += -1
            if ema_slope > 0 :
                score += 1
            else :
                score += -1

            # Interpret RSI
            rsi_value = data['RSI'].iloc[-1]
    
            if rsi_value > 70:
                score += -2
            elif rsi_value < 30:
                score += 2
            elif 50 <= rsi_value <= 70:
                score += -1
            elif 30 <= rsi_value < 50:
                score += 1
            else:
                score += 0
    
            # Interpret MACD
            macd_value = data['MACD'].iloc[-1]
            
            if macd_value > 0 :
                score += 1
            else :
                score += -1
    
            # Interpret ADX
            adx_value = data['ADX'].iloc[-1]
            if adx_value > 25 :
                score += 1
            elif adx_value < 20 :
                score += -1
            else :
                score += 0


            # Fetch data using a consistent method
            ohlcv = fetch_ohlcv(symbol, '1h', 15)
        
            # Ensure data is not empty and has sufficient rows
            if ohlcv is not None and len(ohlcv) >= 15:
                volatility = calculate_volatility(ohlcv)
            else:
                print(f"Not enough data for {symbol}. Expected at least {limit} rows.")


            final_score = score
            #send_message(f"{symbol} final score : {final_score}   volatility : {volatility}")

            # Update the best crypto if this one has the highest score
            #send_telegram_message(f"{symbol} FINAAL SCORE {final_score}")
            if final_score >= highest_score :
                if final_score > highest_score:
                    highest_score = final_score
                    highest_volatility = volatility
                    best_crypto = symbol
                elif final_score == highest_score and volatility > highest_volatility:
                    highest_score = final_score
                    highest_volatility = volatility
                    best_crypto = symbol
                else:
                    continue


        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue  # Skip to the next symbol if there's an error
        time.sleep(2)

    # Report the best crypto if found
    if best_crypto:
        print(f"Best Crypto: {best_crypto}")
        if current_time - last_status_update >= timedelta(hours=1):
            send_message(f"Best Crypto selected for trading: {best_crypto} score : {final_score}, volatility : {highest_volatility},  ")
            last_status_update = current_time  # Update the last status update time

    return best_crypto


import traceback

def run_signal_bot():
    global current_capital, stop_loss, position, best_crypto
    print("Starting trader bot...")
    send_message(f"Starting signal bot... current capital: ${current_capital:.2f}")

    entry_price = None
    dynamic_stop_loss = None
    stop_loss_margin = None  # Margin will be determined by the DQN agent
    profit_loss_percent = 0
    current_time = datetime.now()

    bot_trade_count = 0  # Renamed trade_count to bot_trade_count to avoid conflicts
    trade_count = 0
    cumulative_profit_loss = 0
    returns = []  # Track returns to calculate Sharpe/Sortino ratio
    trade_rewards = []  # Store individual trade rewards

    while True:
        try:
            if position is None:
                best_crypto = score_cryptos(symbols)
                send_message(f"best_crypto0  {best_crypto}")
                if best_crypto:
                    send_message(f"best_crypto1  {best_crypto}")
                    df = fetch_data(best_crypto, '1m', limit=100)
                    df = apply_technical_indicators(df)
                    state = get_state(df).reshape(1, -1)
                    if 'RSI' not in df.columns:
                        raise ValueError(f"RSI indicator is missing from the data for {best_crypto}")

                    action = agent.act(state)

                    if action == 0:  # Buy
                        df = fetch_data(best_crypto, '1s', limit=14)
                        current_price = df['close'].iloc[-1] 
                        limit_price = current_price * 1  # 0.1% lower

                        send_message(f"Placing buy order for {best_crypto} at {limit_price:.4f}, waiting for order to fill...")

                        # Wait for 5 minutes to see if the price hits the limit order
                        order_filled = False
                        start_time = time.time()

                        while best_crypto == score_cryptos(symbols):
                            df = fetch_data(best_crypto, '1s', limit=100)
                            last_price = df['close'].iloc[-1]

                            if last_price <= limit_price:
                                entry_price = last_price
                                position = 'long'
                                agent.last_price = entry_price
                                df = fetch_data(best_crypto, '1h', limit=100)
                                df = apply_technical_indicators(df)  # Ensure ATR is calculated here
                                atr_value = df['ATR'].iloc[-1]  # Fetch the latest ATR value
                                stop_loss_margin = (agent.act(state) / 1000) + (atr_value * 2) / 1000 + 0.02# Calculate the margin based on ATR * 2
                                dynamic_stop_loss = entry_price * (1 - stop_loss_margin)
                                send_message(f"Buy order for {best_crypto} filled at {entry_price}, initial stop loss at {dynamic_stop_loss} (margin: {stop_loss_margin:.3f})")
                                print(f"Buy order for {best_crypto} filled at {entry_price}")
                                order_filled = True
                                break

                            time.sleep(10)  # Check every 10 seconds

                        if not order_filled:
                            continue

                    reward = 0
                    df = fetch_data(best_crypto, '1h', limit=100)
                    df = apply_technical_indicators(df)
                    next_state = get_state(df).reshape(1, -1)
                    agent.remember(state, action, reward, next_state, position is None)

                    if len(agent.memory) > batch_size:
                        agent.replay(batch_size)

            elif position == 'long':
                df = fetch_data(best_crypto, '1s', limit=1)
                last_price = df['close'].iloc[-1]
                df = fetch_data(best_crypto, '1h', limit=500)
                df = apply_technical_indicators(df)
                final_exit_price = last_price

                if last_price < dynamic_stop_loss or last_price >= entry_price * 1.02:
                    send_message(f"Sell {best_crypto} at {final_exit_price} due to stop loss")
                    action = 1  # Force sell
                
                else:
                    state = get_state(df).reshape(1, -1)
                    action = agent.act(state)

                if action == 1:  # Sell based on DQN or forced conditions
                    profit_loss_percent = (((final_exit_price - entry_price) / entry_price * 100) - 0.1) * 5
                    agent.last_price = final_exit_price
                    cumulative_profit_loss += (profit_loss_percent * 5)
                    bot_trade_count += 1
                    current_capital *= (1 + profit_loss_percent / 100)
                    trade_rewards.append(profit_loss_percent)
                    returns.append(profit_loss_percent / 100)

                    sharpe_ratio = agent.calculate_sharpe_ratio(returns)
                    sortino_ratio = agent.calculate_sortino_ratio(returns)
                    if sharpe_ratio < 1 or sortino_ratio < 1:
                        trade_rewards[-1] *= 0.9
                    else:
                        trade_rewards[-1] *= 1.1

                    send_message(f"Trade completed at {final_exit_price} with a {profit_loss_percent:.2f}% change, New Capital: ${current_capital:.2f}")
                    print(f"Selling {best_crypto} at {final_exit_price} due to stop-loss or DQN decision")
                    position = None

                    if trade_count == 5:
                        reward = ((1 + cumulative_profit_loss / 100) ** 2) - 1
                        if agent.detect_sharp_drop(entry_price, df):
                            reward += 1.0
                        indicators = get_state(df).reshape(1, -1).flatten()
                        agent.remember(indicators, 1, reward, None, True)
                        bot_trade_count = 0
                        cumulative_profit_loss = 0
                        trade_rewards.clear()
                        returns.clear()
                        trade_count = 0

                    continue

                else:  # DQN decides to update the stop-loss margin
                    df = fetch_data(best_crypto, '1h', limit=100)
                    df = apply_technical_indicators(df)
                    atr_value = df['ATR'].iloc[-1]
                    new_stop_loss_margin = (agent.act(state) / 1000) + (atr_value * 2) / 1000 + 0.02
                    new_dynamic_stop_loss = (last_price * (1 - new_stop_loss_margin))
                    agent.last_price = final_exit_price
                    if new_dynamic_stop_loss > dynamic_stop_loss and last_price > entry_price:
                        dynamic_stop_loss = new_dynamic_stop_loss
                        send_message(f"Updated dynamic stop loss for {best_crypto} to {dynamic_stop_loss} (margin: {new_stop_loss_margin:.3f})")
                        print(f"Updated dynamic stop loss for {best_crypto} to {dynamic_stop_loss}")

                next_state = get_state(df).reshape(1, -1)
                agent.remember(state, 2, 0, next_state, False)

                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

            time.sleep(30)

        except Exception as e:
            error_message = f"An error occurred: {str(e)}\n"
            error_message += traceback.format_exc()  # Capture the full stack trace
            send_message(error_message)  # Send full error details
            print(error_message)  # Print error details for further analysis
            time.sleep(30)


# Main Function
if __name__ == "__main__":
    run_signal_bot()


