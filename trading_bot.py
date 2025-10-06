"""
Complete Advanced Forex & Indices Trading Bot with Telegram Notifications
Enhanced version with 70% confidence threshold and improved signal display
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
import json
import os
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Structure for trade signals"""
    symbol: str
    action: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    timeframe: str
    indicators: Dict
    timestamp: datetime
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class TelegramNotifier:
    """Handle Telegram notifications with retry logic"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.max_retries = 3
        
    def send_message(self, message: str, parse_mode: str = "HTML", 
                    reply_markup: dict = None, retry_count: int = 0) -> bool:
        """Send message via Telegram with retry logic"""
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': parse_mode
        }
        if reply_markup:
            payload['reply_markup'] = json.dumps(reply_markup)
            
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
                return True
            elif response.status_code == 429 and retry_count < self.max_retries:
                # Rate limited, wait and retry
                retry_after = int(response.json().get('parameters', {}).get('retry_after', 30))
                logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                return self.send_message(message, parse_mode, reply_markup, retry_count + 1)
            else:
                logger.error(f"Failed to send Telegram message: {response.text}")
                return False
        except requests.exceptions.Timeout:
            if retry_count < self.max_retries:
                logger.warning(f"Timeout. Retry {retry_count + 1}/{self.max_retries}")
                time.sleep(5)
                return self.send_message(message, parse_mode, reply_markup, retry_count + 1)
            logger.error("Max retries reached for Telegram message")
            return False
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return False
    
    def send_signal(self, signal: TradeSignal) -> bool:
        """Send formatted trading signal"""
        
        emoji = "🟢" if "BUY" in signal.action else "🔴" if "SELL" in signal.action else "⚪"
        
        # Confidence rating with emojis
        if signal.confidence >= 85:
            confidence_emoji = "🔥🔥🔥"
            confidence_label = "VERY HIGH"
        elif signal.confidence >= 75:
            confidence_emoji = "🔥🔥"
            confidence_label = "HIGH"
        else:
            confidence_emoji = "🔥"
            confidence_label = "MODERATE"
        
        message = f"""
{emoji} <b>{confidence_label} CONFIDENCE SIGNAL</b> {confidence_emoji}

<b>Symbol:</b> {signal.symbol}
<b>Action:</b> {signal.action}

💰 <b>Trade Setup</b>
<b>Confidence:</b> {signal.confidence:.1f}% ⭐
Entry: {signal.entry_price:.5f}
Stop Loss: {signal.stop_loss:.5f}
Take Profit: {signal.take_profit:.5f}
Risk/Reward: 1:{signal.risk_reward_ratio:.2f}

📊 <b>Technical Indicators</b>
RSI: {signal.indicators.get('rsi', 0):.2f}
MACD: {signal.indicators.get('macd', 0):.5f}
ADX: {signal.indicators.get('adx', 0):.2f}
Trend: {signal.indicators.get('trend', 'N/A')}
Momentum: {signal.indicators.get('momentum', 'N/A')}

🕐 <b>Timeframe:</b> {signal.timeframe}
📅 <b>Time:</b> {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

⚠️ <i>Risk Management: Never risk more than 1-2% of your capital per trade</i>
"""
        
        keyboard = {
            'inline_keyboard': [
                [
                    {'text': '✅ Trade Taken', 'callback_data': f'taken_{signal.symbol}'},
                    {'text': '❌ Skip Trade', 'callback_data': f'skip_{signal.symbol}'}
                ],
                [
                    {'text': '📊 View Chart', 'url': f'https://www.tradingview.com/chart/?symbol={signal.symbol}'},
                ]
            ]
        }
        
        return self.send_message(message, reply_markup=keyboard)
    
    def send_error(self, error_msg: str):
        """Send error notification"""
        message = f"⚠️ <b>Bot Error</b>\n\n<code>{error_msg}</code>"
        self.send_message(message)
    
    def send_startup(self, symbols: List[str], min_confidence: float, timeframes: List[str]):
        """Send startup notification"""
        message = f"""
🤖 <b>Trading Bot Started</b>

<b>Configuration:</b>
• Symbols: {', '.join(symbols)}
• Timeframe: {', '.join(timeframes)}
• Min Confidence: {min_confidence}%
• Mode: GitHub Actions (Hourly)

Status: ✅ Active
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)


class AdvancedTradingBot:
    """Advanced trading bot with multi-timeframe analysis"""
    
    def __init__(self, symbols: List[str], telegram_token: str, telegram_chat_id: str, api_key: str):
        self.symbols = symbols
        self.notifier = TelegramNotifier(telegram_token, telegram_chat_id)
        self.min_confidence = 70  # Lowered from 90% to 70%
        self.timeframes = ['1h']  # Single timeframe to stay within API limits
        self.api_key = api_key
        self.api_call_count = 0
        self.max_api_calls = 75  # Alpha Vantage free tier limit per day
        
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV data from Alpha Vantage with improved error handling"""
        
        if self.api_call_count >= self.max_api_calls:
            logger.warning(f"API call limit reached ({self.max_api_calls}). Using demo data.")
            return self._generate_demo_data(symbol, limit)
        
        interval_map = {'15m': '15min', '1h': '60min', '4h': '60min', '1d': 'daily'}
        interval = interval_map.get(timeframe, '60min')
        
        from_symbol = symbol[:3]
        to_symbol = symbol[3:6] if len(symbol) >= 6 else 'USD'
        
        try:
            url = "https://www.alphavantage.co/query"
            
            if interval == 'daily':
                params = {
                    'function': 'FX_DAILY',
                    'from_symbol': from_symbol,
                    'to_symbol': to_symbol,
                    'apikey': self.api_key,
                    'outputsize': 'full'
                }
            else:
                params = {
                    'function': 'FX_INTRADAY',
                    'from_symbol': from_symbol,
                    'to_symbol': to_symbol,
                    'interval': interval,
                    'apikey': self.api_key,
                    'outputsize': 'full'
                }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            self.api_call_count += 1
            logger.info(f"API calls used: {self.api_call_count}/{self.max_api_calls}")
            
            if "Error Message" in data:
                logger.error(f"Alpha Vantage Error: {data['Error Message']}")
                return self._generate_demo_data(symbol, limit)
            
            if "Note" in data:
                logger.warning(f"Alpha Vantage Rate Limit: {data['Note']}")
                return self._generate_demo_data(symbol, limit)
            
            if "Information" in data:
                logger.warning(f"Alpha Vantage Info: {data['Information']}")
                return self._generate_demo_data(symbol, limit)
            
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                logger.warning(f"No time series data found for {symbol}")
                return self._generate_demo_data(symbol, limit)
            
            time_series = data[time_series_key]
            
            if not time_series:
                logger.warning(f"Empty time series for {symbol}")
                return self._generate_demo_data(symbol, limit)
            
            df_data = []
            for timestamp, values in time_series.items():
                try:
                    df_data.append({
                        'timestamp': pd.to_datetime(timestamp),
                        'open': float(values.get('1. open', 0)),
                        'high': float(values.get('2. high', 0)),
                        'low': float(values.get('3. low', 0)),
                        'close': float(values.get('4. close', 0)),
                        'volume': 0  # FX data doesn't have volume
                    })
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid data point: {e}")
                    continue
            
            if not df_data:
                logger.warning(f"No valid data points for {symbol}")
                return self._generate_demo_data(symbol, limit)
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # For 4h timeframe, resample 1h data
            if timeframe == '4h' and interval == '60min':
                df = self._resample_to_4h(df)
            
            df = df.tail(limit)
            
            logger.info(f"Fetched {len(df)} candles for {symbol} ({timeframe})")
            return df
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching data for {symbol}")
            return self._generate_demo_data(symbol, limit)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {symbol}: {str(e)}")
            return self._generate_demo_data(symbol, limit)
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {symbol}: {str(e)}")
            return self._generate_demo_data(symbol, limit)
    
    def _resample_to_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample 1h data to 4h"""
        df.set_index('timestamp', inplace=True)
        df_4h = df.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        df_4h.reset_index(inplace=True)
        return df_4h
    
    def _generate_demo_data(self, symbol: str, limit: int) -> pd.DataFrame:
        """Generate realistic demo data as fallback"""
        logger.warning(f"Using demo data for {symbol}")
        
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1H')
        np.random.seed(hash(symbol) % 2**32)
        
        base_price = {
            'EURUSD': 1.0850, 'GBPUSD': 1.2650, 'USDJPY': 148.50,
            'AUDUSD': 0.6580, 'USDCAD': 1.3850, 'NZDUSD': 0.6120,
            'XAUUSD': 2650.00, 'US30': 38500.00,
            'SPX500': 4850.00, 'NAS100': 16900.00,
            'BOOM500': 450000.00, 'CRASH500': 8500.00,
            'BOOM1000': 850000.00, 'CRASH1000': 4500.00
        }
        
        price = base_price.get(symbol, 1.10)
        
        # Adjust volatility based on instrument type
        if symbol == 'BOOM500':
            volatility = 0.02  # High volatility for Boom 500
        elif symbol == 'BOOM1000':
            volatility = 0.025  # Very high volatility for Boom 1000
        elif symbol == 'CRASH500':
            volatility = 0.015  # High volatility for Crash 500
        elif symbol == 'CRASH1000':
            volatility = 0.018  # High volatility for Crash 1000
        elif symbol == 'XAUUSD':
            volatility = 0.008  # Medium volatility for Gold
        elif symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
            volatility = 0.0002
        else:
            volatility = 0.005
        
        # Generate trending price with noise
        trend = np.linspace(0, np.random.randn() * price * 0.02, limit)
        noise = np.random.randn(limit) * price * volatility
        close_prices = price + trend + noise
        
        # Generate OHLC from close
        high_prices = close_prices + np.random.rand(limit) * price * volatility * 1.5
        low_prices = close_prices - np.random.rand(limit) * price * volatility * 1.5
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        # Add some realistic price action
        for i in range(1, len(close_prices)):
            high_prices[i] = max(open_prices[i], close_prices[i], high_prices[i])
            low_prices[i] = min(open_prices[i], close_prices[i], low_prices[i])
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': 0  # FX doesn't have volume
        })
        
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        
        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Stochastic
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        atr_14 = true_range.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr_14)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=14).mean()
        
        # Momentum indicators
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['roc'] = ((df['close'] - df['close'].shift(12)) / df['close'].shift(12)) * 100
        
        # Fill NaN values
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def identify_trend(self, df: pd.DataFrame) -> str:
        """Identify market trend"""
        latest = df.iloc[-1]
        conditions = []
        
        # EMA alignment
        if latest['ema_9'] > latest['ema_21'] > latest['sma_50']:
            conditions.append('bullish')
        elif latest['ema_9'] < latest['ema_21'] < latest['sma_50']:
            conditions.append('bearish')
        
        # ADX strength
        if latest['adx'] > 25:
            if latest['close'] > latest['sma_50']:
                conditions.append('bullish')
            else:
                conditions.append('bearish')
        
        # Momentum
        if latest['momentum'] > 0 and latest['roc'] > 0:
            conditions.append('bullish')
        elif latest['momentum'] < 0 and latest['roc'] < 0:
            conditions.append('bearish')
        
        # Price vs MA
        if latest['close'] > latest['sma_50'] > latest['sma_200']:
            conditions.append('bullish')
        elif latest['close'] < latest['sma_50'] < latest['sma_200']:
            conditions.append('bearish')
        
        bullish_count = conditions.count('bullish')
        bearish_count = conditions.count('bearish')
        
        if bullish_count > bearish_count:
            return 'STRONG_UPTREND' if bullish_count >= 3 else 'UPTREND'
        elif bearish_count > bullish_count:
            return 'STRONG_DOWNTREND' if bearish_count >= 3 else 'DOWNTREND'
        else:
            return 'SIDEWAYS'
    
    def calculate_confidence(self, signals: List[str], df: pd.DataFrame, timeframe_results: Dict) -> float:
        """Calculate confidence score (0-99%)"""
        
        latest = df.iloc[-1]
        score = 40  # Base score
        
        # Signal agreement (25 points)
        buy_signals = sum(1 for s in signals if 'BUY' in s.upper() or 'BULLISH' in s.upper() or 'oversold' in s.lower())
        sell_signals = sum(1 for s in signals if 'SELL' in s.upper() or 'BEARISH' in s.upper() or 'overbought' in s.lower())
        total_signals = len(signals)
        
        if total_signals > 0:
            agreement = max(buy_signals, sell_signals) / total_signals
            score += agreement * 25
            if agreement >= 0.85 and total_signals >= 5:
                score += 5
        
        # Multi-timeframe alignment (12 points)
        trends = [r['trend'] for r in timeframe_results.values()]
        bullish_trends = sum(1 for t in trends if 'UP' in t)
        bearish_trends = sum(1 for t in trends if 'DOWN' in t)
        strong_trends = sum(1 for t in trends if 'STRONG' in t)
        
        trend_alignment = max(bullish_trends, bearish_trends) / len(trends)
        score += trend_alignment * 12
        
        if strong_trends >= 1:
            score += 3
        
        # ADX strength (12 points)
        if latest['adx'] > 40:
            score += 12
        elif latest['adx'] > 30:
            score += 8
        elif latest['adx'] > 25:
            score += 5
        
        # RSI position (8 points)
        if buy_signals > sell_signals:
            if 30 < latest['rsi'] < 50:
                score += 8
            elif 25 < latest['rsi'] < 55:
                score += 5
            elif latest['rsi'] < 30:
                score += 10  # Oversold bonus
        else:
            if 50 < latest['rsi'] < 70:
                score += 8
            elif 45 < latest['rsi'] < 75:
                score += 5
            elif latest['rsi'] > 70:
                score += 10  # Overbought bonus
        
        # MACD momentum (8 points)
        macd_hist = abs(latest['macd_hist'])
        avg_macd_hist = df['macd_hist'].tail(20).abs().mean()
        
        if macd_hist > avg_macd_hist * 1.5:
            score += 8
        elif macd_hist > avg_macd_hist * 1.2:
            score += 5
        
        # Bollinger Band position (6 points)
        bb_range = latest['bb_upper'] - latest['bb_lower']
        if bb_range > 0:
            bb_position = (latest['close'] - latest['bb_lower']) / bb_range
            if buy_signals > sell_signals:
                if bb_position < 0.3:
                    score += 6
                elif bb_position < 0.4:
                    score += 3
            else:
                if bb_position > 0.7:
                    score += 6
                elif bb_position > 0.6:
                    score += 3
        
        # Stochastic position (5 points)
        if buy_signals > sell_signals:
            if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
                score += 5
            elif latest['stoch_k'] < 30:
                score += 3
        else:
            if latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
                score += 5
            elif latest['stoch_k'] > 70:
                score += 3
        
        # Momentum strength (6 points)
        if abs(latest['momentum']) > df['momentum'].tail(20).abs().mean() * 1.3:
            score += 6
        elif abs(latest['momentum']) > df['momentum'].tail(20).abs().mean():
            score += 3
        
        # MA alignment (8 points)
        if latest['ema_9'] > latest['ema_21'] > latest['sma_50'] > latest['sma_200']:
            score += 8
        elif latest['ema_9'] < latest['ema_21'] < latest['sma_50'] < latest['sma_200']:
            score += 8
        elif (latest['ema_9'] > latest['ema_21'] > latest['sma_50']) or \
             (latest['ema_9'] < latest['ema_21'] < latest['sma_50']):
            score += 5
        
        return min(score, 99)
    
    def calculate_risk_levels(self, df: pd.DataFrame, action: str) -> Tuple[float, float, float]:
        """Calculate stop loss and take profit levels using ATR"""
        
        latest = df.iloc[-1]
        atr = latest['atr']
        entry = latest['close']
        
        # Adjust multipliers for better risk/reward
        if 'BUY' in action:
            stop_loss = entry - (2 * atr)
            take_profit = entry + (4 * atr)
        else:  # SELL
            stop_loss = entry + (2 * atr)
            take_profit = entry - (4 * atr)
        
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        risk_reward = reward / risk if risk > 0 else 0
        
        return stop_loss, take_profit, risk_reward
    
    def analyze_single_timeframe(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Analyze single timeframe"""
        
        df = self.fetch_ohlcv(symbol, timeframe)
        
        if len(df) < 200:
            logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(df)} candles")
            return None
        
        df = self.calculate_indicators(df)
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        signal_score = 0
        
        # RSI signals
        if latest['rsi'] < 30:
            signals.append("RSI oversold")
            signal_score += 4
        elif latest['rsi'] > 70:
            signals.append("RSI overbought")
            signal_score -= 4
        elif 30 <= latest['rsi'] < 40:
            signals.append("RSI approaching oversold")
            signal_score += 2
        elif 60 < latest['rsi'] <= 70:
            signals.append("RSI approaching overbought")
            signal_score -= 2
        
        # MACD crossover
        if prev['macd'] <= prev['macd_signal'] and latest['macd'] > latest['macd_signal']:
            signals.append("MACD bullish crossover")
            signal_score += 5
            if latest['macd_hist'] > 0:
                signal_score += 1
        elif prev['macd'] >= prev['macd_signal'] and latest['macd'] < latest['macd_signal']:
            signals.append("MACD bearish crossover")
            signal_score -= 5
            if latest['macd_hist'] < 0:
                signal_score -= 1
        elif latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > prev['macd_hist']:
            signals.append("MACD bullish momentum")
            signal_score += 2
        elif latest['macd'] < latest['macd_signal'] and latest['macd_hist'] < prev['macd_hist']:
            signals.append("MACD bearish momentum")
            signal_score -= 2
        
        # EMA crossover
        if prev['ema_9'] <= prev['ema_21'] and latest['ema_9'] > latest['ema_21']:
            signals.append("EMA bullish crossover")
            signal_score += 4
        elif prev['ema_9'] >= prev['ema_21'] and latest['ema_9'] < latest['ema_21']:
            signals.append("EMA bearish crossover")
            signal_score -= 4
        
        # Price vs MA
        if latest['close'] > latest['sma_50'] > latest['sma_200']:
            signals.append("Price above key MAs")
            signal_score += 2
        elif latest['close'] < latest['sma_50'] < latest['sma_200']:
            signals.append("Price below key MAs")
            signal_score -= 2
        
        # Bollinger Bands
        if latest['close'] < latest['bb_lower']:
            signals.append("Below lower BB")
            signal_score += 3
        elif latest['close'] > latest['bb_upper']:
            signals.append("Above upper BB")
            signal_score -= 3
        elif prev['close'] <= prev['bb_lower'] and latest['close'] > latest['bb_lower']:
            signals.append("BB bounce from lower")
            signal_score += 3
        elif prev['close'] >= prev['bb_upper'] and latest['close'] < latest['bb_upper']:
            signals.append("BB rejection from upper")
            signal_score -= 3
        
        # Stochastic
        if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
            signals.append("Stochastic oversold")
            signal_score += 3
            if prev['stoch_k'] <= prev['stoch_d'] and latest['stoch_k'] > latest['stoch_d']:
                signals.append("Stochastic bullish cross")
                signal_score += 2
        elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
            signals.append("Stochastic overbought")
            signal_score -= 3
            if prev['stoch_k'] >= prev['stoch_d'] and latest['stoch_k'] < latest['stoch_d']:
                signals.append("Stochastic bearish cross")
                signal_score -= 2
        
        # ADX trend strength
        if latest['adx'] > 30:
            signals.append(f"Strong trend (ADX: {latest['adx']:.1f})")
        elif latest['adx'] > 25:
            signals.append(f"Trending (ADX: {latest['adx']:.1f})")
        
        # Momentum
        if latest['momentum'] > 0:
            signals.append("Positive momentum")
            signal_score += 1
        else:
            signals.append("Negative momentum")
            signal_score -= 1
        
        # ROC
        if abs(latest['roc']) > 2:
            if latest['roc'] > 0:
                signals.append("Strong positive ROC")
                signal_score += 2
            else:
                signals.append("Strong negative ROC")
                signal_score -= 2
        
        trend = self.identify_trend(df)
        
        return {
            'signals': signals,
            'score': signal_score,
            'trend': trend,
            'df': df
        }
    
    def analyze_symbol(self, symbol: str) -> Optional[TradeSignal]:
        """Multi-timeframe analysis for a symbol"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing {symbol}...")
        logger.info(f"{'='*60}")
        
        try:
            timeframe_results = {}
            for tf in self.timeframes:
                result = self.analyze_single_timeframe(symbol, tf)
                if result:
                    timeframe_results[tf] = result
                    logger.info(f"{symbol} {tf}: Score={result['score']}, Trend={result['trend']}, Signals={len(result['signals'])}")
                else:
                    logger.warning(f"No result for {symbol} {tf}")
                
                # Rate limiting between API calls
                time.sleep(13)
            
            if not timeframe_results:
                logger.warning(f"No valid timeframe results for {symbol}")
                return None
            
            # Use primary timeframe (1h) or first available
            primary_tf = '1h' if '1h' in timeframe_results else list(timeframe_results.keys())[0]
            primary_result = timeframe_results[primary_tf]
            
            df = primary_result['df']
            latest = df.iloc[-1]
            
            # Calculate average score across timeframes
            total_score = sum(r['score'] for r in timeframe_results.values())
            avg_score = total_score / len(timeframe_results)
            
            # Analyze trends across timeframes
            trends = [r['trend'] for r in timeframe_results.values()]
            bullish_trends = sum(1 for t in trends if 'UP' in t)
            bearish_trends = sum(1 for t in trends if 'DOWN' in t)
            strong_trends = sum(1 for t in trends if 'STRONG' in t)
            
            logger.info(f"{symbol} - Avg Score: {avg_score:.2f}, Bullish TFs: {bullish_trends}, Bearish TFs: {bearish_trends}, Strong: {strong_trends}")
            
            # Determine action based on scores and trends
            action = None
            
            # Adjusted thresholds for more aggressive signal generation (70% confidence)
            if avg_score >= 4 and bullish_trends >= 1:
                action = "STRONG BUY"
            elif avg_score >= 2 and bullish_trends >= 1:
                action = "BUY"
            elif avg_score <= -4 and bearish_trends >= 1:
                action = "STRONG SELL"
            elif avg_score <= -2 and bearish_trends >= 1:
                action = "SELL"
            else:
                logger.info(f"{symbol}: No clear signal (score: {avg_score:.2f})")
                return None
            
            # Collect all signals from all timeframes
            all_signals = []
            for r in timeframe_results.values():
                all_signals.extend(r['signals'])
            
            # Calculate confidence
            confidence = self.calculate_confidence(all_signals, df, timeframe_results)
            
            logger.info(f"{symbol}: Action={action}, Confidence={confidence:.1f}%")
            
            if confidence < self.min_confidence:
                logger.info(f"{symbol}: Confidence {confidence:.1f}% below threshold {self.min_confidence}%")
                return None
            
            # Calculate risk levels
            stop_loss, take_profit, risk_reward = self.calculate_risk_levels(df, action)
            
            # Create trade signal
            signal = TradeSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                entry_price=latest['close'],
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                timeframe=primary_tf,
                indicators={
                    'rsi': latest['rsi'],
                    'macd': latest['macd'],
                    'adx': latest['adx'],
                    'trend': primary_result['trend'],
                    'momentum': 'BULLISH' if avg_score > 0 else 'BEARISH',
                    'score': avg_score
                },
                timestamp=datetime.now()
            )
            
            logger.info(f"✅ SIGNAL GENERATED: {signal.symbol} {signal.action} @ {signal.entry_price:.5f}")
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}", exc_info=True)
            return None
    
    def run_analysis(self):
        """Run analysis on all symbols - SINGLE EXECUTION"""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🤖 TRADING BOT STARTED - GitHub Actions Mode")
        logger.info(f"{'='*70}")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Timeframes: {', '.join(self.timeframes)}")
        logger.info(f"Min Confidence: {self.min_confidence}%")
        logger.info(f"{'='*70}\n")
        
        # Send startup notification
        self.notifier.send_startup(self.symbols, self.min_confidence, self.timeframes)
        
        signals_found = 0
        signals_list = []
        
        for i, symbol in enumerate(self.symbols, 1):
            try:
                logger.info(f"\n[{i}/{len(self.symbols)}] Processing {symbol}...")
                
                signal = self.analyze_symbol(symbol)
                
                if signal:
                    logger.info(f"✅ HIGH CONFIDENCE SIGNAL: {symbol} {signal.action} ({signal.confidence:.1f}%)")
                    
                    # Send to Telegram
                    if self.notifier.send_signal(signal):
                        signals_found += 1
                        signals_list.append(signal)
                    
                    time.sleep(2)  # Delay between Telegram messages
                else:
                    logger.info(f"ℹ️ No high-confidence signal for {symbol}")
                
                # Add delay between symbols to avoid rate limits
                if i < len(self.symbols):
                    time.sleep(3)
                    
            except Exception as e:
                error_msg = f"Error processing {symbol}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.notifier.send_error(error_msg)
        
        # Send summary
        summary_msg = self._generate_summary(signals_found, signals_list)
        self.notifier.send_message(summary_msg)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ ANALYSIS COMPLETE")
        logger.info(f"Symbols Analyzed: {len(self.symbols)}")
        logger.info(f"Signals Generated: {signals_found}")
        logger.info(f"API Calls Used: {self.api_call_count}/{self.max_api_calls}")
        logger.info(f"{'='*70}\n")
        
        return signals_found
    
    def _generate_summary(self, signals_found: int, signals_list: List[TradeSignal]) -> str:
        """Generate analysis summary message"""
        
        summary = f"""
📊 <b>Hourly Analysis Complete</b>

<b>Summary:</b>
• Symbols Analyzed: {len(self.symbols)}
• Signals Found: {signals_found}
• Confidence Threshold: {self.min_confidence}%
• API Calls Used: {self.api_call_count}/{self.max_api_calls}

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if signals_list:
            summary += "\n<b>Signals Generated:</b>\n"
            for signal in signals_list:
                emoji = "🟢" if "BUY" in signal.action else "🔴"
                summary += f"{emoji} {signal.symbol}: {signal.action} ({signal.confidence:.1f}%)\n"
        else:
            summary += "\n⚪ No high-confidence signals detected this hour.\n"
        
        summary += "\n⏰ Next analysis in 1 hour"
        
        return summary
    
    def save_signal_to_log(self, signal: TradeSignal):
        """Save signal to JSON log file"""
        try:
            log_file = 'signals_log.json'
            
            # Load existing logs
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Append new signal
            logs.append(signal.to_dict())
            
            # Save back to file
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            logger.info(f"Signal saved to {log_file}")
            
        except Exception as e:
            logger.error(f"Error saving signal to log: {str(e)}")


def validate_environment():
    """Validate environment and configuration"""
    logger.info("Validating environment...")
    
    # Check Python version
    import sys
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        logger.error(f"Python 3.7+ required. Current: {python_version.major}.{python_version.minor}")
        return False
    
    # Check required packages
    required_packages = ['pandas', 'numpy', 'requests']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Install with: pip install pandas numpy requests")
        return False
    
    logger.info("✅ Environment validation passed")
    return True


if __name__ == "__main__":
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║       ADVANCED FOREX & INDICES TRADING BOT v2.0                 ║
║         70% Confidence Threshold - Hourly Execution             ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Validate environment
    if not validate_environment():
        print("\n❌ Environment validation failed. Exiting...")
        exit(1)
    
    # Get configuration from environment or use hardcoded values
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7950477685:AAEexbQXDHZ2UHzYJmO_TCrFFlHE__Umicw')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '5490682482')
    ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', 'DUZ125XKRQF0RKD0')
    
    # Analyze XAUUSD and BOOM1000 every hour
    symbols_str = os.getenv('TRADING_SYMBOLS', 'XAUUSD,BOOM1000')
    SYMBOLS = [s.strip() for s in symbols_str.split(',') if s.strip()]
    
    logger.info(f"Analyzing: {', '.join(SYMBOLS)}")
    print(f"📊 Analyzing: {', '.join(SYMBOLS)} (runs every hour)")
    
    # Get minimum confidence (default to 70%)
    try:
        MIN_CONFIDENCE = float(os.getenv('MIN_CONFIDENCE', '70'))
    except ValueError:
        MIN_CONFIDENCE = 70.0
    
    print(f"""
Configuration:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Symbols:          {', '.join(SYMBOLS)}
  Timeframe:        1h only
  Min Confidence:   {MIN_CONFIDENCE}% ⚡ (More Aggressive)
  Execution Mode:   Every hour
  API Provider:     Alpha Vantage (XAUUSD) / Demo (BOOM1000)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Starting analysis...
""")
    
    try:
        # Create bot instance
        bot = AdvancedTradingBot(
            symbols=SYMBOLS,
            telegram_token=TELEGRAM_BOT_TOKEN,
            telegram_chat_id=TELEGRAM_CHAT_ID,
            api_key=ALPHA_VANTAGE_KEY
        )
        
        # Set minimum confidence
        bot.min_confidence = MIN_CONFIDENCE
        
        # Run single analysis
        signals_count = bot.run_analysis()
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ✅ EXECUTION COMPLETE                         ║
╚══════════════════════════════════════════════════════════════════╝

Results:
  • Signals Generated: {signals_count}
  • Check Telegram for details
  • Logs saved to: trading_bot.log

Next execution in 1 hour (via GitHub Actions)
""")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Bot stopped by user")
        logger.info("Bot stopped by user (KeyboardInterrupt)")
        
    except Exception as e:
        error_msg = f"❌ CRITICAL ERROR: {str(e)}"
        print(f"\n{error_msg}")
        logger.error(error_msg, exc_info=True)
        
        # Try to send error notification
        try:
            notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
            notifier.send_error(str(e))
        except:
            logger.error("Failed to send error notification to Telegram")
        
        exit(1)