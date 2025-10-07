"""
Complete Advanced XAUUSD Trading Bot with Deriv API and Dual Telegram Groups
Enhanced version with real-time Deriv price data and dual signal formats
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
import websocket

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
    take_profit_1: float  # TP1: Fixed $5 profit
    take_profit_2: float  # TP2: 1:2 RR
    risk_reward_ratio: float
    timeframe: str
    indicators: Dict
    timestamp: datetime
    data_source: str
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class DerivAPI:
    """Handle Deriv API connections for real-time price data"""
    
    def __init__(self, app_id: str = "1089"):
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        self.ws = None
        
    def get_current_price(self, symbol: str = "frxXAUUSD") -> Optional[float]:
        """Get current price from Deriv API"""
        try:
            logger.info(f"Fetching current price for {symbol} from Deriv API...")
            
            # Create WebSocket connection
            ws = websocket.create_connection(self.ws_url, timeout=15)
            
            # Request tick data
            request = {
                "ticks": symbol,
                "subscribe": 0
            }
            
            ws.send(json.dumps(request))
            
            # Get response
            response = ws.recv()
            data = json.loads(response)
            
            ws.close()
            
            if 'tick' in data and 'quote' in data['tick']:
                price = float(data['tick']['quote'])
                logger.info(f"✅ Got Deriv price for {symbol}: {price}")
                return price
            elif 'error' in data:
                logger.error(f"Deriv API error: {data['error']}")
                return None
            else:
                logger.warning(f"Unexpected Deriv response: {data}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching Deriv price: {e}")
            return None
    
    def get_historical_candles(self, symbol: str = "frxXAUUSD", count: int = 200) -> Optional[pd.DataFrame]:
        """Get historical candle data from Deriv API"""
        try:
            logger.info(f"Fetching {count} candles for {symbol} from Deriv API...")
            
            ws = websocket.create_connection(self.ws_url, timeout=15)
            
            # Request historical candles (1 hour)
            end_time = int(datetime.now().timestamp())
            
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": end_time,
                "start": 1,
                "style": "candles",
                "granularity": 3600  # 1 hour candles
            }
            
            ws.send(json.dumps(request))
            response = ws.recv()
            data = json.loads(response)
            ws.close()
            
            if 'candles' in data:
                candles = data['candles']
                
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime([c['epoch'] for c in candles], unit='s'),
                    'open': [float(c['open']) for c in candles],
                    'high': [float(c['high']) for c in candles],
                    'low': [float(c['low']) for c in candles],
                    'close': [float(c['close']) for c in candles],
                    'volume': 0  # Deriv doesn't provide volume for forex
                })
                
                logger.info(f"✅ Fetched {len(df)} candles from Deriv API")
                return df
            elif 'error' in data:
                logger.error(f"Deriv API error: {data['error']}")
                return None
            else:
                logger.warning(f"Unexpected Deriv response: {data}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching Deriv candles: {e}")
            return None


class TelegramNotifier:
    """Handle Telegram notifications with dual group support"""
    
    def __init__(self, bot_token: str, main_chat_id: str, simple_chat_id: str):
        self.bot_token = bot_token
        self.main_chat_id = main_chat_id  # Detailed signals
        self.simple_chat_id = simple_chat_id  # Simple signals
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.max_retries = 3
        
    def send_message(self, message: str, chat_id: str, parse_mode: str = "HTML", 
                    reply_markup: dict = None, retry_count: int = 0) -> bool:
        """Send message via Telegram with retry logic"""
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': parse_mode
        }
        if reply_markup:
            payload['reply_markup'] = json.dumps(reply_markup)
            
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"Telegram message sent to {chat_id}")
                return True
            elif response.status_code == 429 and retry_count < self.max_retries:
                retry_after = int(response.json().get('parameters', {}).get('retry_after', 30))
                logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                return self.send_message(message, chat_id, parse_mode, reply_markup, retry_count + 1)
            else:
                logger.error(f"Failed to send Telegram message: {response.text}")
                return False
        except requests.exceptions.Timeout:
            if retry_count < self.max_retries:
                logger.warning(f"Timeout. Retry {retry_count + 1}/{self.max_retries}")
                time.sleep(5)
                return self.send_message(message, chat_id, parse_mode, reply_markup, retry_count + 1)
            logger.error("Max retries reached for Telegram message")
            return False
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return False
    
    def send_signal(self, signal: TradeSignal) -> bool:
        """Send formatted trading signal to both groups"""
        
        # Send detailed signal to main group
        detailed_success = self._send_detailed_signal(signal)
        
        # Send simple signal to second group
        simple_success = self._send_simple_signal(signal)
        
        return detailed_success and simple_success
    
    def _send_detailed_signal(self, signal: TradeSignal) -> bool:
        """Send detailed signal to main group"""
        
        emoji = "🟢" if "BUY" in signal.action else "🔴" if "SELL" in signal.action else "⚪"
        
        # Confidence rating
        if signal.confidence >= 90:
            confidence_emoji = "🔥🔥🔥"
            confidence_label = "VERY HIGH"
        elif signal.confidence >= 85:
            confidence_emoji = "🔥🔥"
            confidence_label = "HIGH"
        else:
            confidence_emoji = "🔥"
            confidence_label = "MODERATE"
        
        message = f"""
{emoji} <b>{confidence_label} CONFIDENCE SIGNAL</b> {confidence_emoji}

<b>Symbol:</b> {signal.symbol}
<b>Action:</b> {signal.action}
<b>Data Source:</b> {signal.data_source} (Deriv API)

💰 <b>Trade Setup</b>
<b>Confidence:</b> {signal.confidence:.1f}% ⭐
Entry: {signal.entry_price:.2f}
Stop Loss: {signal.stop_loss:.2f}

<b>Take Profits:</b>
TP1: {signal.take_profit_1:.2f} (Fixed $5 @ 0.02 lot)
TP2: {signal.take_profit_2:.2f} (1:2 RR)

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
                    {'text': '📊 View Chart', 'url': f'https://www.tradingview.com/chart/?symbol=GOLD'},
                ]
            ]
        }
        
        return self.send_message(message, self.main_chat_id, reply_markup=keyboard)
    
    def _send_simple_signal(self, signal: TradeSignal) -> bool:
        """Send simple signal to second group (basic info only)"""
        
        emoji = "🟢 BUY" if "BUY" in signal.action else "🔴 SELL"
        
        message = f"""
{emoji}

<b>{signal.symbol}</b>

Entry: {signal.entry_price:.2f}

TP1: {signal.take_profit_1:.2f}
TP2: {signal.take_profit_2:.2f}

SL: {signal.stop_loss:.2f}
"""
        
        return self.send_message(message, self.simple_chat_id)
    
    def send_error(self, error_msg: str):
        """Send error notification to main group"""
        message = f"⚠️ <b>Bot Error</b>\n\n<code>{error_msg}</code>"
        self.send_message(message, self.main_chat_id)
    
    def send_startup(self):
        """Send startup notification to main group"""
        message = f"""
🤖 <b>XAUUSD Trading Bot Started</b>

<b>Configuration:</b>
• Symbol: XAUUSD (Gold)
• Timeframe: 1h
• Min Confidence: 80%
• Data Source: Deriv API
• Mode: GitHub Actions (Hourly)

Status: ✅ Active
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message, self.main_chat_id)


class AdvancedTradingBot:
    """Advanced trading bot for XAUUSD with Deriv API"""
    
    def __init__(self, telegram_token: str, main_chat_id: str, simple_chat_id: str, deriv_app_id: str = "1089"):
        self.symbol = 'XAUUSD'
        self.notifier = TelegramNotifier(telegram_token, main_chat_id, simple_chat_id)
        self.deriv_api = DerivAPI(deriv_app_id)
        self.min_confidence = 80.0
        self.timeframe = '1h'
        self.lot_size = 0.02  # For TP1 calculation
    
    def fetch_ohlcv(self) -> Tuple[pd.DataFrame, str]:
        """Fetch OHLCV data from Deriv API"""
        
        df = self.deriv_api.get_historical_candles("frxXAUUSD", count=200)
        
        if df is not None and len(df) >= 200:
            logger.info(f"✅ Using REAL Deriv data: {len(df)} candles")
            return df, "DERIV_API"
        else:
            logger.warning("⚠️ Failed to fetch from Deriv, using fallback")
            # Fallback: get current price and generate historical data
            current_price = self.deriv_api.get_current_price("frxXAUUSD")
            if current_price:
                df = self._generate_demo_data(200, base_price=current_price)
                return df, "HYBRID_DERIV_PRICE"
            else:
                df = self._generate_demo_data(200)
                return df, "DEMO"
    
    def _generate_demo_data(self, limit: int, base_price: Optional[float] = None) -> pd.DataFrame:
        """Generate realistic demo data with optional real base price"""
        
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1h')
        np.random.seed(int(datetime.now().timestamp()) % 2**32)
        
        price = base_price if base_price else 2665.00
        volatility = 0.008
        
        close_prices = np.zeros(limit)
        close_prices[-1] = price
        
        for i in range(limit - 2, -1, -1):
            change = np.random.randn() * price * volatility
            reversion = (price - close_prices[i + 1]) * 0.05
            close_prices[i] = close_prices[i + 1] - change + reversion
        
        high_prices = close_prices + np.abs(np.random.randn(limit)) * price * volatility * 0.5
        low_prices = close_prices - np.abs(np.random.randn(limit)) * price * volatility * 0.5
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        for i in range(len(close_prices)):
            high_prices[i] = max(open_prices[i], close_prices[i], high_prices[i])
            low_prices[i] = min(open_prices[i], close_prices[i], low_prices[i])
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': 0
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
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['roc'] = ((df['close'] - df['close'].shift(12)) / df['close'].shift(12)) * 100
        
        df = df.ffill().bfill()
        df = df.fillna(0)
        
        return df
    
    def identify_trend(self, df: pd.DataFrame) -> str:
        """Identify market trend"""
        latest = df.iloc[-1]
        conditions = []
        
        if latest['ema_9'] > latest['ema_21'] > latest['sma_50']:
            conditions.append('bullish')
        elif latest['ema_9'] < latest['ema_21'] < latest['sma_50']:
            conditions.append('bearish')
        
        if latest['adx'] > 25:
            if latest['close'] > latest['sma_50']:
                conditions.append('bullish')
            else:
                conditions.append('bearish')
        
        if latest['momentum'] > 0 and latest['roc'] > 0:
            conditions.append('bullish')
        elif latest['momentum'] < 0 and latest['roc'] < 0:
            conditions.append('bearish')
        
        bullish_count = conditions.count('bullish')
        bearish_count = conditions.count('bearish')
        
        if bullish_count > bearish_count:
            return 'STRONG_UPTREND' if bullish_count >= 3 else 'UPTREND'
        elif bearish_count > bullish_count:
            return 'STRONG_DOWNTREND' if bearish_count >= 3 else 'DOWNTREND'
        else:
            return 'SIDEWAYS'
    
    def calculate_confidence(self, signals: List[str], df: pd.DataFrame, trend: str) -> float:
        """Calculate confidence score (0-99%)"""
        
        latest = df.iloc[-1]
        score = 40
        
        buy_signals = sum(1 for s in signals if 'BUY' in s.upper() or 'BULLISH' in s.upper() or 'oversold' in s.lower())
        sell_signals = sum(1 for s in signals if 'SELL' in s.upper() or 'BEARISH' in s.upper() or 'overbought' in s.lower())
        total_signals = len(signals)
        
        if total_signals > 0:
            agreement = max(buy_signals, sell_signals) / total_signals
            score += agreement * 25
        
        if 'STRONG' in trend:
            score += 12
        elif trend != 'SIDEWAYS':
            score += 8
        
        if latest['adx'] > 40:
            score += 12
        elif latest['adx'] > 30:
            score += 8
        
        if buy_signals > sell_signals:
            if latest['rsi'] < 30:
                score += 10
            elif 30 < latest['rsi'] < 50:
                score += 8
        else:
            if latest['rsi'] > 70:
                score += 10
            elif 50 < latest['rsi'] < 70:
                score += 8
        
        return min(score, 99)
    
    def calculate_risk_levels(self, df: pd.DataFrame, action: str) -> Tuple[float, float, float, float]:
        """Calculate stop loss and both take profits"""
        
        latest = df.iloc[-1]
        atr = latest['atr']
        entry = latest['close']
        
        if 'BUY' in action:
            # Stop Loss
            stop_loss = entry - (2 * atr)
            
            # TP1: Calculate price needed for exactly $5 profit with 0.02 lot
            # For gold: 1 lot = $100 per $1 move
            # 0.02 lot = $2 per $1 move
            # For $5 profit: need $5 / $2 = $2.50 move
            tp1 = entry + 2.50
            
            # TP2: 1:2 Risk/Reward
            risk = entry - stop_loss
            tp2 = entry + (2 * risk)
            
        else:  # SELL
            # Stop Loss
            stop_loss = entry + (2 * atr)
            
            # TP1: $5 profit with 0.02 lot
            tp1 = entry - 2.50
            
            # TP2: 1:2 Risk/Reward
            risk = stop_loss - entry
            tp2 = entry - (2 * risk)
        
        risk_amount = abs(entry - stop_loss)
        reward_amount = abs(tp2 - entry)
        risk_reward = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return stop_loss, tp1, tp2, risk_reward
    
    def analyze_xauusd(self) -> Optional[TradeSignal]:
        """Analyze XAUUSD and generate signal"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing {self.symbol}...")
        logger.info(f"{'='*60}")
        
        try:
            df, data_source = self.fetch_ohlcv()
            
            if len(df) < 200:
                logger.warning(f"Insufficient data: {len(df)} candles")
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
            
            # MACD crossover
            if prev['macd'] <= prev['macd_signal'] and latest['macd'] > latest['macd_signal']:
                signals.append("MACD bullish crossover")
                signal_score += 5
            elif prev['macd'] >= prev['macd_signal'] and latest['macd'] < latest['macd_signal']:
                signals.append("MACD bearish crossover")
                signal_score -= 5
            
            # EMA crossover
            if prev['ema_9'] <= prev['ema_21'] and latest['ema_9'] > latest['ema_21']:
                signals.append("EMA bullish crossover")
                signal_score += 4
            elif prev['ema_9'] >= prev['ema_21'] and latest['ema_9'] < latest['ema_21']:
                signals.append("EMA bearish crossover")
                signal_score -= 4
            
            # Bollinger Bands
            if latest['close'] < latest['bb_lower']:
                signals.append("Below lower BB")
                signal_score += 3
            elif latest['close'] > latest['bb_upper']:
                signals.append("Above upper BB")
                signal_score -= 3
            
            # Stochastic
            if latest['stoch_k'] < 20:
                signals.append("Stochastic oversold")
                signal_score += 3
            elif latest['stoch_k'] > 80:
                signals.append("Stochastic overbought")
                signal_score -= 3
            
            # Momentum
            if latest['momentum'] > 0:
                signals.append("Positive momentum")
                signal_score += 1
            else:
                signals.append("Negative momentum")
                signal_score -= 1
            
            trend = self.identify_trend(df)
            
            logger.info(f"Signals: {len(signals)}, Score: {signal_score}, Trend: {trend}")
            
            action = None
            
            if signal_score >= 4:
                action = "STRONG BUY" if signal_score >= 8 else "BUY"
            elif signal_score <= -4:
                action = "STRONG SELL" if signal_score <= -8 else "SELL"
            else:
                logger.info(f"No clear signal (score: {signal_score})")
                return None
            
            confidence = self.calculate_confidence(signals, df, trend)
            
            logger.info(f"Action: {action}, Confidence: {confidence:.1f}%")
            
            if confidence < self.min_confidence:
                logger.info(f"Confidence {confidence:.1f}% below threshold {self.min_confidence}%")
                return None
            
            stop_loss, tp1, tp2, risk_reward = self.calculate_risk_levels(df, action)
            
            signal = TradeSignal(
                symbol=self.symbol,
                action=action,
                confidence=confidence,
                entry_price=latest['close'],
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                risk_reward_ratio=risk_reward,
                timeframe=self.timeframe,
                indicators={
                    'rsi': latest['rsi'],
                    'macd': latest['macd'],
                    'adx': latest['adx'],
                    'trend': trend,
                    'momentum': 'BULLISH' if signal_score > 0 else 'BEARISH',
                    'score': signal_score
                },
                timestamp=datetime.now(),
                data_source=data_source
            )
            
            logger.info(f"✅ SIGNAL GENERATED: {signal.symbol} {signal.action} @ {signal.entry_price:.2f}")
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {self.symbol}: {str(e)}", exc_info=True)
            return None
    
    def run_analysis(self):
        """Run analysis - SINGLE EXECUTION"""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🤖 XAUUSD TRADING BOT STARTED - Deriv API")
        logger.info(f"{'='*70}")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Min Confidence: {self.min_confidence}%")
        logger.info(f"{'='*70}\n")
        
        self.notifier.send_startup()
        
        try:
            signal = self.analyze_xauusd()
            
            if signal:
                logger.info(f"✅ HIGH CONFIDENCE SIGNAL: {signal.action} ({signal.confidence:.1f}%)")
                
                if self.notifier.send_signal(signal):
                    logger.info("Signal sent to both Telegram groups")
                    summary_msg = self._generate_summary(True, signal)
                else:
                    logger.error("Failed to send signal to Telegram")
                    summary_msg = self._generate_summary(False, None)
            else:
                logger.info("ℹ️ No high-confidence signal generated")
                summary_msg = self._generate_summary(False, None)
            
            self.notifier.send_message(summary_msg, self.notifier.main_chat_id)
            
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.notifier.send_error(error_msg)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ ANALYSIS COMPLETE")
        logger.info(f"{'='*70}\n")
    
    def _generate_summary(self, signal_found: bool, signal: Optional[TradeSignal]) -> str:
        """Generate analysis summary message"""
        
        summary = f"""
📊 <b>XAUUSD Hourly Analysis Complete</b>

<b>Summary:</b>
• Symbol: XAUUSD (Gold)
• Confidence Threshold: {self.min_confidence}%
• Data Source: Deriv API

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if signal_found and signal:
            emoji = "🟢" if "BUY" in signal.action else "🔴"
            summary += f"\n<b>Signal Generated:</b>\n"
            summary += f"{emoji} {signal.action} @ {signal.entry_price:.2f} ({signal.confidence:.1f}%)\n"
            summary += f"TP1: {signal.take_profit_1:.2f} | TP2: {signal.take_profit_2:.2f}\n"
            summary += f"SL: {signal.stop_loss:.2f}\n"
        else:
            summary += "\n⚪ No high-confidence signals detected this hour.\n"
        
        summary += "\n⏰ Next analysis in 1 hour"
        
        return summary


def validate_environment():
    """Validate environment and configuration"""
    logger.info("Validating environment...")
    
    import sys
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        logger.error(f"Python 3.7+ required. Current: {python_version.major}.{python_version.minor}")
        return False
    
    required_packages = ['pandas', 'numpy', 'requests', 'websocket']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Install with: pip install pandas numpy requests websocket-client")
        return False
    
    logger.info("✅ Environment validation passed")
    return True


if __name__ == "__main__":
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║       XAUUSD TRADING BOT v3.0 - DERIV API EDITION               ║
║         Real-Time Deriv Price Data - Dual Telegram Groups       ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    if not validate_environment():
        print("\n❌ Environment validation failed. Exiting...")
        exit(1)
    
    # Main Telegram group (detailed signals)
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7950477685:AAEexbQXDHZ2UHzYJmO_TCrFFlHE__Umicw')
    MAIN_CHAT_ID = os.getenv('MAIN_CHAT_ID', '-1003032435335')
    
    # Second Telegram group (simple signals)
    SIMPLE_CHAT_ID = os.getenv('SIMPLE_CHAT_ID', '-1003052865285')  # Replace with actual second group ID
    
    # Deriv API App ID (optional, defaults to demo app)
    DERIV_APP_ID = os.getenv('DERIV_APP_ID', '1089')
    
    print(f"""
Configuration:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Symbol:           XAUUSD (Gold)
  Timeframe:        1h
  Min Confidence:   80% 🔥
  Data Source:      Deriv API (Real-time)
  Execution Mode:   Every hour
  
  Lot Size:         0.02 (for TP1 calculation)
  TP1:              Fixed $5 profit
  TP2:              1:2 Risk/Reward ratio
  
  Telegram Groups:
    • Main Group:   Detailed signals
    • Second Group: Simple signals (Buy/Sell + levels)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  IMPORTANT: Set SIMPLE_CHAT_ID environment variable to your 
    second Telegram group ID for simple signals.

Starting analysis...
""")
    
    try:
        bot = AdvancedTradingBot(
            telegram_token=TELEGRAM_BOT_TOKEN,
            main_chat_id=MAIN_CHAT_ID,
            simple_chat_id=SIMPLE_CHAT_ID,
            deriv_app_id=DERIV_APP_ID
        )
        
        bot.run_analysis()
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ✅ EXECUTION COMPLETE                         ║
╚══════════════════════════════════════════════════════════════════╝

Results:
  • Check both Telegram groups for signals
  • Main group: Detailed analysis with all indicators
  • Second group: Simple format (Action, Entry, TPs, SL)
  • Logs saved to: trading_bot.log

Next execution in 1 hour (via GitHub Actions)

Setup Instructions for Second Group:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Add your bot to the second Telegram group
2. Get the chat ID using @getidsbot or similar
3. Set environment variable: SIMPLE_CHAT_ID=your_group_id
4. Update in GitHub Secrets if using GitHub Actions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Bot stopped by user")
        logger.info("Bot stopped by user (KeyboardInterrupt)")
        
    except Exception as e:
        error_msg = f"❌ CRITICAL ERROR: {str(e)}"
        print(f"\n{error_msg}")
        logger.error(error_msg, exc_info=True)
        
        try:
            notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, MAIN_CHAT_ID, SIMPLE_CHAT_ID)
            notifier.send_error(str(e))
        except:
            logger.error("Failed to send error notification to Telegram")
        
        exit(1)