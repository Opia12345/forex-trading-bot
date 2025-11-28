"""
SYNTHETIC INDICES TRADING BOT v4.0 - SNIPER EDITION
Specialized for Deriv Volatility, Step, and Jump Indices.
Upgraded to M15 Timeframe Analysis for Institutional Accuracy.
"""

import os
import sys
import logging
from typing import Optional, List, Tuple, Dict
from datetime import datetime
from collections import deque
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import pytz
import requests
import json
import asyncio
import websockets

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('synthetic_bot_v4.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Data Classes ---
class SignalQuality(Enum):
    PREMIUM = "PREMIUM (A+)"
    HIGH = "HIGH (A)"
    MODERATE = "MODERATE (B)"

@dataclass
class TradeSignal:
    signal_id: str
    symbol: str
    action: str
    confidence: float
    quality: SignalQuality
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    stop_loss_distance: float
    risk_reward_ratio: float
    atr_value: float
    trend_state: str
    timestamp: datetime
    strategy_components: List[str] = field(default_factory=list)

# ============================================================================
# TELEGRAM NOTIFIER
# ============================================================================
class TelegramNotifier:
    def __init__(self, token: str, main_chat_id: str):
        self.token = token
        self.main_chat_id = main_chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_signal(self, signal: TradeSignal) -> bool:
        try:
            emoji_map = {
                "PREMIUM (A+)": "💎",
                "HIGH (A)": "🔥",
                "MODERATE (B)": "⚠️"
            }
            q_emoji = emoji_map.get(signal.quality.value, "✨")
            a_emoji = "🟢 BUY" if signal.action == "BUY" else "🔴 SELL"
            
            message = f"""
{q_emoji} <b>SNIPER SIGNAL v4.0</b> {q_emoji}

{a_emoji} <b>{signal.symbol}</b>
⏳ <b>Timeframe:</b> M15 (Intraday)
🛡️ <b>Confidence:</b> {signal.confidence:.1f}% ({signal.quality.value})

<b>📍 ENTRY & EXITS</b>
💵 Entry: <code>{signal.entry_price:.4f}</code>
🛑 Stop Loss: <code>{signal.stop_loss:.4f}</code>
🎯 TP1: <code>{signal.take_profit_1:.4f}</code> (Secure)
🎯 TP2: <code>{signal.take_profit_2:.4f}</code> (Runner)
🎯 TP3: <code>{signal.take_profit_3:.4f}</code> (Moonbag)

<b>📊 TECHNICAL CONFLUENCE</b>
"""
            for component in signal.strategy_components:
                message += f"• {component}\n"

            message += f"""
<b>💼 RISK DATA</b>
• Risk:Reward: 1:{signal.risk_reward_ratio:.2f}
• Volatility (ATR): {signal.atr_value:.4f}
• Trend: {signal.trend_state}

<i>⚠️ Manage your risk. Suggested Lot: Minimal.</i>
<i>ID: {signal.signal_id}</i>
"""
            return self._send_message(message)
        except Exception as e:
            logger.error(f"Error sending signal: {e}")
            return False
    
    def _send_message(self, message: str) -> bool:
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {"chat_id": self.main_chat_id, "text": message, "parse_mode": "HTML"}
            requests.post(url, json=payload, timeout=10)
            return True
        except Exception:
            return False

# ============================================================================
# DERIV DATA FETCHER (Upgraded to CANDLES)
# ============================================================================
class DerivDataFetcher:
    """Fetch M15 Candle data from Deriv"""
    
    # Expanded Symbol Map
    SYMBOLS = {
        # Standard Volatility
        'V10': 'R_10', 'V25': 'R_25', 'V50': 'R_50', 'V75': 'R_75', 'V100': 'R_100',
        # 1-Second Volatility (Faster)
        'V10(1s)': '1HZ10V', 'V25(1s)': '1HZ25V', 'V50(1s)': '1HZ50V', 
        'V75(1s)': '1HZ75V', 'V100(1s)': '1HZ100V',
        # Jump Indices (High Volatility)
        'JUMP10': 'JD10', 'JUMP25': 'JD25', 'JUMP50': 'JD50',
        # Step Index
        'STEP': 'STEP'
    }
    
    def __init__(self, app_id: str = "1089"):
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    
    async def _fetch_candles_async(self, symbol: str, count: int = 300) -> Optional[List]:
        """Fetch 15-minute candles (900 seconds)"""
        try:
            async with websockets.connect(self.ws_url, ping_interval=30, close_timeout=10) as ws:
                request = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": count,
                    "end": "latest",
                    "start": 1,
                    "style": "candles",
                    "granularity": 900 # 900 seconds = 15 Minutes
                }
                
                await ws.send(json.dumps(request))
                response = await asyncio.wait_for(ws.recv(), timeout=20)
                data = json.loads(response)
                
                if 'candles' in data:
                    return data['candles']
                return None
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_candle_data(self, symbol: str) -> Optional[pd.DataFrame]:
        deriv_symbol = self.SYMBOLS.get(symbol)
        if not deriv_symbol: return None
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                candles = loop.run_until_complete(self._fetch_candles_async(deriv_symbol))
            finally:
                loop.close()
            
            if not candles: return None
            
            df = pd.DataFrame(candles)
            # Rename columns to standard format
            df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'epoch': 'time'})
            
            # Convert types
            cols = ['open', 'high', 'low', 'close']
            df[cols] = df[cols].astype(float)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
        except Exception as e:
            logger.error(f"Data conversion error: {e}")
            return None

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================
class TechnicalIndicators:
    
    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return pd.Series(true_range).rolling(window=period).mean()
    
    @staticmethod
    def calculate_macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        exp12 = series.ewm(span=12, adjust=False).mean()
        exp26 = series.ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, period=20, std=2) -> Tuple[pd.Series, pd.Series]:
        sma = series.rolling(window=period).mean()
        std_dev = series.rolling(window=period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, lower

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df['ema_20'] = TechnicalIndicators.calculate_ema(df['close'], 20)
        df['ema_50'] = TechnicalIndicators.calculate_ema(df['close'], 50)
        df['ema_200'] = TechnicalIndicators.calculate_ema(df['close'], 200)
        df['rsi'] = TechnicalIndicators.calculate_rsi(df['close'], 14)
        df['atr'] = TechnicalIndicators.calculate_atr(df, 14)
        df['macd'], df['macd_signal'], df['macd_hist'] = TechnicalIndicators.calculate_macd(df['close'])
        df['bb_upper'], df['bb_lower'] = TechnicalIndicators.calculate_bollinger_bands(df['close'])
        
        # Volatility Squeeze Filter (BB Width)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        df.dropna(inplace=True)
        return df

# ============================================================================
# STRATEGY ANALYZER (Institutional Logic)
# ============================================================================
class StrategyAnalyzer:
    
    @staticmethod
    def analyze(df: pd.DataFrame) -> Tuple[Optional[str], float, List[str], str]:
        """
        Strategy: Trend Following + Momentum Pullback
        Timeframe: M15
        """
        if len(df) < 50: return None, 0.0, [], "Uncertain"
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        score = 0.0
        reasons = []
        action = None
        trend = "NEUTRAL"
        
        # 1. MACRO TREND IDENTIFICATION (EMA 200)
        is_bullish_macro = last['close'] > last['ema_200']
        is_bearish_macro = last['close'] < last['ema_200']
        
        # 2. IMMEDIATE TREND (EMA 50) & ALIGNMENT
        if is_bullish_macro and last['ema_50'] > last['ema_200']:
            trend = "STRONG UPTREND"
            if last['close'] > last['ema_50']:
                action = "BUY"
                score += 30
                reasons.append("✅ Price above EMA 200 & 50 (Strong Trend)")
        elif is_bearish_macro and last['ema_50'] < last['ema_200']:
            trend = "STRONG DOWNTREND"
            if last['close'] < last['ema_50']:
                action = "SELL"
                score += 30
                reasons.append("✅ Price below EMA 200 & 50 (Strong Trend)")
        else:
            return None, 0.0, ["⚠️ Market is ranging/choppy (EMAs crossed)"], "CHOPPY"

        # 3. MOMENTUM FILTER (MACD)
        # We want to enter when MACD confirms direction
        if action == "BUY":
            if last['macd'] > last['macd_signal']:
                score += 20
                reasons.append("✅ MACD is bullish")
            elif last['macd_hist'] > prev['macd_hist']:
                score += 10
                reasons.append("✅ MACD Histogram ticking up")
            else:
                score -= 10
                reasons.append("⚠️ MACD Momentum fading")
                
        elif action == "SELL":
            if last['macd'] < last['macd_signal']:
                score += 20
                reasons.append("✅ MACD is bearish")
            elif last['macd_hist'] < prev['macd_hist']:
                score += 10
                reasons.append("✅ MACD Histogram ticking down")
            else:
                score -= 10
                reasons.append("⚠️ MACD Momentum fading")

        # 4. ENTRY TIMING (RSI) - The "Pullback" Logic
        # We don't want to buy the top. We want to buy a dip in an uptrend.
        if action == "BUY":
            if 40 <= last['rsi'] <= 60:
                score += 25
                reasons.append("✅ RSI in accumulation zone (Perfect Entry)")
            elif last['rsi'] > 70:
                return None, 0.0, ["❌ RSI Overbought - Waiting for pullback"], trend
            elif last['rsi'] < 40:
                score += 15
                reasons.append("⚠️ RSI Oversold (Aggressive Entry)")
                
        elif action == "SELL":
            if 40 <= last['rsi'] <= 60:
                score += 25
                reasons.append("✅ RSI in distribution zone (Perfect Entry)")
            elif last['rsi'] < 30:
                return None, 0.0, ["❌ RSI Oversold - Waiting for pullback"], trend
            elif last['rsi'] > 60:
                score += 15
                reasons.append("⚠️ RSI Overbought (Aggressive Entry)")

        # 5. VOLATILITY CHECK
        # Compare current BB width to average BB width of last 20 periods
        avg_width = df['bb_width'].rolling(20).mean().iloc[-1]
        if last['bb_width'] < (avg_width * 0.7):
            return None, 0.0, ["❌ Volatility Squeeze (Dead Market)"], "SQUEEZE"
            
        return action, score, reasons, trend

# ============================================================================
# BOT CONTROLLER
# ============================================================================
class SyntheticBot:
    
    # Updated Symbol List (No Crash/Boom)
    TARGET_SYMBOLS = [
        'V10', 'V25', 'V50', 'V75', 'V100',
        'V10(1s)', 'V25(1s)', 'V50(1s)', 'V75(1s)', 'V100(1s)',
        'JUMP10', 'JUMP25', 'JUMP50', 'STEP'
    ]
    
    def __init__(self):
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('MAIN_CHAT_ID')
        self.notifier = TelegramNotifier(self.telegram_token, self.chat_id)
        self.fetcher = DerivDataFetcher()
        self.processed_signals = deque(maxlen=50) # Cooldown tracker

    def calculate_levels(self, df: pd.DataFrame, action: str, atr: float) -> dict:
        """Dynamic ATR Based Stops"""
        price = df.iloc[-1]['close']
        
        # 3.0 ATR Stop for Volatility Indices (they wick hard)
        sl_dist = atr * 3.0 
        
        if action == "BUY":
            sl = price - sl_dist
            tp1 = price + (sl_dist * 1.5) # 1:1.5
            tp2 = price + (sl_dist * 2.5) # 1:2.5
            tp3 = price + (sl_dist * 4.0) # 1:4.0
        else:
            sl = price + sl_dist
            tp1 = price - (sl_dist * 1.5)
            tp2 = price - (sl_dist * 2.5)
            tp3 = price - (sl_dist * 4.0)
            
        return {
            "entry": price, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "dist": sl_dist, "rr": 2.5 # Avg RR
        }

    def run(self):
        logger.info("🚀 STARTING SNIPER BOT v4.0 (M15 Analysis)")
        
        for symbol in self.TARGET_SYMBOLS:
            try:
                # 1. Fetch Candle Data
                df = self.fetcher.get_candle_data(symbol)
                if df is None: continue
                
                # 2. Add Indicators
                df = TechnicalIndicators.add_indicators(df)
                
                # 3. Analyze Strategy
                action, confidence, reasons, trend = StrategyAnalyzer.analyze(df)
                
                # 4. Filter Signals
                if action and confidence >= 75.0: # Institutional Threshold
                    
                    # Duplicate check
                    sig_key = f"{symbol}_{action}"
                    if sig_key in self.processed_signals:
                        continue
                        
                    # Calculate Levels
                    atr = df.iloc[-1]['atr']
                    levels = self.calculate_levels(df, action, atr)
                    
                    # Determine Quality
                    quality = SignalQuality.MODERATE
                    if confidence > 85: quality = SignalQuality.HIGH
                    if confidence > 90: quality = SignalQuality.PREMIUM
                    
                    signal = TradeSignal(
                        signal_id=f"{symbol}_{int(datetime.now().timestamp())}",
                        symbol=symbol,
                        action=action,
                        confidence=confidence,
                        quality=quality,
                        entry_price=levels['entry'],
                        stop_loss=levels['sl'],
                        take_profit_1=levels['tp1'],
                        take_profit_2=levels['tp2'],
                        take_profit_3=levels['tp3'],
                        stop_loss_distance=levels['dist'],
                        risk_reward_ratio=levels['rr'],
                        atr_value=atr,
                        trend_state=trend,
                        timestamp=datetime.now(),
                        strategy_components=reasons
                    )
                    
                    logger.info(f"✅ SIGNAL FOUND: {symbol} {action}")
                    if self.notifier.send_signal(signal):
                        self.processed_signals.append(sig_key)
                
            except Exception as e:
                logger.error(f"Error on {symbol}: {e}")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    if not os.getenv('TELEGRAM_BOT_TOKEN') or not os.getenv('MAIN_CHAT_ID'):
        print("❌ Error: Set TELEGRAM_BOT_TOKEN and MAIN_CHAT_ID env vars.")
        sys.exit(1)
        
    bot = SyntheticBot()
    bot.run()