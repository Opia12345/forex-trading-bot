"""
SYNTHETIC INDICES TRADING BOT v5.2 - COMBINED SNIPER EDITION
Specialized for Deriv Volatility, Step, and Jump Indices.
Strategy Focus: Dual Strategy - Trend Pullback (v5.0) & Counter-Trend Reversal (v5.1)

*** CRITICAL UPDATE v5.2: Runs both high-accuracy strategies simultaneously
    to maximize day trading opportunities while maintaining tight risk.
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
import requests
import json
import asyncio
import websockets

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('synthetic_bot_v5_2_combined.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Data Classes ---
class SignalQuality(Enum):
    PREMIUM_PULLBACK = "PREMIUM (PULLBACK)"
    PREMIUM_REVERSAL = "PREMIUM (REVERSAL)"
    HIGH_PULLBACK = "HIGH (PULLBACK)"
    HIGH_REVERSAL = "HIGH (REVERSAL)"

@dataclass
class TradeSignal:
    signal_id: str
    symbol: str
    action: str
    strategy_type: str # New field to distinguish strategy
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
            # Dynamic Emojis based on Strategy Type
            if signal.strategy_type == "REVERSAL":
                q_emoji = "🚨" if signal.quality.name.startswith("PREMIUM") else "⚠️"
                title = "REVERSAL SNIPER v5.2 (Top/Bottom)"
            else: # PULLBACK
                q_emoji = "🎯" if signal.quality.name.startswith("PREMIUM") else "🔥"
                title = "TREND SNIPER v5.2 (Pullback)"

            a_emoji = "🟢 BUY" if signal.action == "BUY" else "🔴 SELL"
            
            # Risk Display
            if signal.strategy_type == "REVERSAL":
                risk_note = "**EXTREME TIGHT** Stop Loss"
            else:
                risk_note = "**TIGHT** Stop Loss"
            
            message = f"""
{q_emoji} <b>{title} - DAY TRADE</b> {q_emoji}

{a_emoji} <b>{signal.symbol}</b>
⏳ <b>Timeframe:</b> M15
🛡️ <b>Confidence:</b> {signal.confidence:.1f}% ({signal.quality.value})

<b>📍 ENTRY & EXITS</b>
💵 Entry: <code>{signal.entry_price:.4f}</code>
🛑 {risk_note}: <code>{signal.stop_loss:.4f}</code>
🎯 TP1 (1:{(signal.risk_reward_ratio * 0.5):.1f}): <code>{signal.take_profit_1:.4f}</code>
🎯 TP2 (1:{signal.risk_reward_ratio:.1f}): <code>{signal.take_profit_2:.4f}</code>
🎯 TP3 (1:{(signal.risk_reward_ratio * 1.5):.1f}): <code>{signal.take_profit_3:.4f}</code>

<b>📊 TECHNICAL CONFLUENCE</b>
"""
            for component in signal.strategy_components:
                message += f"• {component}\n"

            message += f"""
<b>💼 RISK DATA (Small Account Focus)</b>
• Strategy Type: <b>{signal.strategy_type}</b>
• Risk:Reward (Avg): 1:{signal.risk_reward_ratio:.2f}
• Trend: {signal.trend_state}

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
# DERIV DATA FETCHER & TECHNICAL INDICATORS (Unchanged)
# ============================================================================
class DerivDataFetcher:
    SYMBOLS = {
        'V10': 'R_10', 'V25': 'R_25', 'V50': 'R_50', 'V75': 'R_75', 'V100': 'R_100',
        'V10(1s)': '1HZ10V', 'V25(1s)': '1HZ25V', 'V50(1s)': '1HZ50V', 
        'V75(1s)': '1HZ75V', 'V100(1s)': '1HZ100V',
        'JUMP10': 'JD10', 'JUMP25': 'JD25', 'JUMP50': 'JD50',
        'STEP': 'STEP'
    }
    
    def __init__(self, app_id: str = "1089"):
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    
    async def _fetch_candles_async(self, symbol: str, count: int = 300) -> Optional[List]:
        try:
            async with websockets.connect(self.ws_url, ping_interval=30, close_timeout=10) as ws:
                request = {
                    "ticks_history": symbol, "adjust_start_time": 1, "count": count,
                    "end": "latest", "start": 1, "style": "candles", "granularity": 900 
                }
                await ws.send(json.dumps(request))
                response = await asyncio.wait_for(ws.recv(), timeout=20)
                data = json.loads(response)
                return data.get('candles')
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
            df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'epoch': 'time'})
            cols = ['open', 'high', 'low', 'close']
            df[cols] = df[cols].astype(float)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except Exception as e:
            logger.error(f"Data conversion error: {e}")
            return None

class TechnicalIndicators:
    
    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = np.where(loss == 0, 1e-10, gain / loss)
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
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df.dropna(inplace=True)
        return df

    @staticmethod
    def is_reversal_candle(last: pd.Series, action: str) -> bool:
        """Simple Check for reversal candlestick (Hammer/Inverted Hammer logic)"""
        body = abs(last['open'] - last['close'])
        total_range = last['high'] - last['low']
        if total_range == 0 or total_range < (last['atr'] * 0.5): return False
        if body > (total_range * 0.3): return False

        if action == "BUY":
            lower_wick = last['close'] - last['low'] if last['close'] >= last['open'] else last['open'] - last['low']
            return lower_wick > (total_range * 0.5) 
        
        elif action == "SELL":
            upper_wick = last['high'] - last['close'] if last['close'] >= last['open'] else last['high'] - last['open']
            return upper_wick > (total_range * 0.5) 
        
        return False

# ============================================================================
# STRATEGY ANALYZER - DUAL MODE
# ============================================================================
class StrategyAnalyzer:
    
    @staticmethod
    def analyze_trend_pullback(df: pd.DataFrame) -> Tuple[Optional[str], float, List[str], str]:
        """Strategy v5.0: Strict Trend Following + Perfect Momentum Pullback"""
        if len(df) < 200: return None, 0.0, [], "Insufficient Data"
        last = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0.0
        reasons = []
        action = None
        trend = "NEUTRAL"
        
        # 1. MACRO TREND
        is_bullish_macro = last['ema_50'] > last['ema_200']
        is_bearish_macro = last['ema_50'] < last['ema_200']
        
        if is_bullish_macro: trend = "STRONG UPTREND"
        elif is_bearish_macro: trend = "STRONG DOWNTREND"
        else: return None, 0.0, ["❌ EMAs Crossed/Ranging - No Directional Bias"], "CHOPPY"
        
        # 2. IMMEDIATE TREND & ENTRY CANDIDATE
        if trend == "STRONG UPTREND":
            if last['close'] > last['ema_50'] and last['close'] > last['ema_20']:
                action = "BUY"; score += 35; reasons.append("✅ Price above EMAs (Full Alignment)")
            else: return None, 0.0, ["⚠️ Price below short-term EMA 20/50 - Not ready to buy yet"], trend
                
        elif trend == "STRONG DOWNTREND":
            if last['close'] < last['ema_50'] and last['close'] < last['ema_20']:
                action = "SELL"; score += 35; reasons.append("✅ Price below EMAs (Full Alignment)")
            else: return None, 0.0, ["⚠️ Price above short-term EMA 20/50 - Not ready to sell yet"], trend
        
        if not action: return None, 0.0, reasons, trend

        # 3. MOMENTUM FILTER (MACD)
        momentum_score = 0
        if action == "BUY" and last['macd'] > last['macd_signal'] and last['macd_hist'] > prev['macd_hist']:
            momentum_score = 30; reasons.append("✅ MACD is bullish and Histogram expanding")
        elif action == "SELL" and last['macd'] < last['macd_signal'] and last['macd_hist'] < prev['macd_hist']:
            momentum_score = 30; reasons.append("✅ MACD is bearish and Histogram expanding")
                
        if momentum_score < 30: # Maximize confidence for a 90% target
            return None, 0.0, ["❌ MACD Momentum is not expanding strongly enough"], trend

        score += momentum_score

        # 4. ENTRY TIMING (RSI) - The "PERFECT PULLBACK" Logic
        if action == "BUY":
            if 45 <= last['rsi'] <= 65:
                score += 35; reasons.append("🎯 RSI in Sweet Spot (45-65) - Perfect Pullback Entry")
            else:
                return None, 0.0, ["❌ RSI outside (45-65) zone - Waiting for a better pullback"], trend
                
        elif action == "SELL":
            if 35 <= last['rsi'] <= 55:
                score += 35; reasons.append("🎯 RSI in Sweet Spot (35-55) - Perfect Pullback Entry")
            else:
                return None, 0.0, ["❌ RSI outside (35-55) zone - Waiting for a better rally"], trend

        return action, score, reasons, trend
    
    @staticmethod
    def analyze_counter_reversal(df: pd.DataFrame) -> Tuple[Optional[str], float, List[str], str]:
        """Strategy v5.1: Counter-Trend Reversal (Buy Bottom / Sell Top)"""
        if len(df) < 200: return None, 0.0, [], "Insufficient Data"
        last = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0.0
        reasons = []
        action = None
        trend = "REVERSAL HUNT"
        
        # 1. RSI EXTREME FILTER (The exhaustion signal)
        is_oversold = last['rsi'] < 25
        is_overbought = last['rsi'] > 75
        
        if is_oversold:
            action = "BUY"; score += 40; reasons.append("✅ RSI Extreme: Price is Oversold (Potential Bottom)")
        elif is_overbought:
            action = "SELL"; score += 40; reasons.append("✅ RSI Extreme: Price is Overbought (Potential Top)")
        else:
            return None, 0.0, ["❌ RSI not in extreme zone (Not a reversal candidate)"], trend

        # 2. MACD REVERSAL CONFIRMATION (The turning point)
        if action == "BUY":
            if last['macd_hist'] > prev['macd_hist'] and last['macd'] < 0:
                score += 30; reasons.append("✅ MACD Histogram Ticking UP (Momentum Shift Confirmed)")
            else:
                return None, 0.0, ["❌ MACD still bearish/flat (No reversal confirmation)"], trend
                
        elif action == "SELL":
            if last['macd_hist'] < prev['macd_hist'] and last['macd'] > 0:
                score += 30; reasons.append("✅ MACD Histogram Ticking DOWN (Momentum Shift Confirmed)")
            else:
                return None, 0.0, ["❌ MACD still bullish/flat (No reversal confirmation)"], trend
        
        # 3. CANDLESTICK CONFIRMATION
        if TechnicalIndicators.is_reversal_candle(last, action):
            score += 30; reasons.append("🎯 Reversal Candle Confirmed (Pin Bar/Hammer) - Ultimate Sniping")
        else:
            score -= 5; reasons.append("⚠️ No Classic Reversal Candle - Requires more conviction")

        return action, score, reasons, trend


# ============================================================================
# BOT CONTROLLER
# ============================================================================
class SyntheticBot:
    
    TARGET_SYMBOLS = [
        'V10', 'V25', 'V50', 'V75', 'V100',
        'V10(1s)', 'V25(1s)', 'V50(1s)', 'V75(1s)', 'V100(1s)',
        'JUMP10', 'JUMP25', 'JUMP50', 'STEP'
    ]
    
    def __init__(self):
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('MAIN_CHAT_ID')
        
        if not self.telegram_token or not self.chat_id:
            logger.error("TELEGRAM_BOT_TOKEN or MAIN_CHAT_ID not set.")
            sys.exit(1)
            
        self.notifier = TelegramNotifier(self.telegram_token, self.chat_id)
        self.fetcher = DerivDataFetcher()
        self.processed_signals = deque(maxlen=50) 

    def calculate_levels(self, df: pd.DataFrame, action: str, atr: float, strategy_type: str) -> dict:
        """Dynamic ATR Based Stops based on Strategy Type"""
        price = df.iloc[-1]['close']
        
        # ATR Multiplier based on strategy
        if strategy_type == "REVERSAL":
            # Tighter SL for high-precision reversal (1.5x ATR)
            sl_mult = 1.5
            rr_mults = (1.0, 2.0, 3.0) # TP1: 1:1, TP2: 1:2, TP3: 1:3
            avg_rr = 2.0
        else: # PULLBACK
            # Slightly wider SL for trend pullbacks (2.0x ATR)
            sl_mult = 2.0
            rr_mults = (1.5, 2.5, 3.5) # TP1: 1:1.5, TP2: 1:2.5, TP3: 1:3.5
            avg_rr = 2.5
            
        sl_dist = atr * sl_mult
        
        if action == "BUY":
            sl = price - sl_dist
            tp1 = price + (sl_dist * rr_mults[0]) 
            tp2 = price + (sl_dist * rr_mults[1]) 
            tp3 = price + (sl_dist * rr_mults[2]) 
        else:
            sl = price + sl_dist
            tp1 = price - (sl_dist * rr_mults[0])
            tp2 = price - (sl_dist * rr_mults[1])
            tp3 = price - (sl_dist * rr_mults[2])
            
        return {
            "entry": price, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "dist": sl_dist, "rr": avg_rr 
        }

    def process_signal(self, symbol, df, action, confidence, reasons, trend, strategy_type):
        """Helper to create and send a single signal"""
        
        # Check if high confidence threshold is met for sending
        if strategy_type == "PULLBACK" and confidence < 90.0: return
        if strategy_type == "REVERSAL" and confidence < 85.0: return # Reversals have a slightly lower threshold due to nature
        
        # Check for duplicates (important if both strategies fire on the same candle)
        sig_key = f"{symbol}_{action}_{strategy_type}"
        if sig_key in self.processed_signals:
            return
            
        atr = df.iloc[-1]['atr']
        levels = self.calculate_levels(df, action, atr, strategy_type)
        
        # Determine Quality
        if strategy_type == "PULLBACK":
            quality = SignalQuality.PREMIUM_PULLBACK if confidence >= 95.0 else SignalQuality.HIGH_PULLBACK
        else: # REVERSAL
            quality = SignalQuality.PREMIUM_REVERSAL if confidence >= 90.0 else SignalQuality.HIGH_REVERSAL
            
        signal = TradeSignal(
            signal_id=f"V5.2-{strategy_type[0]}-{symbol}_{int(datetime.now().timestamp())}",
            symbol=symbol,
            action=action,
            strategy_type=strategy_type,
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
        
        logger.info(f"SIGNAL FOUND: {symbol} {action} ({strategy_type} - {confidence:.1f}%)")
        if self.notifier.send_signal(signal):
            self.processed_signals.append(sig_key)


    def run(self):
        logger.info("🚀 STARTING DUAL SNIPER BOT v5.2 (Combined Trend & Reversal)")
        
        for symbol in self.TARGET_SYMBOLS:
            try:
                # 1. Fetch Candle Data & Indicators (Only need to do this once)
                df = self.fetcher.get_candle_data(symbol)
                if df is None or df.empty: continue
                df = TechnicalIndicators.add_indicators(df)
                if df.empty: continue
                
                # 2. Analyze Strategy 1: Trend Pullback (High Win-Rate, Safer)
                action_pb, conf_pb, reasons_pb, trend_pb = StrategyAnalyzer.analyze_trend_pullback(df)
                if action_pb:
                    self.process_signal(symbol, df, action_pb, conf_pb, reasons_pb, trend_pb, "PULLBACK")

                # 3. Analyze Strategy 2: Counter Reversal (High Risk/Reward, Top/Bottom)
                action_rev, conf_rev, reasons_rev, trend_rev = StrategyAnalyzer.analyze_counter_reversal(df)
                if action_rev:
                    self.process_signal(symbol, df, action_rev, conf_rev, reasons_rev, trend_rev, "REVERSAL")
                
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