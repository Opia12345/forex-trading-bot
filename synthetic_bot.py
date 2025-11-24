"""
SYNTHETIC INDICES TRADING BOT v3.0 - INSTITUTIONAL GRADE
Specialized for Deriv Volatility Indices (V10, V25, V50, V75, V100)
Refactored for enhanced signal accuracy and robust risk management.
Disclaimer: No trading strategy can guarantee 100% accuracy.
"""

import os
import sys
import logging
from typing import Optional, List, Tuple, Dict
from datetime import datetime, timedelta
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

# --- Logging Setup (No Change) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('synthetic_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Data Classes (Refined for Dataclass Rules) ---
class SignalQuality(Enum):
    EXCELLENT = "EXCELLENT"
    STRONG = "STRONG"
    GOOD = "GOOD"

@dataclass
class TradeSignal:
    # All non-default arguments must come first
    signal_id: str
    symbol: str
    action: str  # BUY or SELL
    confidence: float
    quality: SignalQuality
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    stop_loss_distance: float
    tp1_distance: float
    tp2_distance: float
    tp3_distance: float
    stake_usd: float
    risk_reward_ratio: float
    atr_value: float
    volatility_level: str # <-- Now correctly sequenced before the default
    trend_strength: str
    timestamp: datetime
    # Default argument moved to the end
    strategy_components: List[str] = field(default_factory=list) # <-- CORRECTION 2: Moved to the end and set with a proper default factory

# ============================================================================
# TELEGRAM NOTIFIER (No Change)
# ============================================================================
# (Class TelegramNotifier remains the same as its function is notification)

class TelegramNotifier:
    """Sends trading signals to main Telegram group"""
    
    def __init__(self, token: str, main_chat_id: str):
        self.token = token
        self.main_chat_id = main_chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_signal(self, signal: TradeSignal) -> bool:
        """Send signal to Telegram"""
        try:
            quality_emoji = {
                "EXCELLENT": "🌟",
                "STRONG": "⭐",
                "GOOD": "✨"
            }.get(signal.quality.value, "✨")
            
            action_emoji = "🟢" if signal.action == "BUY" else "🔴"
            
            message = f"""
{quality_emoji} <b>INSTITUTIONAL SIGNAL - {signal.quality.value}</b> {quality_emoji}

{action_emoji} <b>{signal.action} {signal.symbol}</b>
💯 Confidence: {signal.confidence:.1f}%
📊 Quality: {signal.quality.value}

<b>📍 ENTRY & EXIT LEVELS</b>
💵 Entry Price: <code>{signal.entry_price:.2f}</code>
🛑 Stop Loss: <code>{signal.stop_loss:.2f}</code> ({signal.stop_loss_distance:.2f} points)
🎯 TP1: <code>{signal.take_profit_1:.2f}</code> (Min R:R 1:2.0)
🎯 TP2: <code>{signal.take_profit_2:.2f}</code> (R:R 1:3.0+)
🎯 TP3: <code>{signal.take_profit_3:.2f}</code> (R:R 1:4.0+)

<b>💼 POSITION DETAILS</b>
💰 Stake: ${signal.stake_usd:.2f} (Risk Per Trade)
📊 Risk:Reward (Overall): 1:{signal.risk_reward_ratio:.2f}
📈 ATR (14-period): {signal.atr_value:.2f}
🔥 Volatility: {signal.volatility_level}
💪 Trend: {signal.trend_strength}

<b>✅ SIGNAL RATIONALE (Top 8 Factors)</b>
"""
            
            for reason in signal.strategy_components[:8]:
                message += f"• {reason}\n"
            
            message += f"""
<b>⚠️ INSTITUTIONAL TRADE MANAGEMENT</b>
• **TP1 (Exit 50%):** Secure profit and move SL to **Breakeven** (+1 point).
• **TP2 (Exit 30%):** Secure more profit and set a **Trailing Stop** below price action.
• **TP3 (Exit 20%):** Let the final portion run, respecting the trailing stop.

<b>🎯 RISK MANAGEMENT MANDATE</b>
• Max risk per trade: {signal.stake_usd:.2f} USD
• **NEVER** exceed the initial Stop Loss.
• **NEVER** add to a losing position.

<i>Signal ID: {signal.signal_id}</i>
<i>Generated: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</i>
"""
            
            return self._send_message(message)
        
        except Exception as e:
            logger.error(f"Error sending signal: {e}")
            return False
    
    def _send_message(self, message: str) -> bool:
        """Send message to main Telegram group"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.main_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("✅ Signal sent to Telegram")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

# ============================================================================
# DERIV DATA FETCHER (No Change)
# ============================================================================
# (Class DerivDataFetcher remains the same)
class DerivDataFetcher:
    """Fetch synthetic indices data from Deriv"""
    
    SYMBOLS = {
        'V10': 'R_10',
        'V25': 'R_25',
        'V50': 'R_50',
        'V75': 'R_75',
        'V100': 'R_100'
    }
    
    def __init__(self, app_id: str = "1089"):
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    
    async def _fetch_ticks_async(self, symbol: str, count: int = 2000) -> Optional[List]:
        """Fetch tick data via WebSocket"""
        try:
            async with websockets.connect(self.ws_url, ping_interval=30, close_timeout=10) as ws:
                request = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": count,
                    "end": "latest",
                    "start": 1,
                    "style": "ticks"
                }
                
                await ws.send(json.dumps(request))
                response = await asyncio.wait_for(ws.recv(), timeout=30)
                data = json.loads(response)
                
                if 'error' in data:
                    logger.error(f"Deriv API error: {data['error']}")
                    return None
                
                if 'history' in data and 'prices' in data['history']:
                    return data['history']['prices']
                
                return None
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching data for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def get_tick_data(self, symbol: str, count: int = 2000) -> Optional[pd.DataFrame]:
        """Get tick data as DataFrame with sufficient history"""
        deriv_symbol = self.SYMBOLS.get(symbol)
        if not deriv_symbol:
            logger.error(f"Invalid symbol: {symbol}")
            return None
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                prices = loop.run_until_complete(
                    self._fetch_ticks_async(deriv_symbol, count)
                )
            finally:
                loop.close()
            
            if not prices or len(prices) < 200:
                logger.error(f"Insufficient tick data: {len(prices) if prices else 0}")
                return None
            
            df = pd.DataFrame({'price': prices})
            df['tick'] = range(len(df))
            
            logger.info(f"✅ Fetched {len(df)} ticks for {symbol}")
            return df
        
        except Exception as e:
            logger.error(f"Error getting tick data: {e}")
            return None

# ============================================================================
# TECHNICAL INDICATORS (Small Refactoring for Robustness)
# ============================================================================

class TechnicalIndicators:
    """Industry-standard technical indicators"""
    
    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        # Avoid division by zero and handle initial NaNs gracefully
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range - proper calculation"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean() # Using EMA for ATR (smoother)
        
        return atr
    
    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, period: int = 20, std: float = 2.0) -> Tuple:
        """Bollinger Bands"""
        sma = series.rolling(window=period).mean()
        rolling_std = series.rolling(window=period).std()
        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        return upper, sma, lower
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index - trend strength"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = TechnicalIndicators.calculate_atr(df, 1) # True Range for initial DI smoothing
        atr = tr.ewm(span=period, adjust=False).mean() 
        
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr).replace([np.inf, -np.inf], np.nan)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr).replace([np.inf, -np.inf], np.nan)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx.fillna(0)
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to DataFrame"""
        # Create OHLC from tick data (using 20 ticks for better local range)
        window = 20
        df['high'] = df['price'].rolling(window=window).max()
        df['low'] = df['price'].rolling(window=window).min()
        df['close'] = df['price'] # 'close' is just the current price in this tick context
        
        # Moving averages
        df['ema_9'] = TechnicalIndicators.calculate_ema(df['price'], 9)
        df['ema_21'] = TechnicalIndicators.calculate_ema(df['price'], 21)
        df['ema_50'] = TechnicalIndicators.calculate_ema(df['price'], 50)
        df['ema_100'] = TechnicalIndicators.calculate_ema(df['price'], 100)
        df['ema_200'] = TechnicalIndicators.calculate_ema(df['price'], 200)
        
        # Momentum indicators
        df['rsi'] = TechnicalIndicators.calculate_rsi(df['price'], 14)
        df['rsi_9'] = TechnicalIndicators.calculate_rsi(df['price'], 9)
        
        # Volatility indicators
        df['atr'] = TechnicalIndicators.calculate_atr(df, 14)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = \
            TechnicalIndicators.calculate_bollinger_bands(df['price'], 20, 2.0)
        
        # Trend strength
        df['adx'] = TechnicalIndicators.calculate_adx(df, 14)
        
        # Fill NaN values (robust fill)
        df = df.fillna(method='bfill', limit=100).fillna(method='ffill', limit=100)
        
        return df


# ============================================================================
# ADVANCED SIGNAL ANALYZER (Refined for Accuracy)
# ============================================================================

class AdvancedSignalAnalyzer:
    """Advanced signal analysis with proper market structure and filtering"""
    
    @staticmethod
    def analyze_trend(df: pd.DataFrame) -> Tuple[Optional[str], float, List[str], str]:
        """Analyze trend with multiple confirmations and price action structure"""
        last = df.iloc[-1]
        
        score = 0.0
        reasons = []
        action = None
        trend_strength = "WEAK"
        
        # 1. EMA Alignment (40 points max)
        if (last['ema_9'] > last['ema_21'] > last['ema_50'] > last['ema_100']):
            score += 40.0
            action = "BUY"
            reasons.append("✅ Perfect bullish EMA stack (9>21>50>100)")
        elif (last['ema_9'] < last['ema_21'] < last['ema_50'] < last['ema_100']):
            score += 40.0
            action = "SELL"
            reasons.append("✅ Perfect bearish EMA stack (9<21<50<100)")
        elif last['ema_9'] > last['ema_21'] and last['ema_21'] > last['ema_50']:
            score += 30.0
            action = "BUY"
            reasons.append("Strong bullish trend (EMA 9>21>50)")
        elif last['ema_9'] < last['ema_21'] and last['ema_21'] < last['ema_50']:
            score += 30.0
            action = "SELL"
            reasons.append("Strong bearish trend (EMA 9<21<50)")
        elif last['ema_9'] > last['ema_21']:
            score += 15.0
            action = "BUY"
            reasons.append("Moderate bullish trend (EMA 9>21 cross)")
        elif last['ema_9'] < last['ema_21']:
            score += 15.0
            action = "SELL"
            reasons.append("Moderate bearish trend (EMA 9<21 cross)")
        else:
            return None, 0.0, ["❌ No clear trend or conflicting EMAs"], "WEAK"
        
        # 2. Price Action Structure (10 points max)
        recent_highs = df['high'].tail(10)
        recent_lows = df['low'].tail(10)
        
        # Check for Higher Highs and Higher Lows in the last 10 ticks
        if action == "BUY":
            if (recent_highs.iloc[-1] > recent_highs.iloc[-5] and 
                recent_lows.iloc[-1] > recent_lows.iloc[-5]):
                score += 10.0
                reasons.append("✅ Confirmed by Higher Highs/Higher Lows")
            elif recent_highs.iloc[-1] > recent_highs.iloc[-5]:
                score += 5.0
                reasons.append("Higher Highs confirming bullish structure")
            elif recent_lows.iloc[-1] > recent_lows.iloc[-5]:
                score += 5.0
                reasons.append("Higher Lows confirming bullish structure")
            else:
                reasons.append("⚠️ Price structure is currently flat or counter-trending")
                
        # Check for Lower Highs and Lower Lows in the last 10 ticks
        else: # SELL
            if (recent_highs.iloc[-1] < recent_highs.iloc[-5] and 
                recent_lows.iloc[-1] < recent_lows.iloc[-5]):
                score += 10.0
                reasons.append("✅ Confirmed by Lower Highs/Lower Lows")
            elif recent_highs.iloc[-1] < recent_highs.iloc[-5]:
                score += 5.0
                reasons.append("Lower Highs confirming bearish structure")
            elif recent_lows.iloc[-1] < recent_lows.iloc[-5]:
                score += 5.0
                reasons.append("Lower Lows confirming bearish structure")
            else:
                reasons.append("⚠️ Price structure is currently flat or counter-trending")
        
        # Determine overall trend strength
        if score >= 40.0:
            trend_strength = "VERY STRONG"
        elif score >= 25.0:
            trend_strength = "STRONG"
        elif score >= 15.0:
            trend_strength = "MODERATE"
        
        return action, score, reasons, trend_strength
    
    @staticmethod
    def analyze_momentum(df: pd.DataFrame, action: str) -> Tuple[float, List[str]]:
        """Analyze momentum with RSI filtering out extreme conditions"""
        last = df.iloc[-1]
        score = 0.0
        reasons = []
        
        if action == "BUY":
            # Optimal Buy Zone (45-65) - not oversold, but gaining momentum
            if 45 <= last['rsi'] <= 65:
                score = 25.0
                reasons.append(f"✅ RSI in optimal buy momentum zone ({last['rsi']:.1f})")
            # Pullback Buy (35-45) - entering from oversold territory
            elif 35 <= last['rsi'] < 45:
                score = 20.0
                reasons.append(f"RSI pullback from oversold/support ({last['rsi']:.1f})")
            # Overbought Filter: Reject if RSI is too high
            elif last['rsi'] >= 70:
                return 0.0, [f"❌ RSI highly overbought - risky entry ({last['rsi']:.1f})"]
            else:
                score = 10.0
                reasons.append(f"RSI neutral or slightly weak for buy ({last['rsi']:.1f})")
        
        else:  # SELL
            # Optimal Sell Zone (35-55) - not overbought, but losing momentum
            if 35 <= last['rsi'] <= 55:
                score = 25.0
                reasons.append(f"✅ RSI in optimal sell momentum zone ({last['rsi']:.1f})")
            # Pullback Sell (55-65) - entering from overbought territory
            elif 55 < last['rsi'] <= 65:
                score = 20.0
                reasons.append(f"RSI rejection from overbought/resistance ({last['rsi']:.1f})")
            # Oversold Filter: Reject if RSI is too low
            elif last['rsi'] <= 30:
                return 0.0, [f"❌ RSI highly oversold - risky entry ({last['rsi']:.1f})"]
            else:
                score = 10.0
                reasons.append(f"RSI neutral or slightly weak for sell ({last['rsi']:.1f})")
        
        return score, reasons
    
    @staticmethod
    def analyze_price_action(df: pd.DataFrame, action: str) -> Tuple[float, List[str]]:
        """Analyze Bollinger Bands and price action for volatility and exhaustion"""
        last = df.iloc[-1]
        score = 0.0
        reasons = []
        
        bb_width = last['bb_upper'] - last['bb_lower']
        # Avoid division by zero
        bb_position = (last['price'] - last['bb_lower']) / bb_width if bb_width > 0 else 0.5
        
        if action == "BUY":
            # Best entry: Price touching or below the lower band (strong mean reversion potential)
            if bb_position <= 0.1:
                score = 20.0
                reasons.append("✅ Price is at lower BB (strong mean reversion entry)")
            elif bb_position < 0.35:
                score = 15.0
                reasons.append("Price in lower BB zone (bounce potential)")
            # Rejection filter: Reject entry near the upper band
            elif bb_position > 0.75:
                return 0.0, ["❌ Price is near upper BB (too extended for buy)"]
            else:
                score = 8.0
                reasons.append("Price in neutral BB zone")
        
        else:  # SELL
            # Best entry: Price touching or above the upper band (strong mean reversion potential)
            if bb_position >= 0.9:
                score = 20.0
                reasons.append("✅ Price is at upper BB (strong mean reversion entry)")
            elif bb_position > 0.65:
                score = 15.0
                reasons.append("Price in upper BB zone (rejection potential)")
            # Rejection filter: Reject entry near the lower band
            elif bb_position < 0.25:
                return 0.0, ["❌ Price is near lower BB (too extended for sell)"]
            else:
                score = 8.0
                reasons.append("Price in neutral BB zone")
        
        return score, reasons
    
    @staticmethod
    def analyze_trend_strength(df: pd.DataFrame) -> Tuple[float, List[str], str]:
        """Analyze ADX trend strength"""
        last = df.iloc[-1]
        score = 0.0
        reasons = []
        
        if last['adx'] > 40:
            score = 15.0
            strength = "VERY STRONG"
            reasons.append(f"✅ Very strong trend (ADX {last['adx']:.1f})")
        elif last['adx'] > 25:
            score = 15.0
            strength = "STRONG"
            reasons.append(f"✅ Strong trend (ADX {last['adx']:.1f})")
        elif last['adx'] > 20:
            score = 10.0
            strength = "MODERATE"
            reasons.append(f"Moderate trend (ADX {last['adx']:.1f})")
        # Weak Trend Filter: Reject trades in consolidation/no-trend zones
        elif last['adx'] <= 20:
            return 0.0, [f"❌ Insufficient trend strength (ADX {last['adx']:.1f}) - possible chop"], "NONE"
        
        return score, reasons, strength
    
    @staticmethod
    def calculate_confidence(df: pd.DataFrame) -> Tuple[Optional[str], float, List[str], str, str]:
        """Calculate overall confidence score with strict filtering"""
        all_reasons = []
        total_score = 0.0
        
        # 1. Trend Analysis (50 points max - 40 EMA + 10 PA Structure)
        action, trend_score, trend_reasons, trend_strength = AdvancedSignalAnalyzer.analyze_trend(df)
        if not action or trend_score < 25.0: # Minimum 25 for a tradable trend
            return None, 0.0, ["❌ Insufficient trend strength/clarity for entry"] + trend_reasons, "WEAK", "LOW"
        
        total_score += trend_score
        all_reasons.extend(trend_reasons)
        
        # 2. Trend Strength Analysis (15 points max)
        strength_score, strength_reasons, adx_strength = AdvancedSignalAnalyzer.analyze_trend_strength(df)
        if strength_score == 0.0:
            return None, 0.0, strength_reasons, trend_strength, "LOW"
        
        total_score += strength_score
        all_reasons.extend(strength_reasons)
        
        # 3. Momentum Analysis (25 points max)
        momentum_score, momentum_reasons = AdvancedSignalAnalyzer.analyze_momentum(df, action)
        if momentum_score == 0.0:
            return None, 0.0, momentum_reasons, trend_strength, "LOW"
        
        total_score += momentum_score
        all_reasons.extend(momentum_reasons)
        
        # 4. Price Action Analysis (20 points max)
        pa_score, pa_reasons = AdvancedSignalAnalyzer.analyze_price_action(df, action)
        if pa_score == 0.0:
            return None, 0.0, pa_reasons, trend_strength, "LOW"
        
        total_score += pa_score
        all_reasons.extend(pa_reasons)
        
        # Determine volatility level based on ATR relative to recent average
        last = df.iloc[-1]
        recent = df.tail(100)
        # Use a rolling median for a more stable average
        avg_atr = recent['atr'].median() 
        
        if last['atr'] > avg_atr * 1.5:
            vol_level = "VERY HIGH (Extreme Caution)"
        elif last['atr'] > avg_atr * 1.2:
            vol_level = "HIGH"
        elif last['atr'] > avg_atr * 0.8:
            vol_level = "NORMAL"
        else:
            vol_level = "LOW (Choppy/Tight)"
        
        return action, total_score, all_reasons, trend_strength, vol_level


# ============================================================================
# RISK MANAGER (Adjusted for Wider, Safer Stop Losses)
# ============================================================================

class RiskManager:
    """Professional risk management with wider, ATR-based stops for volatility indices"""
    
    @staticmethod
    def calculate_position_levels(df: pd.DataFrame, action: str, symbol: str, 
                                   stake_usd: float) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        last = df.iloc[-1]
        current_price = last['price']
        atr = last['atr']
        
        # Ensure ATR is valid
        if pd.isna(atr) or atr <= 0:
            atr = current_price * 0.01  # Fallback to 1% of price
            logger.warning(f"⚠️ ATR fallback used for {symbol}")
        
        # SYMBOL-SPECIFIC ATR MULTIPLIERS (Wider SL, Higher R:R potential)
        # SL multiplier is increased to give trades more room to breathe.
        # TP multipliers are set to ensure a minimum 1:2.0 R:R on TP1, and 1:4.0 on TP3.
        atr_multipliers = {
            # Symbol: {'sl': SL_MULTIPLIER, 'tp1': TP1_RR_RATIO, 'tp2': TP2_RR_RATIO, 'tp3': TP3_RR_RATIO}
            'V10': {'sl': 3.5, 'tp1': 2.0, 'tp2': 3.0, 'tp3': 4.5}, 
            'V25': {'sl': 3.0, 'tp1': 2.0, 'tp2': 3.0, 'tp3': 4.0}, 
            'V50': {'sl': 2.5, 'tp1': 2.0, 'tp2': 3.0, 'tp3': 4.0}, 
            'V75': {'sl': 2.2, 'tp1': 2.0, 'tp2': 3.0, 'tp3': 4.0}, 
            'V100': {'sl': 2.0, 'tp1': 2.0, 'tp2': 3.0, 'tp3': 4.0} 
        }
        
        multipliers = atr_multipliers.get(symbol, {'sl': 2.5, 'tp1': 2.0, 'tp2': 3.0, 'tp3': 4.0})
        
        sl_distance_points = atr * multipliers['sl']
        
        # Calculate levels based on the fixed SL distance
        if action == "BUY":
            entry = current_price
            stop_loss = entry - sl_distance_points
            
            # TP is calculated as a multiple of the risk (SL distance)
            tp1 = entry + (sl_distance_points * multipliers['tp1'])
            tp2 = entry + (sl_distance_points * multipliers['tp2'])
            tp3 = entry + (sl_distance_points * multipliers['tp3'])
            
        else:  # SELL
            entry = current_price
            stop_loss = entry + sl_distance_points
            
            # TP is calculated as a multiple of the risk (SL distance)
            tp1 = entry - (sl_distance_points * multipliers['tp1'])
            tp2 = entry - (sl_distance_points * multipliers['tp2'])
            tp3 = entry - (sl_distance_points * multipliers['tp3'])
        
        # Recalculate distances (should be clean now)
        sl_distance = abs(entry - stop_loss)
        tp1_distance = abs(tp1 - entry)
        tp2_distance = abs(tp2 - entry)
        tp3_distance = abs(tp3 - entry)
        
        # Final Risk-Reward ratio (based on TP3)
        rr_ratio = tp3_distance / sl_distance if sl_distance > 0 else 0
        
        # Ensure levels are appropriately rounded
        return {
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'tp1': round(tp1, 2),
            'tp2': round(tp2, 2),
            'tp3': round(tp3, 2),
            'sl_distance': round(sl_distance, 2),
            'tp1_distance': round(tp1_distance, 2),
            'tp2_distance': round(tp2_distance, 2),
            'tp3_distance': round(tp3_distance, 2),
            'rr_ratio': round(rr_ratio, 2),
            'atr': round(atr, 2)
        }


# ============================================================================
# SYNTHETIC TRADING BOT v3.0 (Updated Minimum Confidence)
# ============================================================================

class SyntheticTradingBot:
    """Professional-grade trading bot for synthetic indices"""
    
    SYMBOLS = ['V10', 'V25', 'V50', 'V75', 'V100']
    
    # INSTITUTIONAL MINIMUM CONFIDENCE (Raised from 80.0)
    MIN_CONFIDENCE = 85.0 
    
    def __init__(self, telegram_token: str, main_chat_id: str, stake_usd: float = 10.0):
        self.notifier = TelegramNotifier(telegram_token, main_chat_id)
        self.data_fetcher = DerivDataFetcher()
        self.stake_usd = stake_usd
        # Increased cooldown window for greater separation between signals
        self.recent_signals = deque(maxlen=30) 
        
        logger.info("=" * 80)
        logger.info("SYNTHETIC INDICES TRADING BOT v3.0 - INSTITUTIONAL GRADE")
        logger.info("=" * 80)
        logger.info("Features:")
        logger.info("  ✅ WIDER, Dynamic ATR-based Stop Loss calculation (Fix)")
        logger.info("  ✅ Reinforced Trend/PA structure filtering (Fix)")
        logger.info("  ✅ Higher MIN_CONFIDENCE threshold (Fix)")
        logger.info("  ✅ Strict 1:2.0 Min R:R on TP1")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  Symbols: {', '.join(self.SYMBOLS)}")
        logger.info(f"  Minimum Confidence: {self.MIN_CONFIDENCE}%")
        logger.info(f"  Stake per Trade: ${stake_usd:.2f}")
        logger.info("=" * 80)
    
    def analyze_symbol(self, symbol: str) -> Optional[TradeSignal]:
        """Analyze symbol for high-probability trading signals"""
        logger.info(f"\n{'='*70}")
        logger.info(f"ANALYZING {symbol}")
        logger.info(f"{'='*70}")
        
        # Fetch sufficient tick data for proper analysis
        logger.info("📥 Fetching tick data...")
        df = self.data_fetcher.get_tick_data(symbol, count=2000)
        
        if df is None:
            logger.error("❌ Failed to fetch data")
            return None
        
        # Calculate all technical indicators
        logger.info("🔧 Calculating technical indicators...")
        df = TechnicalIndicators.add_all_indicators(df)
        
        # Advanced signal analysis
        logger.info("🎯 Analyzing market structure...")
        action, confidence, reasons, trend_strength, vol_level = \
            AdvancedSignalAnalyzer.calculate_confidence(df)
        
        if not action:
            logger.info("❌ No qualified signal")
            return None
        
        logger.info(f"📊 Preliminary Confidence: {confidence:.1f}%")
        logger.info(f"📈 Trend Strength: {trend_strength}")
        logger.info(f"🔥 Volatility: {vol_level}")
        
        # Check minimum confidence threshold
        if confidence < self.MIN_CONFIDENCE:
            logger.info(f"❌ Below {self.MIN_CONFIDENCE}% confidence threshold - Rejecting")
            return None
        
        logger.info(f"✅ SIGNAL QUALIFIED ({confidence:.1f}%)")
        
        # Calculate position levels with proper risk management
        logger.info("💼 Calculating position levels...")
        levels = RiskManager.calculate_position_levels(df, action, symbol, self.stake_usd)
        
        # R:R Check: Ensure TP1 meets the minimum R:R of 1:2.0
        min_rr_tp1 = 2.0
        if (levels['tp1_distance'] / levels['sl_distance']) < min_rr_tp1:
            logger.warning(f"⚠️ TP1 R:R ({levels['tp1_distance'] / levels['sl_distance']:.2f}) is below minimum 1:{min_rr_tp1:.1f} - Rejecting")
            return None
        
        logger.info(f"💵 Entry: {levels['entry']:.2f}")
        logger.info(f"🛑 Stop Loss: {levels['stop_loss']:.2f} (Distance: {levels['sl_distance']:.2f})")
        logger.info(f"🎯 TP1: {levels['tp1']:.2f} (Distance: {levels['tp1_distance']:.2f})")
        logger.info(f"🎯 TP2: {levels['tp2']:.2f} (Distance: {levels['tp2_distance']:.2f})")
        logger.info(f"🎯 TP3: {levels['tp3']:.2f} (Distance: {levels['tp3_distance']:.2f})")
        logger.info(f"📊 Risk:Reward Ratio (Overall): 1:{levels['rr_ratio']:.2f}")
        logger.info(f"📈 ATR: {levels['atr']:.2f}")
        
        # Determine signal quality
        if confidence >= 95.0: # Ultra High Confidence
            quality = SignalQuality.EXCELLENT
        elif confidence >= 90.0:
            quality = SignalQuality.STRONG
        else:
            quality = SignalQuality.GOOD
        
        # Create signal
        signal_id = f"{symbol}_{action}_{int(datetime.now().timestamp())}"
        timestamp = datetime.now(pytz.UTC)
        
        signal = TradeSignal(
            signal_id=signal_id,
            symbol=symbol,
            action=action,
            confidence=confidence,
            quality=quality,
            entry_price=levels['entry'],
            stop_loss=levels['stop_loss'],
            take_profit_1=levels['tp1'],
            take_profit_2=levels['tp2'],
            take_profit_3=levels['tp3'],
            stop_loss_distance=levels['sl_distance'],
            tp1_distance=levels['tp1_distance'],
            tp2_distance=levels['tp2_distance'],
            tp3_distance=levels['tp3_distance'],
            stake_usd=self.stake_usd,
            risk_reward_ratio=levels['rr_ratio'],
            atr_value=levels['atr'],
            volatility_level=vol_level,
            trend_strength=trend_strength,
            timestamp=timestamp,
            strategy_components=reasons # Now passed as the last argument
        )
        
        logger.info(f"✅ Signal created: {signal_id}")
        return signal
    
    def is_duplicate_signal(self, signal: TradeSignal) -> bool:
        """Check for duplicate signals with extended cooldown period (45 mins)"""
        for recent in self.recent_signals:
            if recent.symbol == signal.symbol and recent.action == signal.action:
                time_diff = (signal.timestamp - recent.timestamp).total_seconds()
                # 45-minute cooldown (2700 seconds)
                if time_diff < 2700: 
                    logger.info(f"⚠️ Duplicate signal blocked (cooldown: {2700 - time_diff:.0f}s remaining)")
                    return True
        return False
    
    def run_analysis_cycle(self) -> int:
        """Run complete analysis cycle for all symbols"""
        logger.info("\n" + "=" * 80)
        logger.info("🚀 STARTING ANALYSIS CYCLE (v3.0)")
        logger.info(f"⏰ {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info("=" * 80)
        
        signals_generated = 0
        
        for symbol in self.SYMBOLS:
            try:
                logger.info(f"\n🔍 Processing {symbol}...")
                signal = self.analyze_symbol(symbol)
                
                if signal:
                    if self.is_duplicate_signal(signal):
                        logger.info(f"⏭️ Skipping duplicate signal for {symbol}")
                        continue
                    
                    # FINAL SANITY CHECK: R:R must be acceptable overall
                    if signal.risk_reward_ratio < 2.5: 
                         logger.warning(f"⚠️ Signal R:R 1:{signal.risk_reward_ratio:.2f} is too low overall - Rejecting")
                         continue

                    if self.notifier.send_signal(signal):
                        self.recent_signals.append(signal)
                        signals_generated += 1
                        logger.info(f"✅ Signal sent successfully: {signal.signal_id}")
                    else:
                        logger.error(f"❌ Failed to send signal for {symbol}")
                else:
                    logger.info(f"ℹ️ No signal generated for {symbol}")
            
            except Exception as e:
                logger.error(f"❌ Error analyzing {symbol}: {e}", exc_info=True)
        
        logger.info("\n" + "=" * 80)
        logger.info(f"✅ ANALYSIS CYCLE COMPLETE (v3.0)")
        logger.info(f"📊 Signals Generated: {signals_generated}/{len(self.SYMBOLS)}")
        logger.info(f"📈 Total Signals in Memory: {len(self.recent_signals)}")
        logger.info("=" * 80)
        
        return signals_generated


# ============================================================================
# MAIN ENTRY POINT (No Change)
# ============================================================================

def main():
    """Main entry point with proper error handling"""
    logger.info("\n" + "=" * 80)
    logger.info("SYNTHETIC INDICES TRADING BOT v3.0 - STARTING")
    logger.info("=" * 80)
    
    # Validate environment variables
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    main_chat_id = os.getenv('MAIN_CHAT_ID')
    
    if not telegram_token:
        logger.error("❌ TELEGRAM_BOT_TOKEN environment variable not set")
        logger.error("   Please set it with: export TELEGRAM_BOT_TOKEN='your_token'")
        return 1
    
    if not main_chat_id:
        logger.error("❌ MAIN_CHAT_ID environment variable not set")
        logger.error("   Please set it with: export MAIN_CHAT_ID='your_chat_id'")
        return 1
    
    # Parse stake amount with proper validation
    try:
        stake_usd_str = os.getenv('STAKE_USD', '10')
        stake_usd = float(stake_usd_str) if stake_usd_str else 10.0
        
        if stake_usd <= 0:
            logger.error("❌ STAKE_USD must be greater than 0")
            return 1
        
        if stake_usd > 1000:
            logger.warning(f"⚠️ STAKE_USD is very high: ${stake_usd:.2f}")
            logger.warning("   Consider using a more conservative stake amount")
    
    except ValueError:
        logger.error(f"❌ Invalid STAKE_USD value: '{stake_usd_str}'")
        logger.error("   Using default value: $10.00")
        stake_usd = 10.0
    
    try:
        # Initialize bot
        bot = SyntheticTradingBot(
            telegram_token=telegram_token,
            main_chat_id=main_chat_id,
            stake_usd=stake_usd
        )
        
        # Run analysis cycle
        signals_count = bot.run_analysis_cycle()
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✅ Bot execution completed successfully")
        logger.info(f"📊 Total signals generated: {signals_count}")
        logger.info(f"⏰ Execution time: {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info("=" * 80)
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("\n⚠️ Bot stopped by user (Ctrl+C)")
        return 0
    
    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("❌ FATAL ERROR")
        logger.error("=" * 80)
        logger.error(f"Error: {e}", exc_info=True)
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())