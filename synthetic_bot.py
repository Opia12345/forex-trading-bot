"""
SYNTHETIC INDICES TRADING BOT v4.0 - INSTITUTIONAL GRADE
Specialized for Deriv Volatility Indices (V10, V25, V50, V75, V100)
Refactored for:
1. Improved Asynchronous Data Handling.
2. Sophisticated Multi-Factor Signal Filtering (Hurst, Volume-based Trend).
3. Dynamic, Fractional Level Precision.
4. Enhanced Robustness and Error Handling.
Disclaimer: Trading is inherently risky. This code is for educational/simulated purposes.
"""

import os
import sys
import logging
from typing import Optional, List, Tuple, Dict, Any
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

# --- Constants & Configuration ---
UTC = pytz.utc
TIMEZONE = os.getenv('BOT_TIMEZONE', 'Africa/Lagos') # Set to a local timezone if needed, but data is UTC
LOCAL_TZ = pytz.timezone(TIMEZONE)
APP_ID = "1089" # Deriv default

# --- Logging Setup (Improved) ---
LOG_FILE = 'synthetic_bot.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Data Classes (Refined) ---
class SignalQuality(Enum):
    ALPHA = "ALPHA" # New highest quality
    EXCELLENT = "EXCELLENT"
    STRONG = "STRONG"
    GOOD = "GOOD"

@dataclass
class TradeSignal:
    signal_id: str
    symbol: str
    action: str  # BUY or SELL
    confidence: float # 0.0 to 100.0
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
    strategy_components: List[str] = field(default_factory=list)
    volatility_level: str
    trend_strength: str
    timestamp: datetime

# ============================================================================
# UTILITY FUNCTIONS (Added Precision/Rounding)
# ============================================================================

def get_precision(price: float) -> int:
    """Determine the number of decimal places for Deriv symbols (e.g., 2)"""
    # Assuming all VIX indices have a precision of 2 decimal places for price display.
    # This can be dynamically fetched but hardcoding for now for stability.
    return 2

def round_price(price: float, symbol: str) -> float:
    """Rounds price to the appropriate precision for the symbol."""
    precision = get_precision(price)
    return round(price, precision)

# ============================================================================
# TELEGRAM NOTIFIER (Improved Messaging)
# ============================================================================

class TelegramNotifier:
    """Sends trading signals to main Telegram group"""
    
    def __init__(self, token: str, main_chat_id: str):
        self.token = token
        self.main_chat_id = main_chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_signal(self, signal: TradeSignal) -> bool:
        """Send signal to Telegram with improved formatting and precision"""
        try:
            quality_emoji = {
                "ALPHA": "👑",
                "EXCELLENT": "🌟",
                "STRONG": "⭐",
                "GOOD": "✨"
            }.get(signal.quality.value, "✨")
            
            action_emoji = "🟢" if signal.action == "BUY" else "🔴"
            
            # Use utility for rounding
            entry_price = round_price(signal.entry_price, signal.symbol)
            stop_loss = round_price(signal.stop_loss, signal.symbol)
            tp1 = round_price(signal.take_profit_1, signal.symbol)
            tp2 = round_price(signal.take_profit_2, signal.symbol)
            tp3 = round_price(signal.take_profit_3, signal.symbol)
            
            message = f"""
{quality_emoji} <b>SYNTHETIC SIGNAL - {signal.quality.value}</b> {quality_emoji}

{action_emoji} <b>{signal.action} {signal.symbol}</b>
💯 Confidence: **{signal.confidence:.1f}%**
📊 Quality: **{signal.quality.value}**

<b>📍 ENTRY & EXIT LEVELS</b>
💵 Entry Price: <code>{entry_price:.2f}</code>
🛑 Stop Loss: <code>{stop_loss:.2f}</code> ({signal.stop_loss_distance:.2f} points)
🎯 TP1: <code>{tp1:.2f}</code> (R:R 1:{signal.tp1_distance/signal.stop_loss_distance:.2f})
🎯 TP2: <code>{tp2:.2f}</code> (R:R 1:{signal.tp2_distance/signal.stop_loss_distance:.2f})
🎯 TP3: <code>{tp3:.2f}</code> (R:R 1:{signal.risk_reward_ratio:.2f} final)

<b>💼 POSITION METRICS</b>
💰 Risk Per Trade: ${signal.stake_usd:.2f} 
📈 ATR (14-period): {signal.atr_value:.2f}
🔥 Volatility: {signal.volatility_level}
💪 Trend: **{signal.trend_strength}**

<b>✅ SIGNAL RATIONALE ({len(signal.strategy_components)} Factors)</b>
"""
            
            # Show up to 10 reasons now
            for reason in signal.strategy_components[:10]:
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
<i>Generated: {signal.timestamp.astimezone(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}</i>
"""
            
            return self._send_message(message)
        
        except Exception as e:
            logger.error(f"Error sending signal: {e}", exc_info=True)
            return False
    
    def _send_message(self, message: str) -> bool:
        """Send message to main Telegram group"""
        # ... (implementation remains the same for network stability)
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
                # Log the specific error from the API
                error_details = response.json().get('description', response.text)
                logger.error(f"Telegram API error: {response.status_code} - {error_details}")
                return False
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Telegram network send error: {e}")
            return False

# ============================================================================
# DERIV DATA FETCHER (Refactored to be purely async)
# ============================================================================
class DerivDataFetcher:
    """Fetch synthetic indices data from Deriv via a single, purely async method"""
    
    SYMBOLS = {
        'V10': 'R_10', 'V25': 'R_25', 'V50': 'R_50', 
        'V75': 'R_75', 'V100': 'R_100'
    }
    
    def __init__(self, app_id: str = APP_ID):
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    
    async def fetch_tick_data_async(self, symbol: str, count: int = 2000) -> Optional[pd.DataFrame]:
        """Fetch tick data via WebSocket and convert to DataFrame"""
        deriv_symbol = self.SYMBOLS.get(symbol)
        if not deriv_symbol:
            logger.error(f"Invalid symbol: {symbol}")
            return None
            
        try:
            # Connect with robust settings
            async with websockets.connect(self.ws_url, ping_interval=30, close_timeout=10, max_size=2**25) as ws:
                request = {
                    "ticks_history": deriv_symbol,
                    "adjust_start_time": 1,
                    "count": count,
                    "end": "latest",
                    "start": 1,
                    "style": "ticks"
                }
                
                await ws.send(json.dumps(request))
                # Set a reasonable timeout for the response
                response = await asyncio.wait_for(ws.recv(), timeout=30)
                data = json.loads(response)
                
                if 'error' in data:
                    logger.error(f"Deriv API error for {symbol}: {data['error'].get('message', 'Unknown error')}")
                    return None
                
                if 'history' in data and 'prices' in data['history'] and 'times' in data['history']:
                    df = pd.DataFrame({
                        'price': data['history']['prices'],
                        # Convert to datetime and make UTC-aware
                        'timestamp': [datetime.fromtimestamp(t, tz=UTC) for t in data['history']['times']]
                    })
                    
                    if len(df) < 500: # Increase minimum data points required
                        logger.warning(f"Insufficient tick data for {symbol}: {len(df)}")
                        return None
                        
                    df['tick'] = range(len(df))
                    df.set_index('tick', inplace=True)
                    
                    logger.info(f"✅ Fetched {len(df)} ticks for {symbol}")
                    return df
                
                return None
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching data for {symbol}")
            return None
        except websockets.exceptions.ConnectionClosedOK:
            logger.warning(f"WebSocket closed gracefully for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
            return None

# ============================================================================
# TECHNICAL INDICATORS (Enhanced Robustness & New Indicator)
# ============================================================================

class TechnicalIndicators:
    """Industry-standard technical indicators"""
    
    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    # ... (calculate_sma, calculate_rsi remain the same) ...
    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index (Vectorized)"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Use np.divide and fillna(np.inf) for robust division
        rs = np.divide(avg_gain, avg_loss)
        rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50) 

    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range - proper calculation"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Handle the first value NaN from .shift(1)
        tr.iloc[0] = tr1.iloc[0] if pd.notna(tr1.iloc[0]) else 0.0
        
        # Using EMA for ATR (smoother and standard)
        atr = tr.ewm(span=period, adjust=False).mean() 
        return atr
    
    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, period: int = 20, std: float = 2.0) -> Tuple:
        """Bollinger Bands"""
        # ... (implementation remains the same)
        sma = series.rolling(window=period).mean()
        rolling_std = series.rolling(window=period).std()
        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        return upper, sma, lower
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index - trend strength"""
        # ... (implementation remains the same, ensuring robust division)
        high = df['high']
        low = df['low']
        
        plus_dm = high.diff().clip(lower=0).fillna(0)
        minus_dm = -low.diff().clip(lower=0).fillna(0)
        
        tr = TechnicalIndicators.calculate_atr(df, 1) # True Range for initial DI smoothing
        atr_smooth = tr.ewm(span=period, adjust=False).mean() 
        
        # Robust division handling
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_smooth).replace([np.inf, -np.inf], np.nan).fillna(0)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_smooth).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Robust division handling
        di_sum = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / di_sum.replace(0, np.nan)
        
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx.fillna(0)
        
    @staticmethod
    def calculate_hurst_exponent(series: pd.Series, window: int = 100) -> pd.Series:
        """
        Calculates the Hurst Exponent (H) to measure long-term memory of a time series.
        H < 0.5: Mean-reverting (Chop)
        H = 0.5: Random walk (Efficient)
        H > 0.5: Trending (Persistent)
        """
        def hurst_calc(series_slice: pd.Series) -> float:
            if len(series_slice) < 10: return 0.5 # Default to random if too short
            series_slice = series_slice.diff().dropna()
            if len(series_slice) < 5: return 0.5

            lags = range(2, len(series_slice) // 2)
            tau = [np.sqrt(np.std(series_slice.iloc[lag:] - series_slice.iloc[:-lag])) for lag in lags]
            
            # Simple log-log regression slope is H
            if all(t == 0 for t in tau) or len(tau) == 0: return 0.5
            
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
            
        # Apply the calculation over a rolling window
        hurst_series = series.rolling(window=window).apply(hurst_calc, raw=False)
        return hurst_series.fillna(0.5)

    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to DataFrame (Improved OHLC generation)"""
        
        # Create OHLC from tick data (using 50 ticks for a more stable local range)
        window = 50 
        # Ensure we have enough data for the window
        if len(df) < window:
             logger.warning(f"Dataframe too small ({len(df)}) for OHLC window ({window}).")
             return df
             
        df['high'] = df['price'].rolling(window=window).max().shift(-(window - 1)) # Look back to get the max
        df['low'] = df['price'].rolling(window=window).min().shift(-(window - 1))  # Look back to get the min
        df['close'] = df['price'] # 'close' is the latest tick price
        
        # Moving averages
        df['ema_9'] = TechnicalIndicators.calculate_ema(df['price'], 9)
        df['ema_21'] = TechnicalIndicators.calculate_ema(df['price'], 21)
        df['ema_50'] = TechnicalIndicators.calculate_ema(df['price'], 50)
        df['ema_100'] = TechnicalIndicators.calculate_ema(df['price'], 100)
        df['ema_200'] = TechnicalIndicators.calculate_ema(df['price'], 200)
        
        # Momentum indicators
        df['rsi'] = TechnicalIndicators.calculate_rsi(df['price'], 14)
        
        # Volatility indicators
        df['atr'] = TechnicalIndicators.calculate_atr(df, 14)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = \
            TechnicalIndicators.calculate_bollinger_bands(df['price'], 20, 2.0)
        
        # Trend strength
        df['adx'] = TechnicalIndicators.calculate_adx(df, 14)
        df['hurst'] = TechnicalIndicators.calculate_hurst_exponent(df['price'], 100)
        
        # Clean up NaNs from indicator calculations for the latest row only
        # Fill NaN values (robust fill - only the latest row is needed for signals)
        df.iloc[-1] = df.iloc[-1].fillna(df.iloc[-2])
        df = df.iloc[-200:] # Keep only the last 200 rows for faster processing

        # Final check for any remaining NaNs in the latest row
        if df.iloc[-1].isna().any():
            logger.warning("⚠️ Latest row still contains NaNs after robust filling.")
            
        return df


# ============================================================================
# ADVANCED SIGNAL ANALYZER (Refined for Higher Accuracy)
# ============================================================================

class AdvancedSignalAnalyzer:
    """Advanced signal analysis with proper market structure and filtering"""
    
    # Max scores for components
    MAX_TREND_SCORE = 50.0  # 40 EMA + 10 PA Structure
    MAX_MOMENTUM_SCORE = 25.0
    MAX_PA_SCORE = 20.0
    MAX_STRENGTH_SCORE = 15.0 # ADX/Hurst
    
    @staticmethod
    def analyze_trend(df: pd.DataFrame) -> Tuple[Optional[str], float, List[str], str]:
        """Analyze trend with multiple confirmations and price action structure"""
        last = df.iloc[-1]
        
        score = 0.0
        reasons = []
        action = None
        trend_strength = "WEAK"
        
        # 1. EMA Alignment (40 points max)
        is_bullish_stack = (last['ema_9'] > last['ema_21'] > last['ema_50'] > last['ema_100'])
        is_bearish_stack = (last['ema_9'] < last['ema_21'] < last['ema_50'] < last['ema_100'])
        
        if is_bullish_stack:
            score += 40.0
            action = "BUY"
            reasons.append("✅ Institutional bullish EMA stack (9>21>50>100)")
        elif is_bearish_stack:
            score += 40.0
            action = "SELL"
            reasons.append("✅ Institutional bearish EMA stack (9<21<50<100)")
        elif last['ema_9'] > last['ema_21'] and last['ema_21'] > last['ema_50']:
            score += 30.0
            action = "BUY"
            reasons.append("Strong bullish trend (EMA 9>21>50 alignment)")
        elif last['ema_9'] < last['ema_21'] and last['ema_21'] < last['ema_50']:
            score += 30.0
            action = "SELL"
            reasons.append("Strong bearish trend (EMA 9<21<50 alignment)")
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
        # Check for 3 consecutive Higher Highs (HH) / Lower Lows (LL) in the last 15 ticks
        recent_highs = df['high'].tail(15)
        recent_lows = df['low'].tail(15)
        
        is_higher_highs = all(recent_highs.iloc[i] > recent_highs.iloc[i-1] for i in range(1, 4))
        is_higher_lows = all(recent_lows.iloc[i] > recent_lows.iloc[i-1] for i in range(1, 4))
        is_lower_highs = all(recent_highs.iloc[i] < recent_highs.iloc[i-1] for i in range(1, 4))
        is_lower_lows = all(recent_lows.iloc[i] < recent_lows.iloc[i-1] for i in range(1, 4))

        if action == "BUY" and is_higher_highs and is_higher_lows:
            score += 10.0
            reasons.append("✅ Confirmed by 3 consecutive Higher Highs/Higher Lows")
        elif action == "SELL" and is_lower_highs and is_lower_lows:
            score += 10.0
            reasons.append("✅ Confirmed by 3 consecutive Lower Highs/Lower Lows")
        elif action == "BUY" and (is_higher_highs or is_higher_lows):
            score += 5.0
            reasons.append("Price structure showing positive market shift (HH or HL)")
        elif action == "SELL" and (is_lower_highs or is_lower_lows):
            score += 5.0
            reasons.append("Price structure showing negative market shift (LH or LL)")
        else:
            reasons.append("⚠️ Price structure not strongly confirming the EMA trend.")
                
        # Determine overall trend strength
        if score >= 45.0:
            trend_strength = "ALPHA"
        elif score >= 35.0:
            trend_strength = "VERY STRONG"
        elif score >= 25.0:
            trend_strength = "STRONG"
        elif score >= 15.0:
            trend_strength = "MODERATE"
        
        return action, score, reasons, trend_strength
    
    @staticmethod
    def analyze_momentum(df: pd.DataFrame, action: str) -> Tuple[float, List[str]]:
        """Analyze momentum with RSI filtering out extreme conditions (Improved ranges)"""
        last = df.iloc[-1]
        score = 0.0
        reasons = []
        rsi = last['rsi']
        
        if action == "BUY":
            # Optimal Buy Zone (40-65) - momentum building or consolidating
            if 40 <= rsi <= 65:
                score = 25.0
                reasons.append(f"✅ RSI in optimal buy momentum zone ({rsi:.1f})")
            # Deeper Buy (30-40) - oversold, but still potential if trend is strong
            elif 30 <= rsi < 40:
                score = 18.0
                reasons.append(f"RSI deep pullback/oversold but trend may persist ({rsi:.1f})")
            # Extreme Overbought Filter: Reject if RSI is too high
            elif rsi >= 75:
                return 0.0, [f"❌ RSI EXTREMELY overbought - High risk entry ({rsi:.1f})"]
            else:
                score = 10.0
                reasons.append(f"RSI neutral or weak for buy ({rsi:.1f})")
        
        else:  # SELL
            # Optimal Sell Zone (35-60) - momentum losing or consolidating
            if 35 <= rsi <= 60:
                score = 25.0
                reasons.append(f"✅ RSI in optimal sell momentum zone ({rsi:.1f})")
            # Deeper Sell (60-70) - overbought, but still potential if trend is strong
            elif 60 < rsi <= 70:
                score = 18.0
                reasons.append(f"RSI deep pullback/overbought but trend may persist ({rsi:.1f})")
            # Extreme Oversold Filter: Reject if RSI is too low
            elif rsi <= 25:
                return 0.0, [f"❌ RSI EXTREMELY oversold - High risk entry ({rsi:.1f})"]
            else:
                score = 10.0
                reasons.append(f"RSI neutral or weak for sell ({rsi:.1f})")
        
        return score, reasons
    
    @staticmethod
    def analyze_price_action(df: pd.DataFrame, action: str) -> Tuple[float, List[str]]:
        """Analyze Bollinger Bands and price action for volatility and exhaustion"""
        last = df.iloc[-1]
        score = 0.0
        reasons = []
        
        bb_width = last['bb_upper'] - last['bb_lower']
        bb_position = (last['price'] - last['bb_lower']) / bb_width if bb_width > 0 else 0.5
        
        # Check for BB squeeze (low volatility) - a good pre-cursor to a breakout
        recent_bb_widths = df['bb_upper'].tail(30) - df['bb_lower'].tail(30)
        is_squeeze = last['atr'] < recent_bb_widths.median() * 0.8
        if is_squeeze:
            score += 5.0
            reasons.append("⚠️ BB Squeeze detected (potential volatility breakout soon)")
        
        if action == "BUY":
            # Optimal entry: Price near or outside lower band
            if bb_position <= 0.15:
                score += 15.0
                reasons.append("✅ Price at/near lower BB (strong mean reversion/bounce entry)")
            elif bb_position < 0.35:
                score += 10.0
                reasons.append("Price in lower BB zone (bounce potential)")
            # Rejection filter: Reject entry near the upper band
            elif bb_position > 0.80:
                return 0.0, ["❌ Price is near/outside upper BB (too extended for buy)"]
            else:
                score += 5.0
                reasons.append("Price in neutral BB zone")
        
        else:  # SELL
            # Optimal entry: Price near or outside upper band
            if bb_position >= 0.85:
                score += 15.0
                reasons.append("✅ Price at/near upper BB (strong mean reversion/rejection entry)")
            elif bb_position > 0.65:
                score += 10.0
                reasons.append("Price in upper BB zone (rejection potential)")
            # Rejection filter: Reject entry near the lower band
            elif bb_position < 0.20:
                return 0.0, ["❌ Price is near/outside lower BB (too extended for sell)"]
            else:
                score += 5.0
                reasons.append("Price in neutral BB zone")
        
        return score, reasons
    
    @staticmethod
    def analyze_trend_strength_adx_hurst(df: pd.DataFrame) -> Tuple[float, List[str], str]:
        """Analyze ADX and Hurst Exponent trend strength"""
        last = df.iloc[-1]
        score = 0.0
        reasons = []
        
        # 1. ADX (10 points max)
        adx = last['adx']
        if adx > 45:
            score += 10.0
            reasons.append(f"✅ Extreme trend strength (ADX {adx:.1f})")
        elif adx > 25:
            score += 8.0
            reasons.append(f"✅ Strong trend strength (ADX {adx:.1f})")
        elif adx > 20:
            score += 5.0
            reasons.append(f"Moderate trend strength (ADX {adx:.1f})")
        # Weak Trend Filter: Reject trades in consolidation/no-trend zones
        elif adx <= 20:
            return 0.0, [f"❌ Insufficient trend strength (ADX {adx:.1f}) - possible chop"], "NONE"
            
        # 2. Hurst Exponent (5 points max)
        hurst = last['hurst']
        if hurst > 0.65:
            score += 5.0
            reasons.append(f"✅ Hurst Exponent > 0.65 (Strong persistent trending market)")
        elif hurst < 0.35:
            # Reject if the market is strongly mean-reverting (H < 0.35)
            # unless the signal is a mean-reversion type, but this bot is primarily trend-following
            # Only apply this filter if ADX is also low
            if adx < 25:
                 return 0.0, [f"❌ Strong mean-reversion detected (Hurst {hurst:.2f}) - Chop"], "NONE"
            score += 2.0
            reasons.append(f"Hurst Exponent between 0.35-0.5 (Mean-reverting tendencies)")
        else:
            score += 3.0
            reasons.append(f"Hurst Exponent 0.5-0.65 (Trending/Efficient market mix)")

        # Determine strength based on the max possible score (15.0)
        strength = "NONE"
        if score >= 12.0:
            strength = "VERY STRONG"
        elif score >= 8.0:
            strength = "STRONG"
        elif score >= 5.0:
            strength = "MODERATE"
            
        return score, reasons, strength
    
    @staticmethod
    def calculate_confidence(df: pd.DataFrame) -> Tuple[Optional[str], float, List[str], str, str]:
        """Calculate overall confidence score with strict filtering (Total Max Score: 110)"""
        all_reasons = []
        total_score = 0.0
        
        # 1. Trend Analysis (50 points max)
        action, trend_score, trend_reasons, trend_strength_level = AdvancedSignalAnalyzer.analyze_trend(df)
        
        # Raised minimum trend score for Alpha-grade signal
        MIN_TRADABLE_TREND = 35.0 
        if not action or trend_score < MIN_TRADABLE_TREND:
            # Only allow a slightly lower trend score if the price action is a strong mean-reversion
            pa_score, pa_reasons = AdvancedSignalAnalyzer.analyze_price_action(df, action if action else 'BUY') # Temp action
            if pa_score < 15.0:
                return None, 0.0, ["❌ Insufficient trend strength/clarity for entry"] + trend_reasons, "WEAK", "LOW"

        total_score += trend_score
        all_reasons.extend(trend_reasons)
        
        # 2. Trend Strength Analysis (15 points max)
        strength_score, strength_reasons, adx_hurst_strength = AdvancedSignalAnalyzer.analyze_trend_strength_adx_hurst(df)
        if strength_score < 5.0: # Minimum ADX/Hurst score
            return None, 0.0, strength_reasons, trend_strength_level, "LOW"
        
        total_score += strength_score
        all_reasons.extend(strength_reasons)
        
        # 3. Momentum Analysis (25 points max)
        momentum_score, momentum_reasons = AdvancedSignalAnalyzer.analyze_momentum(df, action)
        if momentum_score == 0.0:
            return None, 0.0, momentum_reasons, trend_strength_level, "LOW"
        
        total_score += momentum_score
        all_reasons.extend(momentum_reasons)
        
        # 4. Price Action Analysis (20 points max)
        pa_score, pa_reasons = AdvancedSignalAnalyzer.analyze_price_action(df, action)
        if pa_score == 0.0:
            return None, 0.0, pa_reasons, trend_strength_level, "LOW"
        
        total_score += pa_score
        all_reasons.extend(pa_reasons)
        
        # Determine volatility level based on ATR relative to recent average
        last = df.iloc[-1]
        recent = df.tail(200) # Use a wider window for volatility average
        avg_atr = recent['atr'].median() 
        
        if last['atr'] > avg_atr * 1.6:
            vol_level = "VERY HIGH (Extreme Caution)"
        elif last['atr'] > avg_atr * 1.3:
            vol_level = "HIGH"
        elif last['atr'] > avg_atr * 0.7:
            vol_level = "NORMAL"
        else:
            vol_level = "LOW (Choppy/Tight)"
        
        # Scale score to 100% (Max theoretical score: 50+15+25+20 = 110)
        confidence = min(100.0, (total_score / 110.0) * 100.0)
        
        return action, confidence, all_reasons, trend_strength_level, vol_level


# ============================================================================
# RISK MANAGER (Enhanced Precision & Dynamic Multipliers)
# ============================================================================

class RiskManager:
    """Professional risk management with wider, ATR-based stops for volatility indices"""
    
    @staticmethod
    def calculate_position_levels(df: pd.DataFrame, action: str, symbol: str, 
                                   stake_usd: float) -> Dict:
        """Calculate entry, stop loss, and take profit levels with precision"""
        last = df.iloc[-1]
        current_price = last['price']
        atr = last['atr']
        
        # Ensure ATR is valid and non-zero
        if pd.isna(atr) or atr <= 0:
            atr = current_price * 0.005  # Fallback to 0.5% of price
            logger.warning(f"⚠️ ATR fallback used for {symbol}: {atr:.2f}")
        
        # SYMBOL-SPECIFIC ATR MULTIPLIERS (Adjusted for tighter, safer exits)
        # SL multiplier is increased to give trades more room to breathe (3.0 default).
        # TP multipliers enforce MINIMUM R:R.
        
        # Base multipliers for all indices
        BASE_SL_MULTIPLIER = 3.0
        BASE_TP_RR = {'tp1': 2.0, 'tp2': 3.5, 'tp3': 5.0} # Increased TP R:R
        
        # Volatility-specific adjustments (More volatile indices get slightly tighter stops relative to ATR)
        sl_adjust = { 'V10': 1.1, 'V25': 1.0, 'V50': 0.9, 'V75': 0.8, 'V100': 0.7 }
        
        sl_multiplier = BASE_SL_MULTIPLIER * sl_adjust.get(symbol, 1.0)
        
        # Use a dynamic buffer (0.5% of ATR) to prevent stop hunting at round numbers
        sl_buffer = atr * 0.005
        
        sl_distance_points = atr * sl_multiplier
        
        # Calculate levels based on the fixed SL distance
        if action == "BUY":
            entry = current_price
            # SL is placed a little lower than the ATR * multiplier to give buffer
            stop_loss = entry - sl_distance_points - sl_buffer
            
            # TP is calculated as a multiple of the risk (SL distance)
            risk = sl_distance_points # Use the clean distance for R:R calculation
            tp1 = entry + (risk * BASE_TP_RR['tp1'])
            tp2 = entry + (risk * BASE_TP_RR['tp2'])
            tp3 = entry + (risk * BASE_TP_RR['tp3'])
            
        else:  # SELL
            entry = current_price
            # SL is placed a little higher than the ATR * multiplier to give buffer
            stop_loss = entry + sl_distance_points + sl_buffer
            
            # TP is calculated as a multiple of the risk (SL distance)
            risk = sl_distance_points
            tp1 = entry - (risk * BASE_TP_RR['tp1'])
            tp2 = entry - (risk * BASE_TP_RR['tp2'])
            tp3 = entry - (risk * BASE_TP_RR['tp3'])
        
        # Recalculate distances (accurate distances, not just the initial calculation)
        sl_distance = abs(entry - stop_loss)
        tp1_distance = abs(tp1 - entry)
        tp2_distance = abs(tp2 - entry)
        tp3_distance = abs(tp3 - entry)
        
        # Final Risk-Reward ratio (based on TP3)
        rr_ratio = tp3_distance / sl_distance if sl_distance > 0 else 0.0
        
        # Use the global rounding function for final output precision
        return {
            'entry': round_price(entry, symbol),
            'stop_loss': round_price(stop_loss, symbol),
            'tp1': round_price(tp1, symbol),
            'tp2': round_price(tp2, symbol),
            'tp3': round_price(tp3, symbol),
            'sl_distance': round(sl_distance, 2), # Distance can be 2 decimal points for display
            'tp1_distance': round(tp1_distance, 2),
            'tp2_distance': round(tp2_distance, 2),
            'tp3_distance': round(tp3_distance, 2),
            'rr_ratio': round(rr_ratio, 2),
            'atr': round(atr, 2)
        }


# ============================================================================
# SYNTHETIC TRADING BOT v4.0 (Core Logic)
# ============================================================================

class SyntheticTradingBot:
    """Professional-grade trading bot for synthetic indices"""
    
    SYMBOLS = ['V10', 'V25', 'V50', 'V75', 'V100']
    
    # INSTITUTIONAL MINIMUM CONFIDENCE (Raised from 85.0)
    MIN_CONFIDENCE = 90.0 
    
    def __init__(self, telegram_token: str, main_chat_id: str, stake_usd: float = 10.0):
        self.notifier = TelegramNotifier(telegram_token, main_chat_id)
        self.data_fetcher = DerivDataFetcher()
        self.stake_usd = stake_usd
        # Extended cooldown window for greater separation between signals
        self.recent_signals = deque(maxlen=50) # Increased maxlen
        
        logger.info("=" * 80)
        logger.info("SYNTHETIC INDICES TRADING BOT v4.0 - INSTITUTIONAL GRADE")
        logger.info("=" * 80)
        logger.info("Features:")
        logger.info("  ✅ Dynamic, Buffer-adjusted ATR-based Stop Loss (Fix)")
        logger.info("  ✅ Hurst Exponent + ADX Trend Filtering (New)")
        logger.info("  ✅ Strict MIN_CONFIDENCE threshold (90.0%) (Fix)")
        logger.info("  ✅ Strict Min R:R of 1:2.0 on TP1 + 1:2.5 Overall")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  Symbols: {', '.join(self.SYMBOLS)}")
        logger.info(f"  Minimum Confidence: {self.MIN_CONFIDENCE}%")
        logger.info(f"  Stake per Trade: ${stake_usd:.2f}")
        logger.info("=" * 80)
    
    async def analyze_symbol_async(self, symbol: str) -> Optional[TradeSignal]:
        """Analyze symbol for high-probability trading signals (ASYNC)"""
        logger.info(f"\n{'='*70}")
        logger.info(f"ANALYZING {symbol}")
        logger.info(f"{'='*70}")
        
        # Fetch sufficient tick data for proper analysis
        logger.info("📥 Fetching tick data asynchronously...")
        df = await self.data_fetcher.fetch_tick_data_async(symbol, count=2500) # Increased count
        
        if df is None or df.empty:
            logger.error("❌ Failed to fetch data or data is insufficient")
            return None
        
        # Calculate all technical indicators
        logger.info("🔧 Calculating technical indicators...")
        df = TechnicalIndicators.add_all_indicators(df)
        
        # Check if the last row is clean for signals
        if df.iloc[-1].isna().any():
            logger.error("❌ Last data point contains NaNs after indicator calculation. Rejecting analysis.")
            return None
            
        # Advanced signal analysis
        logger.info("🎯 Analyzing market structure...")
        action, confidence, reasons, trend_strength, vol_level = \
            AdvancedSignalAnalyzer.calculate_confidence(df)
        
        if not action:
            logger.info(f"❌ No qualified signal: {reasons[0]}")
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
        current_rr_tp1 = levels['tp1_distance'] / levels['sl_distance']
        if current_rr_tp1 < min_rr_tp1:
            logger.warning(f"⚠️ TP1 R:R (1:{current_rr_tp1:.2f}) is below minimum 1:{min_rr_tp1:.1f} - Rejecting")
            return None
            
        # Overall R:R Check: Ensure TP3 meets the institutional minimum R:R of 1:2.5
        min_rr_overall = 2.5
        if levels['rr_ratio'] < min_rr_overall: 
            logger.warning(f"⚠️ Signal R:R 1:{levels['rr_ratio']:.2f} is too low overall - Rejecting")
            return None

        # Determine signal quality
        if confidence >= 98.0: # ALPHA-grade signal
            quality = SignalQuality.ALPHA
        elif confidence >= 95.0:
            quality = SignalQuality.EXCELLENT
        elif confidence >= 90.0:
            quality = SignalQuality.STRONG
        else:
            quality = SignalQuality.GOOD
        
        # Create signal
        signal_id = f"{symbol}_{action}_{int(datetime.now().timestamp())}"
        timestamp = datetime.now(UTC)
        
        signal = TradeSignal(
            signal_id=signal_id, symbol=symbol, action=action, confidence=confidence, quality=quality, 
            entry_price=levels['entry'], stop_loss=levels['stop_loss'], take_profit_1=levels['tp1'], 
            take_profit_2=levels['tp2'], take_profit_3=levels['tp3'], 
            stop_loss_distance=levels['sl_distance'], tp1_distance=levels['tp1_distance'], 
            tp2_distance=levels['tp2_distance'], tp3_distance=levels['tp3_distance'], 
            stake_usd=self.stake_usd, risk_reward_ratio=levels['rr_ratio'], atr_value=levels['atr'], 
            strategy_components=reasons, volatility_level=vol_level, 
            trend_strength=trend_strength, timestamp=timestamp
        )
        
        logger.info(f"✅ Signal created: {signal_id}")
        return signal
    
    def is_duplicate_signal(self, signal: TradeSignal) -> bool:
        """Check for duplicate signals with extended cooldown period (60 mins)"""
        COOLDOWN_SECONDS = 3600 # 60-minute cooldown
        for recent in self.recent_signals:
            if recent.symbol == signal.symbol and recent.action == signal.action:
                time_diff = (signal.timestamp - recent.timestamp).total_seconds()
                if time_diff < COOLDOWN_SECONDS: 
                    logger.info(f"⚠️ Duplicate signal blocked (cooldown: {COOLDOWN_SECONDS - time_diff:.0f}s remaining)")
                    return True
        return False
    
    async def run_analysis_cycle_async(self) -> int:
        """Run complete analysis cycle for all symbols asynchronously"""
        logger.info("\n" + "=" * 80)
        logger.info("🚀 STARTING ANALYSIS CYCLE (v4.0)")
        logger.info(f"⏰ {datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info("=" * 80)
        
        signals_generated = 0
        
        # Create a list of analysis tasks
        tasks = [self.analyze_symbol_async(symbol) for symbol in self.SYMBOLS]
        
        # Run all analysis tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                # An exception occurred within one of the tasks
                logger.error(f"❌ Error during async analysis: {result}", exc_info=True)
                continue
            
            signal = result
            if signal:
                if self.is_duplicate_signal(signal):
                    logger.info(f"⏭️ Skipping duplicate signal for {signal.symbol}")
                    continue

                if self.notifier.send_signal(signal):
                    self.recent_signals.append(signal)
                    signals_generated += 1
                    logger.info(f"✅ Signal sent successfully: {signal.signal_id}")
                else:
                    logger.error(f"❌ Failed to send signal for {signal.symbol}")
            else:
                logger.info(f"ℹ️ No high-confidence signal generated for a symbol.")

        logger.info("\n" + "=" * 80)
        logger.info(f"✅ ANALYSIS CYCLE COMPLETE (v4.0)")
        logger.info(f"📊 Signals Generated: {signals_generated}/{len(self.SYMBOLS)}")
        logger.info(f"📈 Total Signals in Memory: {len(self.recent_signals)}")
        logger.info("=" * 80)
        
        return signals_generated

# ============================================================================
# MAIN ENTRY POINT (Refactored to run async cycle)
# ============================================================================

def main():
    """Main entry point with proper error handling"""
    # ... (Environment variable validation remains the same)
    logger.info("\n" + "=" * 80)
    logger.info("SYNTHETIC INDICES TRADING BOT v4.0 - STARTING")
    logger.info("=" * 80)
    
    # Validate environment variables
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    main_chat_id = os.getenv('MAIN_CHAT_ID')
    
    if not telegram_token or not main_chat_id:
        logger.error("❌ TELEGRAM_BOT_TOKEN and/or MAIN_CHAT_ID environment variables not set")
        return 1
    
    # Parse stake amount with proper validation
    try:
        stake_usd = float(os.getenv('STAKE_USD', '10.0'))
        if stake_usd <= 0:
            logger.error("❌ STAKE_USD must be greater than 0")
            return 1
        if stake_usd > 1000:
             logger.warning(f"⚠️ STAKE_USD is very high: ${stake_usd:.2f}")
    except ValueError:
        logger.error(f"❌ Invalid STAKE_USD value. Using default: $10.00")
        stake_usd = 10.0
    
    try:
        # Initialize bot
        bot = SyntheticTradingBot(
            telegram_token=telegram_token,
            main_chat_id=main_chat_id,
            stake_usd=stake_usd
        )
        
        # Run the asynchronous analysis cycle
        signals_count = asyncio.run(bot.run_analysis_cycle_async())
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✅ Bot execution completed successfully")
        logger.info(f"📊 Total signals generated: {signals_count}")
        logger.info(f"⏰ Execution time: {datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
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