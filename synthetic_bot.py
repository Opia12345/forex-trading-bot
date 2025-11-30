"""
DERIV SYNTHETIC INDICES BOT - HIGH FREQUENCY SCALPING v4.0
Optimized for DAILY trades on Volatility Indices (V10, V25, V50, V75, V100)

v4.0 Focus: MORE FREQUENT TRADES
1. M1 and M5 timeframes (faster signals)
2. Multiple strategies running simultaneously
3. Lower confidence thresholds (50%+)
4. Quick scalps (5-20 minute holds)
5. Micro mean reversions + volatility spikes
6. All volatility indices (V10, V25, V50, V75, V100)

Expected: 10-20 signals per day across all symbols
Win Rate Target: 55-65%
Hold Time: 5-30 minutes average
"""

import os
import sys
import logging
from typing import Optional, List, Tuple, Dict
from datetime import datetime, time
from collections import deque
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import requests
import json
import asyncio
import websockets
from scipy import stats

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('synthetic_bot_scalping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Data Classes ---
class StrategyType(Enum):
    MICRO_REVERSAL = "MICRO_REVERSAL"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    MOMENTUM_SCALP = "MOMENTUM_SCALP"
    BOLLINGER_BOUNCE = "BOLLINGER_BOUNCE"

@dataclass
class TradeSignal:
    signal_id: str
    symbol: str
    action: str
    strategy_type: StrategyType
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    expected_hold_minutes: int
    z_score: float
    volatility_state: str
    timestamp: datetime
    reasoning: List[str] = field(default_factory=list)

# ============================================================================
# RISK MANAGEMENT
# ============================================================================
class RiskManager:
    def __init__(self):
        self.max_daily_loss_pct = 5.0
        self.max_concurrent_trades = 5  # Increased for scalping
        self.max_risk_per_trade = 1.0  # 1% per trade
        self.daily_trades = 0
        self.max_daily_trades = 30  # Cap at 30 trades per day
        
    def can_trade(self) -> Tuple[bool, str]:
        """Check if we can take another trade"""
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit reached ({self.max_daily_trades})"
        return True, "OK"
    
    def calculate_position_size(self, confidence: float, symbol: str) -> float:
        """Dynamic position sizing"""
        base_size = self.max_risk_per_trade
        
        # Adjust for confidence
        if confidence >= 65:
            return base_size * 1.2  # 1.2%
        elif confidence >= 55:
            return base_size  # 1.0%
        else:
            return base_size * 0.7  # 0.7%

# ============================================================================
# TELEGRAM NOTIFIER
# ============================================================================
class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_signal(self, signal: TradeSignal) -> bool:
        try:
            a_emoji = "🟢" if signal.action == "BUY" else "🔴"
            strategy_emoji = {
                StrategyType.MICRO_REVERSAL: "🔄",
                StrategyType.VOLATILITY_SPIKE: "⚡",
                StrategyType.MOMENTUM_SCALP: "🚀",
                StrategyType.BOLLINGER_BOUNCE: "📊"
            }.get(signal.strategy_type, "💹")
            
            message = f"""
{strategy_emoji} <b>SCALP SIGNAL - {signal.strategy_type.value.replace('_', ' ')}</b>

{a_emoji} <b>{signal.symbol}</b> {a_emoji}
🎯 Confidence: <b>{signal.confidence:.0f}%</b>
⏱️ Expected Hold: <b>{signal.expected_hold_minutes} min</b>

<b>💰 LEVELS</b>
📍 Entry: <code>{signal.entry_price:.5f}</code>
🛑 Stop: <code>{signal.stop_loss:.5f}</code>
🎯 Target: <code>{signal.take_profit:.5f}</code>
💼 Risk: <b>{signal.position_size_pct:.1f}%</b>

<b>📊 STATS</b>
• Z-Score: {signal.z_score:.2f}
• Vol State: {signal.volatility_state}

<b>✅ SETUP</b>
"""
            for reason in signal.reasoning:
                message += f"• {reason}\n"
            
            message += f"""
<i>🆔 {signal.signal_id}</i>
<i>⏱️ {signal.timestamp.strftime('%H:%M:%S')}</i>
"""
            return self._send_message(message)
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False
    
    def _send_message(self, message: str) -> bool:
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except:
            return False

# ============================================================================
# DERIV DATA FETCHER - FAST TIMEFRAMES
# ============================================================================
class DerivDataFetcher:
    SYMBOLS = {
        'V10': 'R_10',
        'V25': 'R_25',
        'V50': 'R_50',
        'V75': 'R_75',
        'V100': 'R_100',
    }
    
    GRANULARITIES = {
        'M1': 60,
        'M5': 300,
    }
    
    SPREADS = {
        'V10': 0.00010,
        'V25': 0.00015,
        'V50': 0.00015,
        'V75': 0.00020,
        'V100': 0.00025,
    }
    
    def __init__(self, app_id: str = "1089"):
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    
    async def _fetch_candles_async(self, symbol: str, granularity: int, count: int = 300) -> Optional[List]:
        try:
            async with websockets.connect(self.ws_url, ping_interval=30, close_timeout=10) as ws:
                request = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": count,
                    "end": "latest",
                    "start": 1,
                    "style": "candles",
                    "granularity": granularity
                }
                await ws.send(json.dumps(request))
                response = await asyncio.wait_for(ws.recv(), timeout=20)
                data = json.loads(response)
                return data.get('candles')
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def get_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch M1 and M5 data"""
        deriv_symbol = self.SYMBOLS.get(symbol)
        if not deriv_symbol:
            return {}
        
        result = {}
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            for tf_name, granularity in self.GRANULARITIES.items():
                try:
                    candles = loop.run_until_complete(
                        self._fetch_candles_async(deriv_symbol, granularity)
                    )
                    
                    if candles:
                        df = pd.DataFrame(candles)
                        df = df.rename(columns={
                            'open': 'open', 'high': 'high',
                            'low': 'low', 'close': 'close',
                            'epoch': 'time'
                        })
                        cols = ['open', 'high', 'low', 'close']
                        df[cols] = df[cols].astype(float)
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        result[tf_name] = df
                except Exception as e:
                    logger.error(f"Error processing {tf_name}: {e}")
                    continue
        finally:
            loop.close()
        
        return result
    
    def get_spread(self, symbol: str) -> float:
        return self.SPREADS.get(symbol, 0.0002)

# ============================================================================
# SCALPING INDICATORS
# ============================================================================
class ScalpingIndicators:
    
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add fast scalping indicators"""
        # Price action
        df['returns'] = df['close'].pct_change()
        
        # Fast EMAs
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # Bollinger Bands (20-period, 2 std)
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Z-score (50-period for quick reversions)
        mean_50 = df['close'].rolling(window=50).mean()
        std_50 = df['close'].rolling(window=50).std()
        df['z_score'] = (df['close'] - mean_50) / std_50
        
        # ATR (14-period)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = pd.Series(true_range).rolling(window=14).mean()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['vol_ma'] = df['volatility'].rolling(window=50).mean()
        df['vol_spike'] = df['volatility'] > (df['vol_ma'] * 1.5)
        
        # RSI (14-period)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Price position in BB
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        df.dropna(inplace=True)
        return df

# ============================================================================
# SCALPING STRATEGIES
# ============================================================================
class ScalpingStrategies:
    
    @staticmethod
    def strategy_micro_reversal(df: pd.DataFrame) -> Tuple[Optional[str], float, List[str], float, str]:
        """
        Micro Mean Reversion (Quick 5-15min scalps)
        - Price touches BB extremes
        - RSI oversold/overbought
        - Quick reversion to middle BB
        """
        if len(df) < 50:
            return None, 0.0, [], 0.0, ""
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        reasoning = []
        score = 0.0
        action = None
        
        # 1. BB Touch (30 points)
        if last['close'] <= last['bb_lower']:
            action = "BUY"
            score += 30
            reasoning.append("✅ Price at lower BB")
        elif last['close'] >= last['bb_upper']:
            action = "SELL"
            score += 30
            reasoning.append("✅ Price at upper BB")
        else:
            return None, 0.0, ["❌ Price not at BB extremes"], 0.0, ""
        
        # 2. RSI Confirmation (20 points)
        if action == "BUY" and last['rsi'] < 35:
            score += 20
            reasoning.append(f"✅ RSI oversold ({last['rsi']:.0f})")
        elif action == "SELL" and last['rsi'] > 65:
            score += 20
            reasoning.append(f"✅ RSI overbought ({last['rsi']:.0f})")
        else:
            score += 5
            reasoning.append(f"⚠️ RSI neutral ({last['rsi']:.0f})")
        
        # 3. Z-Score (15 points)
        z_score = last['z_score']
        if (action == "BUY" and z_score < -1.5) or (action == "SELL" and z_score > 1.5):
            score += 15
            reasoning.append(f"✅ Z-score extreme ({z_score:.2f})")
        
        # 4. Not in volatility spike (10 points)
        if not last['vol_spike']:
            score += 10
            reasoning.append("✅ Stable volatility")
        
        vol_state = "SPIKE" if last['vol_spike'] else "STABLE"
        
        return action, score, reasoning, z_score, vol_state
    
    @staticmethod
    def strategy_volatility_spike(df: pd.DataFrame) -> Tuple[Optional[str], float, List[str], float, str]:
        """
        Volatility Spike Fade (Ride the spike then fade)
        - Detect volatility spike
        - Enter on pullback
        - Exit on reversion
        """
        if len(df) < 50:
            return None, 0.0, [], 0.0, ""
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        reasoning = []
        score = 0.0
        action = None
        
        # 1. Volatility spike detected (25 points)
        if not last['vol_spike']:
            return None, 0.0, ["❌ No volatility spike"], 0.0, ""
        
        score += 25
        reasoning.append("✅ Volatility spike detected")
        
        # 2. Price moved significantly (20 points)
        price_move = abs(last['returns']) * 100
        if price_move > 0.3:
            score += 20
            reasoning.append(f"✅ Strong move ({price_move:.2f}%)")
        
        # 3. Reversal candle (20 points)
        if last['close'] < last['open'] and prev['close'] > prev['open']:
            action = "SELL"
            score += 20
            reasoning.append("✅ Bearish reversal candle")
        elif last['close'] > last['open'] and prev['close'] < prev['open']:
            action = "BUY"
            score += 20
            reasoning.append("✅ Bullish reversal candle")
        else:
            return None, 0.0, ["❌ No reversal pattern"], 0.0, "SPIKE"
        
        # 4. EMA alignment (10 points)
        if action == "BUY" and last['ema_5'] > last['ema_10']:
            score += 10
            reasoning.append("✅ EMA bullish")
        elif action == "SELL" and last['ema_5'] < last['ema_10']:
            score += 10
            reasoning.append("✅ EMA bearish")
        
        z_score = last['z_score']
        
        return action, score, reasoning, z_score, "SPIKE"
    
    @staticmethod
    def strategy_momentum_scalp(df: pd.DataFrame) -> Tuple[Optional[str], float, List[str], float, str]:
        """
        Momentum Scalp (Ride short bursts)
        - EMA crossover
        - RSI trending
        - Quick in/out
        """
        if len(df) < 50:
            return None, 0.0, [], 0.0, ""
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        reasoning = []
        score = 0.0
        action = None
        
        # 1. EMA crossover (30 points)
        if prev['ema_5'] <= prev['ema_10'] and last['ema_5'] > last['ema_10']:
            action = "BUY"
            score += 30
            reasoning.append("✅ EMA bullish crossover")
        elif prev['ema_5'] >= prev['ema_10'] and last['ema_5'] < last['ema_10']:
            action = "SELL"
            score += 30
            reasoning.append("✅ EMA bearish crossover")
        else:
            return None, 0.0, ["❌ No EMA crossover"], 0.0, ""
        
        # 2. RSI momentum (20 points)
        if action == "BUY" and 45 < last['rsi'] < 65:
            score += 20
            reasoning.append(f"✅ RSI bullish ({last['rsi']:.0f})")
        elif action == "SELL" and 35 < last['rsi'] < 55:
            score += 20
            reasoning.append(f"✅ RSI bearish ({last['rsi']:.0f})")
        
        # 3. Price above/below EMA20 (15 points)
        if action == "BUY" and last['close'] > last['ema_20']:
            score += 15
            reasoning.append("✅ Price above EMA20")
        elif action == "SELL" and last['close'] < last['ema_20']:
            score += 15
            reasoning.append("✅ Price below EMA20")
        
        z_score = last['z_score']
        vol_state = "SPIKE" if last['vol_spike'] else "STABLE"
        
        return action, score, reasoning, z_score, vol_state
    
    @staticmethod
    def strategy_bollinger_bounce(df: pd.DataFrame) -> Tuple[Optional[str], float, List[str], float, str]:
        """
        Bollinger Bounce (Trade the channel)
        - Price bounces off BB
        - Stay in channel
        - Quick scalp to opposite side
        """
        if len(df) < 50:
            return None, 0.0, [], 0.0, ""
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        reasoning = []
        score = 0.0
        action = None
        
        # 1. Bounce off lower BB (25 points)
        if prev['close'] < prev['bb_lower'] and last['close'] >= last['bb_lower']:
            action = "BUY"
            score += 25
            reasoning.append("✅ Bounce off lower BB")
        elif prev['close'] > prev['bb_upper'] and last['close'] <= last['bb_upper']:
            action = "SELL"
            score += 25
            reasoning.append("✅ Bounce off upper BB")
        else:
            return None, 0.0, ["❌ No BB bounce"], 0.0, ""
        
        # 2. BB width not too wide (15 points)
        if last['bb_width'] < 0.03:  # Tight channel
            score += 15
            reasoning.append(f"✅ Tight BB channel ({last['bb_width']:.4f})")
        
        # 3. RSI confirmation (15 points)
        if action == "BUY" and last['rsi'] < 40:
            score += 15
            reasoning.append(f"✅ RSI low ({last['rsi']:.0f})")
        elif action == "SELL" and last['rsi'] > 60:
            score += 15
            reasoning.append(f"✅ RSI high ({last['rsi']:.0f})")
        
        # 4. Candle confirmation (10 points)
        if action == "BUY" and last['close'] > last['open']:
            score += 10
            reasoning.append("✅ Bullish candle")
        elif action == "SELL" and last['close'] < last['open']:
            score += 10
            reasoning.append("✅ Bearish candle")
        
        z_score = last['z_score']
        vol_state = "SPIKE" if last['vol_spike'] else "STABLE"
        
        return action, score, reasoning, z_score, vol_state

# ============================================================================
# BOT CONTROLLER
# ============================================================================
class SyntheticBot:
    
    # All volatility indices (excluding 1s versions)
    TARGET_SYMBOLS = ['V10', 'V25', 'V50', 'V75', 'V100']
    
    def __init__(self):
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('MAIN_CHAT_ID')
        
        if not self.telegram_token or not self.chat_id:
            logger.error("TELEGRAM_BOT_TOKEN or MAIN_CHAT_ID not set.")
            sys.exit(1)
        
        self.notifier = TelegramNotifier(self.telegram_token, self.chat_id)
        self.fetcher = DerivDataFetcher()
        self.risk_manager = RiskManager()
        self.processed_signals = deque(maxlen=100)
    
    def calculate_levels(self, df: pd.DataFrame, action: str, strategy: StrategyType) -> dict:
        """Calculate SL and TP based on strategy"""
        last = df.iloc[-1]
        price = last['close']
        atr = last['atr']
        
        # Tight stops for scalping
        if strategy == StrategyType.MICRO_REVERSAL:
            sl_mult = 1.5
            tp_mult = 1.0  # 1:0.66 R:R (higher win rate compensates)
            hold_min = 10
        elif strategy == StrategyType.VOLATILITY_SPIKE:
            sl_mult = 2.0
            tp_mult = 1.5
            hold_min = 15
        elif strategy == StrategyType.MOMENTUM_SCALP:
            sl_mult = 1.5
            tp_mult = 1.2
            hold_min = 12
        else:  # BOLLINGER_BOUNCE
            sl_mult = 1.0
            tp_mult = 1.5
            hold_min = 8
        
        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult
        
        if action == "BUY":
            sl = price - sl_distance
            tp = price + tp_distance
        else:
            sl = price + sl_distance
            tp = price - tp_distance
        
        return {
            'entry': price,
            'sl': sl,
            'tp': tp,
            'hold_min': hold_min
        }
    
    def run(self):
        logger.info("🚀 STARTING HIGH FREQUENCY SCALPING BOT v4.0")
        logger.info(f"Target: {len(self.TARGET_SYMBOLS)} symbols, 10-20 trades/day")
        
        # Check risk limits
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.warning(f"⛔ {reason}")
            return
        
        signals_found = 0
        
        for symbol in self.TARGET_SYMBOLS:
            try:
                # Fetch M1 and M5 data
                mtf_data = self.fetcher.get_multi_timeframe_data(symbol)
                
                if 'M5' not in mtf_data:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                # Primary analysis on M5
                df = ScalpingIndicators.add_indicators(mtf_data['M5'])
                
                if df.empty or len(df) < 50:
                    continue
                
                # Run all 4 strategies
                strategies = [
                    (ScalpingStrategies.strategy_micro_reversal, StrategyType.MICRO_REVERSAL),
                    (ScalpingStrategies.strategy_volatility_spike, StrategyType.VOLATILITY_SPIKE),
                    (ScalpingStrategies.strategy_momentum_scalp, StrategyType.MOMENTUM_SCALP),
                    (ScalpingStrategies.strategy_bollinger_bounce, StrategyType.BOLLINGER_BOUNCE),
                ]
                
                for strategy_func, strategy_type in strategies:
                    action, confidence, reasoning, z_score, vol_state = strategy_func(df)
                    
                    if not action:
                        continue
                    
                    # Lower threshold: 50%+ (more signals)
                    if confidence < 50:
                        continue
                    
                    # Check duplicates (10-minute window)
                    sig_key = f"{symbol}_{strategy_type.value}_{int(datetime.now().timestamp() / 600)}"
                    if sig_key in self.processed_signals:
                        continue
                    
                    # Calculate levels
                    levels = self.calculate_levels(df, action, strategy_type)
                    
                    # Position sizing
                    pos_size = self.risk_manager.calculate_position_size(confidence, symbol)
                    
                    # Create signal
                    signal = TradeSignal(
                        signal_id=f"SCALP-{symbol}-{strategy_type.value[:3]}-{int(datetime.now().timestamp())}",
                        symbol=symbol,
                        action=action,
                        strategy_type=strategy_type,
                        confidence=confidence,
                        entry_price=levels['entry'],
                        stop_loss=levels['sl'],
                        take_profit=levels['tp'],
                        position_size_pct=pos_size,
                        expected_hold_minutes=levels['hold_min'],
                        z_score=z_score,
                        volatility_state=vol_state,
                        timestamp=datetime.now(),
                        reasoning=reasoning
                    )
                    
                    # Send signal
                    logger.info(f"✅ {symbol} {action} {strategy_type.value} @ {confidence:.0f}%")
                    if self.notifier.send_signal(signal):
                        self.processed_signals.append(sig_key)
                        self.risk_manager.daily_trades += 1
                        signals_found += 1
                
            except Exception as e:
                logger.error(f"Error on {symbol}: {e}", exc_info=True)
        
        logger.info(f"📊 Scan complete: {signals_found} signals sent")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    if not os.getenv('TELEGRAM_BOT_TOKEN') or not os.getenv('MAIN_CHAT_ID'):
        print("❌ Error: Set TELEGRAM_BOT_TOKEN and MAIN_CHAT_ID environment variables.")
        sys.exit(1)
    
    bot = SyntheticBot()
    bot.run()