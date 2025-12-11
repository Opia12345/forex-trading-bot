"""
APEX - Autonomous Precision EXecution Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Boom/Crash algorithmic trading system with first-principles optimization.
Built for efficiency, scalability, and maximum return on computational resources.

Philosophy: Remove all friction. Maximize signal/noise ratio. Execute flawlessly.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import logging
import json
from typing import Optional, List, Tuple, Dict, NamedTuple
from datetime import datetime, time, timezone
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import requests
import asyncio
import websockets

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOGGING - Structured, minimal, actionable
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('apex.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION - First principles, no unnecessary constraints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TradingMode(Enum):
    CONSERVATIVE = "conservative"  # Safe, steady gains
    AGGRESSIVE = "aggressive"      # Maximum velocity
    AUTOPILOT = "autopilot"        # Adaptive intelligence

class Config:
    """Minimal configuration. Maximum flexibility."""
    
    # Assets
    SYMBOLS = {
        'CRASH': 'CRASH500',
        'BOOM': 'BOOM500',
    }
    
    # Operating mode - easily switchable
    MODE = TradingMode.AUTOPILOT
    
    # Risk parameters by mode
    RISK_PROFILES = {
        TradingMode.CONSERVATIVE: {
            'risk_per_trade': 0.3,
            'max_daily_trades': 6,
            'max_concurrent': 2,
            'min_confidence': 75,
            'max_daily_loss': 1.5,
        },
        TradingMode.AGGRESSIVE: {
            'risk_per_trade': 1.0,
            'max_daily_trades': 15,
            'max_concurrent': 5,
            'min_confidence': 60,
            'max_daily_loss': 4.0,
        },
        TradingMode.AUTOPILOT: {
            'risk_per_trade': 0.5,
            'max_daily_trades': 10,
            'max_concurrent': 3,
            'min_confidence': 68,
            'max_daily_loss': 2.5,
        }
    }
    
    # Technical parameters - optimized through iteration
    TREND_FAST = 20
    TREND_MID = 50
    TREND_SLOW = 100
    TREND_STRENGTH_MIN = 0.3
    
    RSI_OVERSOLD = 35
    RSI_OVERBOUGHT = 65
    RSI_EXTREME_LOW = 25
    RSI_EXTREME_HIGH = 75
    
    # Execution
    TARGET_PIPS_TREND = 35
    STOP_PIPS_TREND = 22
    TARGET_PIPS_SPIKE = 18
    STOP_PIPS_SPIKE = 28
    
    # Data
    LOOKBACK_CANDLES = 500
    SR_LOOKBACK = 120
    
    # Communications
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT = os.getenv('MAIN_CHAT_ID', '')
    
    @classmethod
    def get_profile(cls):
        """Get current risk profile"""
        return cls.RISK_PROFILES[cls.MODE]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA STRUCTURES - Efficient, type-safe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MarketRegime(Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    RANGING = "ranging"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

class TradeType(Enum):
    TREND_MOMENTUM = "trend_momentum"
    TREND_PULLBACK = "trend_pullback"
    SPIKE_REVERSAL = "spike_reversal"
    BREAKOUT = "breakout"

@dataclass
class Level:
    price: float
    strength: float
    touches: int
    level_type: str  # 'support' or 'resistance'

@dataclass
class MarketContext:
    regime: MarketRegime
    regime_score: float
    trend_strength: float
    volatility: float
    price: float
    rsi: float
    support_levels: List[Level]
    resistance_levels: List[Level]

@dataclass
class TradeSignal:
    id: str
    symbol: str
    direction: str  # BUY/SELL
    entry: float
    stop_loss: float
    take_profit: float
    confidence: float
    trade_type: TradeType
    context: MarketContext
    reasoning: List[str]
    timestamp: datetime
    
    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry - self.stop_loss)
        reward = abs(self.take_profit - self.entry)
        return reward / risk if risk > 0 else 0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA ACQUISITION - Fast, reliable, minimal latency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DataEngine:
    """Optimized market data acquisition"""
    
    SYMBOL_MAP = {
        'BOOM': 'BOOM500',
        'CRASH': 'CRASH500',
    }
    
    def __init__(self, app_id: str = "1089"):
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        self._cache = {}
    
    async def _fetch(self, symbol: str, count: int = 500) -> Optional[List]:
        """Async data fetch with timeout protection"""
        try:
            async with websockets.connect(self.ws_url, ping_interval=30) as ws:
                req = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": count,
                    "end": "latest",
                    "start": 1,
                    "style": "candles",
                    "granularity": 60
                }
                await ws.send(json.dumps(req))
                resp = await asyncio.wait_for(ws.recv(), timeout=15)
                data = json.loads(resp)
                
                if 'candles' in data:
                    return data['candles']
                elif 'error' in data:
                    log.error(f"API error: {data['error'].get('message')}")
                
        except asyncio.TimeoutError:
            log.error(f"Timeout fetching {symbol}")
        except Exception as e:
            log.error(f"Fetch error: {e}")
        
        return None
    
    def get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get processed market data"""
        deriv_symbol = self.SYMBOL_MAP.get(symbol)
        if not deriv_symbol:
            return None
        
        # Event loop handling
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Fetch and process
        candles = loop.run_until_complete(self._fetch(deriv_symbol, Config.LOOKBACK_CANDLES))
        
        if not candles:
            return None
        
        df = pd.DataFrame(candles)
        df.rename(columns={'epoch': 'time'}, inplace=True)
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        return df

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TECHNICAL ANALYSIS - Efficient computation, clear signal extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TechnicalEngine:
    """Vectorized technical indicators - maximum efficiency"""
    
    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Single-pass indicator computation"""
        # EMAs - trend identification
        df['ema_20'] = df['close'].ewm(span=Config.TREND_FAST, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=Config.TREND_MID, adjust=False).mean()
        df['ema_100'] = df['close'].ewm(span=Config.TREND_SLOW, adjust=False).mean()
        
        # ATR - volatility measurement
        hl = df['high'] - df['low']
        hc = np.abs(df['high'] - df['close'].shift())
        lc = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # RSI - momentum
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD - momentum confirmation
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Trend strength (ADX computation)
        df['adx'] = TechnicalEngine._compute_adx(df)
        
        # Volatility ratio
        df['vol_ratio'] = df['atr'] / df['close']
        
        df.dropna(inplace=True)
        return df
    
    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute ADX efficiently"""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        hl = df['high'] - df['low']
        hc = np.abs(df['high'] - df['close'].shift())
        lc = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr)
        
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(period).mean() / 100
        
        return adx

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MARKET ANALYSIS - Intelligence layer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MarketIntelligence:
    """High-level market understanding"""
    
    @staticmethod
    def identify_regime(df: pd.DataFrame) -> Tuple[MarketRegime, float, List[str]]:
        """Determine current market regime with confidence"""
        current = df.iloc[-1]
        score = 0
        evidence = []
        
        # EMA alignment
        if current['ema_20'] > current['ema_50'] > current['ema_100']:
            score += 40
            evidence.append("EMA stack: bullish")
        elif current['ema_20'] < current['ema_50'] < current['ema_100']:
            score -= 40
            evidence.append("EMA stack: bearish")
        
        # Price vs trend
        if current['close'] > current['ema_20']:
            score += 20
        else:
            score -= 20
        
        # Trend strength
        if current['adx'] > Config.TREND_STRENGTH_MIN:
            multiplier = 1.5 if score > 0 else -1.5
            score *= multiplier
            evidence.append(f"Strong trend: ADX {current['adx']:.2f}")
        
        # MACD momentum
        if current['macd'] > current['macd_signal']:
            score += 15
            evidence.append("MACD: bullish")
        else:
            score -= 15
            evidence.append("MACD: bearish")
        
        # Classify regime
        if score >= 70:
            return MarketRegime.STRONG_UPTREND, score, evidence
        elif score >= 30:
            return MarketRegime.UPTREND, score, evidence
        elif score <= -70:
            return MarketRegime.STRONG_DOWNTREND, score, evidence
        elif score <= -30:
            return MarketRegime.DOWNTREND, score, evidence
        else:
            return MarketRegime.RANGING, score, evidence
    
    @staticmethod
    def find_key_levels(df: pd.DataFrame) -> Tuple[List[Level], List[Level]]:
        """Identify high-probability support/resistance"""
        recent = df.tail(Config.SR_LOOKBACK)
        
        swing_highs = []
        swing_lows = []
        
        # Identify swing points
        for i in range(2, len(recent) - 2):
            # Swing high
            if (recent.iloc[i]['high'] > recent.iloc[i-1]['high'] and
                recent.iloc[i]['high'] > recent.iloc[i-2]['high'] and
                recent.iloc[i]['high'] > recent.iloc[i+1]['high'] and
                recent.iloc[i]['high'] > recent.iloc[i+2]['high']):
                swing_highs.append(recent.iloc[i]['high'])
            
            # Swing low
            if (recent.iloc[i]['low'] < recent.iloc[i-1]['low'] and
                recent.iloc[i]['low'] < recent.iloc[i-2]['low'] and
                recent.iloc[i]['low'] < recent.iloc[i+1]['low'] and
                recent.iloc[i]['low'] < recent.iloc[i+2]['low']):
                swing_lows.append(recent.iloc[i]['low'])
        
        # Cluster and score
        resistance = MarketIntelligence._cluster_levels(swing_highs, 'resistance')
        support = MarketIntelligence._cluster_levels(swing_lows, 'support')
        
        return support, resistance
    
    @staticmethod
    def _cluster_levels(prices: List[float], level_type: str, threshold: float = 0.004) -> List[Level]:
        """Cluster nearby levels and calculate strength"""
        if not prices:
            return []
        
        prices = sorted(prices)
        clusters = []
        current = [prices[0]]
        
        for p in prices[1:]:
            if abs(p - current[-1]) / current[-1] < threshold:
                current.append(p)
            else:
                avg = np.mean(current)
                touches = len(current)
                strength = min(100, touches * 20)
                clusters.append(Level(avg, strength, touches, level_type))
                current = [p]
        
        # Final cluster
        avg = np.mean(current)
        touches = len(current)
        strength = min(100, touches * 20)
        clusters.append(Level(avg, strength, touches, level_type))
        
        return sorted(clusters, key=lambda x: x.strength, reverse=True)[:5]
    
    @staticmethod
    def build_context(df: pd.DataFrame) -> MarketContext:
        """Build complete market context"""
        current = df.iloc[-1]
        regime, regime_score, _ = MarketIntelligence.identify_regime(df)
        support, resistance = MarketIntelligence.find_key_levels(df)
        
        return MarketContext(
            regime=regime,
            regime_score=regime_score,
            trend_strength=current['adx'],
            volatility=current['vol_ratio'],
            price=current['close'],
            rsi=current['rsi'],
            support_levels=support,
            resistance_levels=resistance
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRATEGY ENGINE - Decision making core
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StrategyEngine:
    """Core trading logic - first principles decision making"""
    
    @staticmethod
    def analyze_crash(df: pd.DataFrame, ctx: MarketContext) -> Optional[TradeSignal]:
        """
        CRASH Analysis:
        - Predominantly sells (crashes are downward spikes)
        - Buy only in strong uptrends at major support
        """
        profile = Config.get_profile()
        current = df.iloc[-1]
        
        # SELL OPPORTUNITIES
        
        # 1. Trend momentum sell
        if ctx.regime in [MarketRegime.STRONG_DOWNTREND, MarketRegime.DOWNTREND]:
            score = abs(ctx.regime_score)
            reasons = [f"Regime: {ctx.regime.value}", f"Trend strength: {ctx.trend_strength:.2f}"]
            
            # Pullback to EMA
            if abs(current['close'] - current['ema_20']) / current['close'] < 0.003:
                score += 25
                reasons.append("Pullback to EMA20")
                
                if current['rsi'] > 50:
                    score += 15
                    reasons.append(f"RSI pullback: {current['rsi']:.0f}")
                
                if score >= profile['min_confidence']:
                    return StrategyEngine._build_signal(
                        df, 'CRASH', 'SELL', score, TradeType.TREND_PULLBACK, 
                        ctx, reasons
                    )
            
            # Resistance rejection
            for lvl in ctx.resistance_levels[:2]:
                if abs(current['close'] - lvl.price) / current['close'] < 0.002:
                    score += lvl.strength * 0.4
                    reasons.append(f"At resistance: {lvl.price:.2f} ({lvl.touches}x)")
                    
                    if score >= profile['min_confidence']:
                        return StrategyEngine._build_signal(
                            df, 'CRASH', 'SELL', score, TradeType.TREND_MOMENTUM,
                            ctx, reasons
                        )
        
        # 2. Spike reversal sell
        for lvl in ctx.resistance_levels:
            if abs(current['close'] - lvl.price) / current['close'] < 0.002:
                score = lvl.strength
                reasons = [f"Spike at resistance: {lvl.price:.2f}"]
                
                if current['rsi'] > Config.RSI_OVERBOUGHT:
                    score += 30
                    reasons.append(f"Overbought: RSI {current['rsi']:.0f}")
                
                if current['close'] > current['ema_20']:
                    score += 15
                    reasons.append("Extended above trend")
                
                if score >= profile['min_confidence'] + 5:  # Higher bar for counter-trend
                    return StrategyEngine._build_signal(
                        df, 'CRASH', 'SELL', score, TradeType.SPIKE_REVERSAL,
                        ctx, reasons
                    )
        
        # BUY OPPORTUNITIES (rare, high conviction only)
        
        if ctx.regime in [MarketRegime.STRONG_UPTREND]:
            for lvl in ctx.support_levels[:1]:  # Only strongest support
                if (lvl.strength >= 80 and
                    abs(current['close'] - lvl.price) / current['close'] < 0.002):
                    score = ctx.regime_score + lvl.strength * 0.3
                    reasons = [
                        f"Strong uptrend + major support",
                        f"Support: {lvl.price:.2f} ({lvl.touches}x)"
                    ]
                    
                    if current['rsi'] < Config.RSI_OVERSOLD:
                        score += 25
                        reasons.append(f"Oversold: RSI {current['rsi']:.0f}")
                    
                    if score >= profile['min_confidence'] + 10:
                        return StrategyEngine._build_signal(
                            df, 'CRASH', 'BUY', score, TradeType.TREND_PULLBACK,
                            ctx, reasons
                        )
        
        return None
    
    @staticmethod
    def analyze_boom(df: pd.DataFrame, ctx: MarketContext) -> Optional[TradeSignal]:
        """
        BOOM Analysis:
        - Predominantly buys (booms are upward spikes)
        - Sell only in strong downtrends at major resistance
        """
        profile = Config.get_profile()
        current = df.iloc[-1]
        
        # BUY OPPORTUNITIES
        
        # 1. Trend momentum buy
        if ctx.regime in [MarketRegime.STRONG_UPTREND, MarketRegime.UPTREND]:
            score = ctx.regime_score
            reasons = [f"Regime: {ctx.regime.value}", f"Trend strength: {ctx.trend_strength:.2f}"]
            
            # Pullback to EMA
            if abs(current['close'] - current['ema_20']) / current['close'] < 0.003:
                score += 25
                reasons.append("Pullback to EMA20")
                
                if current['rsi'] < 50:
                    score += 15
                    reasons.append(f"RSI pullback: {current['rsi']:.0f}")
                
                if score >= profile['min_confidence']:
                    return StrategyEngine._build_signal(
                        df, 'BOOM', 'BUY', score, TradeType.TREND_PULLBACK,
                        ctx, reasons
                    )
            
            # Support bounce
            for lvl in ctx.support_levels[:2]:
                if abs(current['close'] - lvl.price) / current['close'] < 0.002:
                    score += lvl.strength * 0.4
                    reasons.append(f"At support: {lvl.price:.2f} ({lvl.touches}x)")
                    
                    if score >= profile['min_confidence']:
                        return StrategyEngine._build_signal(
                            df, 'BOOM', 'BUY', score, TradeType.TREND_MOMENTUM,
                            ctx, reasons
                        )
        
        # 2. Spike reversal buy
        for lvl in ctx.support_levels:
            if abs(current['close'] - lvl.price) / current['close'] < 0.002:
                score = lvl.strength
                reasons = [f"Spike at support: {lvl.price:.2f}"]
                
                if current['rsi'] < Config.RSI_OVERSOLD:
                    score += 30
                    reasons.append(f"Oversold: RSI {current['rsi']:.0f}")
                
                if current['close'] < current['ema_20']:
                    score += 15
                    reasons.append("Extended below trend")
                
                if score >= profile['min_confidence'] + 5:
                    return StrategyEngine._build_signal(
                        df, 'BOOM', 'BUY', score, TradeType.SPIKE_REVERSAL,
                        ctx, reasons
                    )
        
        # SELL OPPORTUNITIES (rare, high conviction only)
        
        if ctx.regime in [MarketRegime.STRONG_DOWNTREND]:
            for lvl in ctx.resistance_levels[:1]:
                if (lvl.strength >= 80 and
                    abs(current['close'] - lvl.price) / current['close'] < 0.002):
                    score = abs(ctx.regime_score) + lvl.strength * 0.3
                    reasons = [
                        f"Strong downtrend + major resistance",
                        f"Resistance: {lvl.price:.2f} ({lvl.touches}x)"
                    ]
                    
                    if current['rsi'] > Config.RSI_OVERBOUGHT:
                        score += 25
                        reasons.append(f"Overbought: RSI {current['rsi']:.0f}")
                    
                    if score >= profile['min_confidence'] + 10:
                        return StrategyEngine._build_signal(
                            df, 'BOOM', 'SELL', score, TradeType.TREND_PULLBACK,
                            ctx, reasons
                        )
        
        return None
    
    @staticmethod
    def _build_signal(df: pd.DataFrame, symbol: str, direction: str, 
                     confidence: float, trade_type: TradeType,
                     ctx: MarketContext, reasons: List[str]) -> TradeSignal:
        """Construct complete trade signal"""
        current = df.iloc[-1]
        entry = current['close']
        
        # Dynamic SL/TP based on trade type
        pip = 0.01
        
        if trade_type in [TradeType.TREND_MOMENTUM, TradeType.TREND_PULLBACK]:
            target_pips = Config.TARGET_PIPS_TREND
            stop_pips = Config.STOP_PIPS_TREND
        else:
            target_pips = Config.TARGET_PIPS_SPIKE
            stop_pips = Config.STOP_PIPS_SPIKE
        
        # Calculate levels
        if direction == 'BUY':
            sl = entry - (stop_pips * pip)
            tp = entry + (target_pips * pip)
        else:
            sl = entry + (stop_pips * pip)
            tp = entry - (target_pips * pip)
        
        return TradeSignal(
            id=f"{symbol}-{int(datetime.now().timestamp())}",
            symbol=symbol,
            direction=direction,
            entry=entry,
            stop_loss=sl,
            take_profit=tp,
            confidence=confidence,
            trade_type=trade_type,
            context=ctx,
            reasoning=reasons,
            timestamp=datetime.now(timezone.utc)
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RISK MANAGEMENT - Protect capital at all costs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RiskController:
    """Capital preservation system"""
    
    def __init__(self):
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.active_trades = 0
        self.consecutive_losses = 0
        self.last_reset = datetime.now(timezone.utc).date()
        self.performance_history = deque(maxlen=100)
    
    def reset_daily(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset:
            log.info(f"Daily reset | Trades: {self.daily_trades} | P&L: {self.daily_pnl:+.2f}%")
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset = today
    
    def can_trade(self) -> Tuple[bool, str]:
        self.reset_daily()
        profile = Config.get_profile()
        
        if self.daily_trades >= profile['max_daily_trades']:
            return False, f"Daily limit reached ({profile['max_daily_trades']})"
        
        if abs(self.daily_pnl) >= profile['max_daily_loss']:
            return False, f"Max daily loss hit ({profile['max_daily_loss']}%)"
        
        if self.active_trades >= profile['max_concurrent']:
            return False, f"Max concurrent ({profile['max_concurrent']})"
        
        if self.consecutive_losses >= 4:
            return False, "Circuit breaker: 4 consecutive losses"
        
        return True, "CLEAR"
    
    def register_trade(self, signal: TradeSignal):
        """Record new trade"""
        self.daily_trades += 1
        self.active_trades += 1
    
    def close_trade(self, profit_pct: float):
        """Record trade closure"""
        self.active_trades -= 1
        self.daily_pnl += profit_pct
        self.performance_history.append(profit_pct)
        
        if profit_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
    
    def get_win_rate(self) -> float:
        """Calculate recent win rate"""
        if not self.performance_history:
            return 0.0
        wins = sum(1 for p in self.performance_history if p > 0)
        return (wins / len(self.performance_history)) * 100

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMMUNICATIONS - Clean, actionable alerts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CommunicationHub:
    """Telegram notifications - zero fluff"""
    
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT
        self.enabled = bool(self.token and self.chat_id)
        self.base_url = f"https://api.telegram.org/bot{self.token}"
    
    def send_signal(self, signal: TradeSignal, win_rate: float) -> bool:
        if not self.enabled:
            return False
        
        try:
            # Clean, data-dense format
            icon = "🟢" if signal.direction == "BUY" else "🔴"
            
            msg = f"""
{icon} <b>{signal.symbol} {signal.direction}</b>

Confidence: {signal.confidence:.0f}%
Type: {signal.trade_type.value.replace('_', ' ').title()}
R:R = 1:{signal.risk_reward:.1f}

<b>EXECUTION</b>
Entry: <code>{signal.entry:.2f}</code>
Stop:  <code>{signal.stop_loss:.2f}</code>
Target: <code>{signal.take_profit:.2f}</code>

<b>CONTEXT</b>
Regime: {signal.context.regime.value}
Trend Strength: {signal.context.trend_strength:.2f}
RSI: {signal.context.rsi:.0f}

<b>LOGIC</b>
{chr(10).join('• ' + r for r in signal.reasoning)}

<b>STATS</b>
Win Rate: {win_rate:.0f}%
Mode: {Config.MODE.value.upper()}

<i>APEX | {signal.timestamp.strftime('%H:%M:%S')}</i>
"""
            
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": self.chat_id, "text": msg, "parse_mode": "HTML"},
                timeout=10
            )
            return resp.status_code == 200
            
        except Exception as e:
            log.error(f"Comms error: {e}")
            return False
    
    def send_status(self, msg: str) -> bool:
        """Send status update"""
        if not self.enabled:
            return False
        
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": self.chat_id, "text": msg, "parse_mode": "HTML"},
                timeout=10
            )
            return resp.status_code == 200
        except:
            return False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ORCHESTRATION - Main control system
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class APEX:
    """Autonomous Precision EXecution Engine"""
    
    def __init__(self):
        self.data_engine = DataEngine()
        self.risk = RiskController()
        self.comms = CommunicationHub()
        self.signal_history = deque(maxlen=50)
        
        log.info("━" * 70)
        log.info("APEX initialized")
        log.info(f"Mode: {Config.MODE.value.upper()}")
        log.info(f"Profile: {Config.get_profile()}")
        log.info("━" * 70)
    
    def scan_markets(self):
        """Execute market scan"""
        log.info("Scanning markets...")
        
        # Risk check
        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            log.warning(f"Trading halted: {reason}")
            return
        
        signals_generated = 0
        
        for symbol_key, symbol_code in Config.SYMBOLS.items():
            log.info(f"\nAnalyzing {symbol_key}...")
            
            try:
                # Get data
                df = self.data_engine.get_market_data(symbol_key)
                if df is None or len(df) < 200:
                    log.warning(f"Insufficient data for {symbol_key}")
                    continue
                
                # Compute indicators
                df = TechnicalEngine.compute_indicators(df)
                if df.empty:
                    continue
                
                # Build context
                ctx = MarketIntelligence.build_context(df)
                
                # Log context
                current = df.iloc[-1]
                log.info(f"Price: {current['close']:.2f} | Regime: {ctx.regime.value} | RSI: {ctx.rsi:.0f}")
                log.info(f"Trend: {ctx.trend_strength:.2f} | Volatility: {ctx.volatility:.4f}")
                
                # Generate signal
                if symbol_key == 'CRASH':
                    signal = StrategyEngine.analyze_crash(df, ctx)
                else:
                    signal = StrategyEngine.analyze_boom(df, ctx)
                
                if signal is None:
                    log.info("No setup detected")
                    continue
                
                # Check for duplicate
                sig_id = f"{symbol_key}_{signal.direction}_{int(datetime.now().timestamp() / 1800)}"
                if sig_id in self.signal_history:
                    log.info("Duplicate signal (30min cooldown)")
                    continue
                
                # Execute
                log.info(f"SIGNAL: {signal.direction} @ {signal.confidence:.0f}% | Type: {signal.trade_type.value}")
                log.info(f"Entry: {signal.entry:.2f} | SL: {signal.stop_loss:.2f} | TP: {signal.take_profit:.2f}")
                
                # Send notification
                win_rate = self.risk.get_win_rate()
                if self.comms.send_signal(signal, win_rate):
                    self.signal_history.append(sig_id)
                    self.risk.register_trade(signal)
                    signals_generated += 1
                    log.info("✓ Signal transmitted")
                else:
                    log.warning("⚠ Transmission failed")
                
            except Exception as e:
                log.error(f"Error analyzing {symbol_key}: {e}", exc_info=True)
        
        log.info(f"\nScan complete | Signals: {signals_generated}")
        log.info("━" * 70)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   █████╗ ██████╗ ███████╗██╗  ██╗                               ║
║  ██╔══██╗██╔══██╗██╔════╝╚██╗██╔╝                               ║
║  ███████║██████╔╝█████╗   ╚███╔╝                                ║
║  ██╔══██║██╔═══╝ ██╔══╝   ██╔██╗                                ║
║  ██║  ██║██║     ███████╗██╔╝ ██╗                               ║
║  ╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝                               ║
║                                                                   ║
║  Autonomous Precision EXecution Engine                           ║
║  First-principles algorithmic trading for Boom/Crash             ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODE: {mode}

PROFILE:
  • Risk per trade: {risk}%
  • Max daily trades: {max_trades}
  • Min confidence: {min_conf}%
  • Max daily loss: {max_loss}%

STRATEGY:
  • Trend following with pullback entries
  • High-probability support/resistance reversals
  • Dynamic stop-loss and take-profit
  • Regime-aware position sizing

EXECUTION:
  • Zero latency data acquisition
  • Vectorized technical computation
  • Real-time market intelligence
  • Instant signal transmission

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHILOSOPHY:

"First principles: Reason from the ground up. Question assumptions.
Optimize for reality, not tradition. Edge = Information × Speed × Execution."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEPLOYMENT:
  1. Verify on DEMO for 100+ trades
  2. Track: Win rate, avg R:R, max drawdown
  3. Optimize mode based on performance data
  4. Scale position sizing only after consistent profitability

RUN FREQUENCY: Every 5-15 minutes during active sessions
RECOMMENDED: Set up cron job or systemd timer for automation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Initializing...
""".format(
        mode=Config.MODE.value.upper(),
        risk=Config.get_profile()['risk_per_trade'],
        max_trades=Config.get_profile()['max_daily_trades'],
        min_conf=Config.get_profile()['min_confidence'],
        max_loss=Config.get_profile()['max_daily_loss']
    ))
    
    if not Config.TELEGRAM_TOKEN or not Config.TELEGRAM_CHAT:
        print("⚠️  WARNING: Telegram credentials not configured")
        print("Set environment variables: TELEGRAM_BOT_TOKEN, MAIN_CHAT_ID\n")
    
    engine = APEX()
    engine.scan_markets()

if __name__ == "__main__":
    main()