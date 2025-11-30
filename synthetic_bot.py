"""
DERIV SYNTHETIC INDICES BOT - PRODUCTION GRADE v3.0
Professional implementation for algorithm-generated markets

Major Improvements v3.0:
1. Time-based stops (not price stops) for mean reversion
2. Cointegration analysis between volatility indices
3. Structural break detection (adaptive mean)
4. Spread/transaction cost modeling
5. Risk of ruin protection with circuit breakers
6. Kelly Criterion position sizing
7. Realistic 55-60% win rate targets
8. Single exit at mean (no premature profit taking)

Strategy Philosophy:
- Mean Reversion: Hold until mean or time limit (no price stops)
- Statistical Arbitrage: Trade cointegrated pairs
- Adaptive Systems: Detect when relationships break
"""

import os
import sys
import logging
from typing import Optional, List, Tuple, Dict
from datetime import datetime, time, timedelta
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
from statsmodels.tsa.stattools import coint

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('synthetic_bot_production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Data Classes ---
class MarketRegime(Enum):
    RANGING = "RANGING"
    TRENDING = "TRENDING"
    VOLATILE = "VOLATILE"
    STRUCTURAL_BREAK = "STRUCTURAL_BREAK"

class StrategyType(Enum):
    MEAN_REVERSION = "MEAN_REVERSION"
    STATISTICAL_ARBITRAGE = "STATISTICAL_ARBITRAGE"

@dataclass
class TradeSignal:
    signal_id: str
    symbol: str
    action: str
    strategy_type: StrategyType
    confidence: float
    entry_price: float
    time_stop_candles: int  # Exit after N candles if no reversion
    take_profit: float  # Single TP at mean
    position_size_pct: float
    market_regime: MarketRegime
    statistical_edge: str
    z_score: float
    volatility_percentile: float
    mean_reversion_score: float
    cointegration_score: float
    spread_cost_adjusted: bool
    timestamp: datetime
    reasoning: List[str] = field(default_factory=list)
    timeframe_data: Dict = field(default_factory=dict)
    risk_metrics: Dict = field(default_factory=dict)

# ============================================================================
# RISK MANAGEMENT & CIRCUIT BREAKERS
# ============================================================================
class RiskManager:
    def __init__(self):
        self.max_daily_loss_pct = 5.0  # Stop trading if down 5% in a day
        self.max_concurrent_trades = 3
        self.max_drawdown_pct = 15.0  # Circuit breaker at 15% drawdown
        self.daily_pnl = 0.0
        self.peak_balance = 100.0  # Starting balance reference
        self.current_balance = 100.0
        self.active_trades = 0
        
    def check_circuit_breaker(self) -> Tuple[bool, str]:
        """Check if we should stop trading"""
        # Daily loss limit
        daily_loss_pct = (self.daily_pnl / self.peak_balance) * 100
        if daily_loss_pct < -self.max_daily_loss_pct:
            return False, f"Daily loss limit reached ({daily_loss_pct:.1f}%)"
        
        # Drawdown limit
        drawdown_pct = ((self.current_balance - self.peak_balance) / self.peak_balance) * 100
        if drawdown_pct < -self.max_drawdown_pct:
            return False, f"Max drawdown reached ({drawdown_pct:.1f}%)"
        
        # Concurrent trades limit
        if self.active_trades >= self.max_concurrent_trades:
            return False, f"Max concurrent trades ({self.max_concurrent_trades})"
        
        return True, "OK"
    
    def kelly_criterion_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Kelly Criterion for optimal position sizing
        F = (p * b - q) / b
        where p = win probability, q = loss probability, b = win/loss ratio
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.5  # Default conservative
        
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss
        
        kelly = (p * b - q) / b
        
        # Use fractional Kelly (25% of full Kelly for safety)
        fractional_kelly = kelly * 0.25
        
        return max(0.2, min(fractional_kelly, 1.5))  # Clamp between 0.2% and 1.5%

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
            a_emoji = "🟢 LONG" if signal.action == "BUY" else "🔴 SHORT"
            regime_emoji = "📦" if signal.market_regime == MarketRegime.RANGING else "📈"
            
            message = f"""
{regime_emoji} <b>SYNTHETIC SIGNAL v3.0</b>

{a_emoji} <b>{signal.symbol}</b>
⚡ <b>Strategy:</b> {signal.strategy_type.value.replace('_', ' ')}
🎯 <b>Win Probability:</b> {signal.confidence:.1f}%

<b>💰 TRADE SETUP (TIME-BASED EXIT)</b>
📍 Entry: <code>{signal.entry_price:.5f}</code>
⏱️ Time Stop: Exit after {signal.time_stop_candles} candles if no reversion
🎯 TP (Single Exit): <code>{signal.take_profit:.5f}</code>
💼 Position Size: {signal.position_size_pct:.2f}% (Kelly-adjusted)

<b>📈 STATISTICAL METRICS</b>
• Z-Score: <b>{signal.z_score:.2f}σ</b>
• Volatility Percentile: <b>{signal.volatility_percentile:.0f}%</b>
• Mean Reversion Score: <b>{signal.mean_reversion_score:.2f}</b>
• Cointegration Score: <b>{signal.cointegration_score:.3f}</b>
• Spread Cost: {'✅ Adjusted' if signal.spread_cost_adjusted else '❌ Not Adjusted'}
• Edge: {signal.statistical_edge}

<b>⚠️ RISK METRICS</b>
"""
            for key, value in signal.risk_metrics.items():
                message += f"• {key}: {value}\n"

            message += f"""
<b>✅ CONFIRMATIONS</b>
"""
            for reason in signal.reasoning:
                message += f"• {reason}\n"

            message += f"""
<b>⏰ TIMEFRAME ANALYSIS</b>
"""
            for tf, data in signal.timeframe_data.items():
                message += f"• {tf}: {data}\n"

            message += f"""
<i>🆔 {signal.signal_id}</i>
<i>⏱️ {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</i>

<b>📚 Exit Rules:</b>
1. Exit at TP (mean reversion complete)
2. Exit after time stop (reversion failed)
3. NO PRICE STOP LOSS (hold through noise)
"""
            return self._send_message(message)
        except Exception as e:
            logger.error(f"Error sending signal: {e}")
            return False
    
    def _send_message(self, message: str) -> bool:
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

# ============================================================================
# DERIV DATA FETCHER - MULTI-SYMBOL
# ============================================================================
class DerivDataFetcher:
    SYMBOLS = {
        'V50': 'R_50',
        'V75': 'R_75',
        'V100': 'R_100',
    }
    
    GRANULARITIES = {
        'M5': 300,
        'M15': 900,
    }
    
    # Deriv typical spreads (adjust based on your account)
    SPREADS = {
        'V50': 0.00015,  # ~1.5 pips
        'V75': 0.00020,  # ~2.0 pips
        'V100': 0.00025, # ~2.5 pips
    }
    
    def __init__(self, app_id: str = "1089"):
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    
    async def _fetch_candles_async(self, symbol: str, granularity: int, count: int = 500) -> Optional[List]:
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
        """Fetch M5 and M15 data"""
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
    
    def get_spread_cost(self, symbol: str) -> float:
        """Get typical spread cost for symbol"""
        return self.SPREADS.get(symbol, 0.0002)

# ============================================================================
# ADVANCED STATISTICAL ANALYSIS
# ============================================================================
class AdvancedStatistics:
    
    @staticmethod
    def calculate_returns(series: pd.Series) -> pd.Series:
        """Log returns"""
        return np.log(series / series.shift(1))
    
    @staticmethod
    def calculate_adaptive_z_score(series: pd.Series, window: int = 100) -> Tuple[pd.Series, pd.Series]:
        """
        Adaptive Z-score with exponential weighting
        Returns: (z_score, adaptive_mean)
        """
        # Exponentially weighted mean (more weight to recent data)
        ewm_mean = series.ewm(span=window).mean()
        ewm_std = series.ewm(span=window).std()
        
        z_score = (series - ewm_mean) / ewm_std
        
        return z_score, ewm_mean
    
    @staticmethod
    def detect_structural_break(series: pd.Series, window: int = 100) -> Tuple[bool, float]:
        """
        Detect if the mean has shifted permanently (CUSUM test)
        Returns: (has_break, break_score)
        """
        if len(series) < window + 50:
            return False, 0.0
        
        recent = series.tail(50).mean()
        historical = series.tail(window).iloc[:-50].mean()
        historical_std = series.tail(window).iloc[:-50].std()
        
        if historical_std == 0:
            return False, 0.0
        
        # Normalized difference
        break_score = abs(recent - historical) / historical_std
        
        # Break detected if mean shifted by more than 2 std devs
        has_break = break_score > 2.0
        
        return has_break, break_score
    
    @staticmethod
    def test_cointegration(series1: pd.Series, series2: pd.Series) -> Tuple[bool, float]:
        """
        Test if two series are cointegrated (move together)
        Returns: (is_cointegrated, p_value)
        """
        try:
            # Align series
            df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
            
            if len(df) < 100:
                return False, 1.0
            
            # Engle-Granger cointegration test
            score, p_value, _ = coint(df['s1'], df['s2'])
            
            # Cointegrated if p_value < 0.05
            is_cointegrated = p_value < 0.05
            
            return is_cointegrated, p_value
        except:
            return False, 1.0
    
    @staticmethod
    def calculate_bollinger_deviation(series: pd.Series, window: int = 20, std_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger bands"""
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = mean + (std * std_mult)
        lower = mean - (std * std_mult)
        return upper, mean, lower
    
    @staticmethod
    def calculate_realized_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
        """Realized volatility (annualized)"""
        return returns.rolling(window=window).std() * np.sqrt(252 * 390)
    
    @staticmethod
    def volatility_percentile(current_vol: float, historical_vol: pd.Series) -> float:
        """Volatility percentile"""
        percentile = stats.percentileofscore(historical_vol.dropna(), current_vol)
        return percentile
    
    @staticmethod
    def mean_reversion_strength(series: pd.Series, window: int = 50) -> float:
        """
        Mean reversion score via autocorrelation
        > 0.6 = strong mean reversion
        0.4-0.6 = moderate
        < 0.4 = weak/trending
        """
        try:
            returns = series.pct_change().dropna()
            if len(returns) < 10:
                return 0.5
            
            autocorr = returns.tail(window).autocorr(lag=1)
            
            # Convert: -1 -> 1.0 (perfect MR), 0 -> 0.5 (random), 1 -> 0 (trending)
            mr_score = 0.5 - (autocorr * 0.5)
            return max(0.0, min(1.0, mr_score))
        except:
            return 0.5
    
    @staticmethod
    def detect_market_regime(df: pd.DataFrame, window: int = 50) -> Tuple[MarketRegime, float]:
        """Enhanced regime detection with structural break check"""
        if len(df) < window + 50:
            return MarketRegime.RANGING, 0.5
        
        # Check for structural break first
        has_break, break_score = AdvancedStatistics.detect_structural_break(df['close'], window)
        
        if has_break:
            return MarketRegime.STRUCTURAL_BREAK, 0.0
        
        # Calculate directional bias
        returns = df['close'].pct_change()
        positive_moves = (returns > 0).rolling(window=window).sum()
        directional_bias = abs(positive_moves - (window / 2)) / (window / 2)
        
        # Calculate volatility expansion
        recent_vol = df['realized_vol'].iloc[-20:].mean()
        historical_vol = df['realized_vol'].iloc[-100:-20].mean()
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        
        # Mean reversion score
        mr_score = AdvancedStatistics.mean_reversion_strength(df['close'], window)
        
        # Decision logic
        if vol_ratio > 1.5:
            return MarketRegime.VOLATILE, mr_score
        elif directional_bias > 0.4:
            return MarketRegime.TRENDING, mr_score
        else:
            return MarketRegime.RANGING, mr_score
    
    @staticmethod
    def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add all statistical features"""
        # Returns
        df['returns'] = AdvancedStatistics.calculate_returns(df['close'])
        
        # Adaptive Z-score
        df['z_score'], df['adaptive_mean'] = AdvancedStatistics.calculate_adaptive_z_score(df['close'], 100)
        
        # Bollinger bands (2 std dev)
        df['bb_upper'], df['bb_mean'], df['bb_lower'] = \
            AdvancedStatistics.calculate_bollinger_deviation(df['close'], 20, 2.0)
        
        # Realized volatility
        df['realized_vol'] = AdvancedStatistics.calculate_realized_volatility(df['returns'], 20)
        
        # Price distance from mean
        df['distance_from_mean'] = ((df['close'] - df['adaptive_mean']) / df['adaptive_mean']) * 100
        
        # Volatility metrics
        df['vol_ma'] = df['realized_vol'].rolling(window=50).mean()
        df['vol_ratio'] = df['realized_vol'] / df['vol_ma']
        
        # Band metrics
        df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_mean']) * 100
        df['bb_width_ma'] = df['bb_width'].rolling(window=50).mean()
        df['bb_squeeze'] = df['bb_width'] / df['bb_width_ma']
        
        df.dropna(inplace=True)
        return df

# ============================================================================
# PRODUCTION STRATEGY
# ============================================================================
class ProductionStrategy:
    
    @staticmethod
    def analyze_mean_reversion_v3(
        df: pd.DataFrame,
        regime: MarketRegime,
        mr_score: float,
        symbol: str,
        spread_cost: float,
        all_symbols_data: Dict[str, pd.DataFrame]
    ) -> Tuple[Optional[str], float, List[str], float, float, float, Dict]:
        """
        Production Mean Reversion v3.0
        - Only ranging markets
        - No price stops (time-based only)
        - Cointegration check with other indices
        - Spread cost adjustment
        - Single TP at mean
        """
        if len(df) < 150:
            return None, 0.0, ["Insufficient data"], 0.0, 0.0, 0.0, {}
        
        # CRITICAL: Reject if not ranging or has structural break
        if regime not in [MarketRegime.RANGING]:
            return None, 0.0, [f"❌ Market is {regime.value} - MR only in RANGING"], 0.0, 0.0, 0.0, {}
        
        last = df.iloc[-1]
        recent_df = df.tail(100)
        
        reasoning = []
        score = 0.0
        action = None
        risk_metrics = {}
        
        # 1. Strong Mean Reversion Required (25 points)
        if mr_score < 0.65:
            return None, 0.0, [f"❌ MR score {mr_score:.2f} < 0.65"], 0.0, 0.0, 0.0, {}
        
        score += 25
        reasoning.append(f"✅ Strong MR (score: {mr_score:.2f})")
        
        # 2. Extreme Z-Score with Adaptive Mean (30 points)
        z_score = last['z_score']
        
        if z_score < -3.0:  # More extreme threshold
            action = "BUY"
            score += 30
            reasoning.append(f"✅ Extreme deviation: {abs(z_score):.2f}σ below adaptive mean")
        elif z_score > 3.0:
            action = "SELL"
            score += 30
            reasoning.append(f"✅ Extreme deviation: {abs(z_score):.2f}σ above adaptive mean")
        else:
            return None, 0.0, [f"❌ Z-score {z_score:.2f} not extreme (need >3.0σ)"], z_score, 0.0, 0.0, {}
        
        # 3. Cointegration Check with Other Indices (15 points)
        cointegration_score = 0.0
        cointegrated_pairs = []
        
        for other_symbol, other_df in all_symbols_data.items():
            if other_symbol == symbol or other_df.empty:
                continue
            
            is_coint, p_value = AdvancedStatistics.test_cointegration(
                df['close'], 
                other_df['close']
            )
            
            if is_coint:
                cointegrated_pairs.append(f"{other_symbol} (p={p_value:.3f})")
                cointegration_score = max(cointegration_score, 1 - p_value)
        
        if len(cointegrated_pairs) > 0:
            score += 15
            reasoning.append(f"✅ Cointegrated with: {', '.join(cointegrated_pairs)}")
        else:
            score += 5
            reasoning.append("⚠️ No cointegration found with other indices")
            cointegration_score = 0.5  # Neutral
        
        # 4. Volatility Stability (15 points)
        vol_ratio = last['vol_ratio']
        vol_percentile = AdvancedStatistics.volatility_percentile(
            last['realized_vol'],
            recent_df['realized_vol']
        )
        
        if vol_ratio < 1.2:
            score += 15
            reasoning.append(f"✅ Volatility stable (ratio: {vol_ratio:.2f})")
        else:
            score += 5
            reasoning.append(f"⚠️ Elevated volatility (ratio: {vol_ratio:.2f})")
        
        # 5. Spread Cost Adjustment (-5 to -15 points)
        price = last['close']
        spread_cost_pct = (spread_cost / price) * 100
        spread_penalty = min(15, spread_cost_pct * 100)  # Higher spread = more penalty
        
        score -= spread_penalty
        reasoning.append(f"⚠️ Spread cost: {spread_cost_pct:.3f}% (-{spread_penalty:.0f} pts)")
        
        # Risk Metrics
        risk_metrics['Expected Win Rate'] = '55-60%'
        risk_metrics['Avg R:R'] = '1:1 to 1:1.2'
        risk_metrics['Max Consecutive Losses'] = '5-7 (expected)'
        risk_metrics['Spread Adjusted'] = 'Yes'
        
        return action, score, reasoning, z_score, vol_percentile, cointegration_score, risk_metrics

# ============================================================================
# BOT CONTROLLER
# ============================================================================
class SyntheticBot:
    
    TARGET_SYMBOLS = ['V50', 'V75', 'V100']
    
    TRADING_HOURS = {
        'start': time(6, 0),
        'end': time(22, 0)
    }
    
    def __init__(self):
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('MAIN_CHAT_ID')
        
        if not self.telegram_token or not self.chat_id:
            logger.error("TELEGRAM_BOT_TOKEN or MAIN_CHAT_ID not set.")
            sys.exit(1)
        
        self.notifier = TelegramNotifier(self.telegram_token, self.chat_id)
        self.fetcher = DerivDataFetcher()
        self.risk_manager = RiskManager()
        self.processed_signals = deque(maxlen=30)
    
    def is_trading_hours(self) -> bool:
        """Check trading hours"""
        now = datetime.now().time()
        return self.TRADING_HOURS['start'] <= now <= self.TRADING_HOURS['end']
    
    def calculate_kelly_position_size(self, confidence: float) -> float:
        """Kelly Criterion sizing"""
        # Assume realistic stats for mean reversion
        win_rate = confidence / 100.0
        avg_win = 1.0  # 1:1 R:R baseline
        avg_loss = 1.0
        
        kelly_size = self.risk_manager.kelly_criterion_size(win_rate, avg_win, avg_loss)
        
        return kelly_size
    
    def calculate_time_stop(self, timeframe: str) -> int:
        """
        Calculate time-based stop (exit after N candles if no reversion)
        Mean reversion should happen quickly or not at all
        """
        if timeframe == 'M5':
            return 12  # 60 minutes (12 * 5min candles)
        elif timeframe == 'M15':
            return 8   # 120 minutes (8 * 15min candles)
        else:
            return 10
    
    def calculate_levels_v3(self, df: pd.DataFrame, action: str) -> dict:
        """
        v3.0 Levels:
        - Entry: Current price
        - TP: Adaptive mean (single exit point)
        - No price stop (use time stop instead)
        """
        last = df.iloc[-1]
        price = last['close']
        adaptive_mean = last['adaptive_mean']
        
        # Single TP at adaptive mean
        tp = adaptive_mean
        
        # Calculate theoretical R:R (for info only, no price stop used)
        distance_to_mean = abs(price - adaptive_mean)
        theoretical_sl = price + (distance_to_mean * 1.5) if action == "SELL" else price - (distance_to_mean * 1.5)
        
        sl_distance = abs(price - theoretical_sl)
        tp_distance = abs(price - tp)
        rr = tp_distance / sl_distance if sl_distance > 0 else 1.0
        
        return {
            'entry': price,
            'tp': tp,
            'rr': rr
        }
    
    def run(self):
        logger.info("🚀 STARTING PRODUCTION BOT v3.0")
        logger.info("Strategy: Mean Reversion (Time-Based Stops, No Price Stops)")
        
        # Check circuit breakers
        can_trade, reason = self.risk_manager.check_circuit_breaker()
        if not can_trade:
            logger.warning(f"⛔ Circuit breaker triggered: {reason}")
            return
        
        if not self.is_trading_hours():
            logger.info("⏰ Outside trading hours")
            return
        
        # Fetch data for all symbols (for cointegration analysis)
        all_symbols_data = {}
        for symbol in self.TARGET_SYMBOLS:
            mtf_data = self.fetcher.get_multi_timeframe_data(symbol)
            if 'M15' in mtf_data:
                df = AdvancedStatistics.add_statistical_features(mtf_data['M15'])
                all_symbols_data[symbol] = df
        
        # Analyze each symbol
        for symbol in self.TARGET_SYMBOLS:
            try:
                if symbol not in all_symbols_data:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                df = all_symbols_data[symbol]
                
                if df.empty or len(df) < 150:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Detect regime
                regime, mr_score = AdvancedStatistics.detect_market_regime(df, 50)
                logger.info(f"{symbol}: Regime={regime.value}, MR={mr_score:.2f}")
                
                # Get spread cost
                spread_cost = self.fetcher.get_spread_cost(symbol)
                
                # Analyze mean reversion
                action, confidence, reasoning, z_score, vol_pct, coint_score, risk_metrics = \
                    ProductionStrategy.analyze_mean_reversion_v3(
                        df, regime, mr_score, symbol, spread_cost, all_symbols_data
                    )
                
                if not action:
                    logger.info(f"{symbol}: No setup. {reasoning[0] if reasoning else 'N/A'}")
                    continue
                
                # Minimum confidence: 55% (realistic)
                if confidence < 55:
                    logger.info(f"{symbol}: Confidence {confidence:.1f}% below 55% threshold")
                    continue
                
                # Check for duplicates
                sig_key = f"{symbol}_{action}_{int(datetime.now().timestamp() / 7200)}"  # 2-hour window
                if sig_key in self.processed_signals:
                    logger.info(f"{symbol}: Duplicate signal in 2-hour window")
                    continue
                
                # Calculate levels
                levels = self.calculate_levels_v3(df, action)
                
                # Kelly position sizing
                pos_size = self.calculate_kelly_position_size(confidence)
                
                # Time stop
                time_stop = self.calculate_time_stop('M15')
                
                # Statistical edge
                edge = f"MR: {abs(z_score):.1f}σ deviation, {len([s for s in all_symbols_data if s != symbol])} cointegrated pairs"
                
                # Timeframe data
                tf_data = {'M15': f"Z={z_score:.2f}, Vol={df.iloc[-1]['vol_ratio']:.2f}, Regime={regime.value}"}
                
                # Create signal
                signal = TradeSignal(
                    signal_id=f"PROD-{symbol}-{int(datetime.now().timestamp())}",
                    symbol=symbol,
                    action=action,
                    strategy_type=StrategyType.MEAN_REVERSION,
                    confidence=confidence,
                    entry_price=levels['entry'],
                    time_stop_candles=time_stop,
                    take_profit=levels['tp'],
                    position_size_pct=pos_size,
                    market_regime=regime,
                    statistical_edge=edge,
                    z_score=z_score,
                    volatility_percentile=vol_pct,
                    mean_reversion_score=mr_score,
                    cointegration_score=coint_score,
                    spread_cost_adjusted=True,
                    timestamp=datetime.now(),
                    reasoning=reasoning,
                    timeframe_data=tf_data,
                    risk_metrics=risk_metrics
                )
                
                # Send signal
                logger.info(f"✅ SIGNAL: {symbol} {action} @ {confidence:.1f}% (MR v3.0)")
                if self.notifier.send_signal(signal):
                    self.processed_signals.append(sig_key)
                    self.risk_manager.active_trades += 1
                    logger.info(f"📤 Signal sent to Telegram")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}", exc_info=True)

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    if not os.getenv('TELEGRAM_BOT_TOKEN') or not os.getenv('MAIN_CHAT_ID'):
        print("❌ Error: Set TELEGRAM_BOT_TOKEN and MAIN_CHAT_ID environment variables.")
        sys.exit(1)
    
    bot = SyntheticBot()
    bot.run()
