"""
PROFESSIONAL TRADING BOT v13.0 - PRODUCTION READY
Simplified indicators, slippage modeling, performance tracking, position management
"""

import os
import sys
import logging
from typing import Optional, List, Tuple, Dict
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
from dataclasses import dataclass
import pandas as pd
import numpy as np
import pytz

sys.path.append('scripts')

from trade_journal import TradeJournal, TradeRecord
from position_manager import PositionManager, PositionState

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Signal type classification"""
    SCALP = "SCALP"
    DAY_TRADE = "DAY_TRADE"
    SWING = "SWING"


class SignalQuality(Enum):
    """Signal quality rating"""
    EXCELLENT = "EXCELLENT"
    STRONG = "STRONG"
    GOOD = "GOOD"


class TrendDirection(Enum):
    """Trend direction classification"""
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


@dataclass
class TradeSignal:
    """Complete trade signal with all details"""
    signal_id: str
    symbol: str
    action: str
    signal_type: SignalType
    confidence: float
    quality: SignalQuality
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    breakeven_price: float
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]
    key_level: Optional[float]
    stop_loss_pips: float
    tp1_pips: float
    tp2_pips: float
    tp3_pips: float
    position_size: float
    risk_amount_usd: float
    risk_reward_ratio: float
    timeframe: str
    htf_timeframe: str
    htf_trend: str
    market_phase: str
    trading_session: str
    strategy_components: List[str]
    technical_summary: Dict
    timestamp: datetime
    valid_until: datetime


class TelegramNotifier:
    """Placeholder - import from your original bot"""
    def __init__(self, token, main_chat, simple_chat):
        self.token = token
        self.main_chat = main_chat
        self.simple_chat = simple_chat
    
    def send_signal(self, signal):
        logger.info(f"Sending signal: {signal.signal_id}")
        return True


class DerivDataFetcher:
    """Placeholder - import from your original bot"""
    def __init__(self, app_id):
        self.app_id = app_id
    
    def get_historical_data(self, symbol, timeframe, count):
        logger.info(f"Fetching {symbol} {timeframe} data")
        return None


class NewsMonitor:
    """Placeholder - import from your original bot"""
    def __init__(self, buffer_minutes=60):
        self.buffer_minutes = buffer_minutes


class MarketHoursValidator:
    """Placeholder - import from your original bot"""
    @staticmethod
    def get_market_status(symbol):
        return True, "Market Open"
    
    @staticmethod
    def get_trading_session():
        return "London", 100.0


class TechnicalIndicators:
    """Placeholder - import from your original bot"""
    @staticmethod
    def add_all_indicators(df):
        return df


class MarketStructureAnalyzer:
    """Placeholder - import from your original bot"""
    @staticmethod
    def determine_trend_direction(df):
        return TrendDirection.BULLISH


class RiskCalculator:
    """Placeholder - import from your original bot"""
    @staticmethod
    def calculate_trade_levels(price, atr, action, symbol, balance, signal_type):
        return {
            'entry': price,
            'stop_loss': price - atr if action == "BUY" else price + atr,
            'tp1': price + atr if action == "BUY" else price - atr,
            'tp2': price + atr * 2 if action == "BUY" else price - atr * 2,
            'tp3': price + atr * 3 if action == "BUY" else price - atr * 3,
            'breakeven': price,
            'sl_pips': 15.0,
            'tp1_pips': 20.0,
            'tp2_pips': 40.0,
            'tp3_pips': 60.0,
            'rr_ratio': 2.0
        }
    
    @staticmethod
    def calculate_position_size(balance, risk_pct, sl_pips, symbol):
        return 0.01, balance * (risk_pct / 100)


# ============================================================================
# SLIPPAGE & EXECUTION MODEL
# ============================================================================

class ExecutionModel:
    """
    Models realistic trade execution with slippage and spread
    """
    
    @staticmethod
    def apply_slippage(price: float, action: str, symbol: str, 
                       signal_type, liquidity: float) -> Tuple[float, float]:
        """
        Apply realistic slippage based on signal type and liquidity
        
        Returns: (executed_price, slippage_pips)
        """
        # Base slippage in pips
        if signal_type == SignalType.SCALP:
            base_slippage = 2.0  # Scalps get more slippage
        elif signal_type == SignalType.DAY_TRADE:
            base_slippage = 1.5
        else:
            base_slippage = 1.0
        
        # Adjust for liquidity
        liquidity_factor = 1.0
        if liquidity < 60:
            liquidity_factor = 1.5  # 50% more slippage in low liquidity
        elif liquidity >= 100:
            liquidity_factor = 0.7  # 30% less slippage in peak liquidity
        
        total_slippage_pips = base_slippage * liquidity_factor
        
        # Convert to price
        pip_size = 0.01 if symbol == 'XAUUSD' else 1.0
        slippage_price = total_slippage_pips * pip_size
        
        # Apply slippage (always against the trader)
        if action == "BUY":
            executed_price = price + slippage_price
        else:
            executed_price = price - slippage_price
        
        return executed_price, total_slippage_pips
    
    @staticmethod
    def get_spread(symbol: str, liquidity: float) -> float:
        """Get current spread in pips"""
        # Base spreads
        base_spread = {
            'XAUUSD': 3.0,  # 3 pips typical
            'BTCUSD': 5.0   # 5 pips typical
        }.get(symbol, 2.0)
        
        # Widen spread in low liquidity
        if liquidity < 60:
            base_spread *= 1.5
        
        return base_spread


# ============================================================================
# SIMPLIFIED SIGNAL SCORING (Core Indicators Only)
# ============================================================================

class SimplifiedSignalScorer:
    """
    Simplified scoring using only 4 core indicators:
    - EMA (trend)
    - RSI (momentum)
    - ADX (trend strength)
    - ATR (volatility)
    
    Removes: MACD, Bollinger Bands, Stochastic to avoid overfitting
    """
    
    MAX_SCORE = 100.0
    
    MIN_CONFIDENCE_SCALP = 80.0
    MIN_CONFIDENCE_DAY = 75.0
    MIN_CONFIDENCE_SWING = 70.0
    
    @staticmethod
    def score_trend_alignment(df_entry: pd.DataFrame, df_trend: pd.DataFrame, action: str) -> Tuple[float, List[str]]:
        """Score EMA trend alignment (0-35 points)"""
        score = 0.0
        reasons = []
        
        last_entry = df_entry.iloc[-1]
        last_trend = df_trend.iloc[-1]
        
        if action == "BUY":
            # Entry timeframe
            if last_entry['ema_9'] > last_entry['ema_21'] > last_entry['ema_50']:
                score += 15.0
                reasons.append("✅ Perfect EMA alignment (entry)")
            elif last_entry['ema_9'] > last_entry['ema_21']:
                score += 10.0
                reasons.append("EMA bullish (entry)")
            else:
                return 0.0, ["❌ No bullish EMA trend"]
            
            # Higher timeframe
            if last_trend['ema_9'] > last_trend['ema_21'] > last_trend['ema_50']:
                score += 20.0
                reasons.append("✅ HTF EMA aligned")
            elif last_trend['ema_9'] > last_trend['ema_21']:
                score += 12.0
                reasons.append("HTF EMA bullish")
            else:
                score += 5.0
                reasons.append("⚠️ Weak HTF trend")
        
        else:  # SELL
            if last_entry['ema_9'] < last_entry['ema_21'] < last_entry['ema_50']:
                score += 15.0
                reasons.append("✅ Perfect EMA alignment (entry)")
            elif last_entry['ema_9'] < last_entry['ema_21']:
                score += 10.0
                reasons.append("EMA bearish (entry)")
            else:
                return 0.0, ["❌ No bearish EMA trend"]
            
            if last_trend['ema_9'] < last_trend['ema_21'] < last_trend['ema_50']:
                score += 20.0
                reasons.append("✅ HTF EMA aligned")
            elif last_trend['ema_9'] < last_trend['ema_21']:
                score += 12.0
                reasons.append("HTF EMA bearish")
            else:
                score += 5.0
                reasons.append("⚠️ Weak HTF trend")
        
        return score, reasons
    
    @staticmethod
    def score_momentum(df: pd.DataFrame, action: str) -> Tuple[float, List[str]]:
        """Score RSI momentum (0-25 points)"""
        score = 0.0
        reasons = []
        
        last = df.iloc[-1]
        
        if action == "BUY":
            if 30 <= last['rsi'] <= 50:
                score = 25.0
                reasons.append(f"✅ RSI optimal ({last['rsi']:.1f})")
            elif 50 < last['rsi'] < 60:
                score = 18.0
                reasons.append(f"RSI good ({last['rsi']:.1f})")
            elif last['rsi'] < 30:
                score = 10.0
                reasons.append(f"⚠️ RSI oversold ({last['rsi']:.1f})")
            elif last['rsi'] >= 70:
                return 0.0, [f"❌ RSI overbought ({last['rsi']:.1f})"]
            else:
                score = 12.0
                reasons.append(f"RSI neutral ({last['rsi']:.1f})")
        
        else:  # SELL
            if 50 <= last['rsi'] <= 70:
                score = 25.0
                reasons.append(f"✅ RSI optimal ({last['rsi']:.1f})")
            elif 40 < last['rsi'] < 50:
                score = 18.0
                reasons.append(f"RSI good ({last['rsi']:.1f})")
            elif last['rsi'] > 70:
                score = 10.0
                reasons.append(f"⚠️ RSI overbought ({last['rsi']:.1f})")
            elif last['rsi'] <= 30:
                return 0.0, [f"❌ RSI oversold ({last['rsi']:.1f})"]
            else:
                score = 12.0
                reasons.append(f"RSI neutral ({last['rsi']:.1f})")
        
        return score, reasons
    
    @staticmethod
    def score_trend_strength(df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Score ADX trend strength (0-25 points)"""
        score = 0.0
        reasons = []
        
        last = df.iloc[-1]
        
        if last['adx'] > 30:
            score = 25.0
            reasons.append(f"✅ Very strong trend (ADX {last['adx']:.1f})")
        elif last['adx'] > 25:
            score = 20.0
            reasons.append(f"✅ Strong trend (ADX {last['adx']:.1f})")
        elif last['adx'] > 20:
            score = 12.0
            reasons.append(f"Good trend (ADX {last['adx']:.1f})")
        elif last['adx'] > 15:
            score = 6.0
            reasons.append(f"⚠️ Weak trend (ADX {last['adx']:.1f})")
        else:
            return 0.0, [f"❌ No trend (ADX {last['adx']:.1f})"]
        
        return score, reasons
    
    @staticmethod
    def score_volatility(df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Score ATR volatility (0-15 points)"""
        score = 0.0
        reasons = []
        
        last = df.iloc[-1]
        recent = df.tail(20)
        avg_atr = recent['atr'].mean()
        
        if last['atr'] > avg_atr * 1.3:
            score = 15.0
            reasons.append("✅ High volatility (good for trading)")
        elif last['atr'] > avg_atr * 1.1:
            score = 12.0
            reasons.append("Above-avg volatility")
        elif last['atr'] > avg_atr * 0.8:
            score = 8.0
            reasons.append("Normal volatility")
        else:
            score = 4.0
            reasons.append("⚠️ Low volatility")
        
        return score, reasons
    
    @staticmethod
    def calculate_confidence(df_entry: pd.DataFrame, df_trend: pd.DataFrame, action: str,
                            session: str, liquidity: float) -> Tuple[float, List[str]]:
        """
        Calculate confidence using only 4 core indicators
        Total: 100 points (35 + 25 + 25 + 15)
        """
        all_reasons = []
        total_score = 0.0
        
        # 1. EMA Trend Alignment (35 points)
        trend_score, trend_reasons = SimplifiedSignalScorer.score_trend_alignment(df_entry, df_trend, action)
        total_score += trend_score
        all_reasons.extend(trend_reasons)
        
        if trend_score < 15.0:
            return 0.0, ["❌ Insufficient trend alignment"]
        
        # 2. RSI Momentum (25 points)
        momentum_score, momentum_reasons = SimplifiedSignalScorer.score_momentum(df_entry, action)
        total_score += momentum_score
        all_reasons.extend(momentum_reasons)
        
        if momentum_score == 0.0:
            return 0.0, momentum_reasons
        
        # 3. ADX Trend Strength (25 points)
        adx_score, adx_reasons = SimplifiedSignalScorer.score_trend_strength(df_entry)
        total_score += adx_score
        all_reasons.extend(adx_reasons)
        
        if adx_score < 6.0:
            return 0.0, ["❌ Trend too weak"]
        
        # 4. ATR Volatility (15 points)
        vol_score, vol_reasons = SimplifiedSignalScorer.score_volatility(df_entry)
        total_score += vol_score
        all_reasons.extend(vol_reasons)
        
        return total_score, all_reasons


# ============================================================================
# ENHANCED TRADING BOT v13.0 (Production Ready)
# ============================================================================

class ProductionTradingBot:
    """
    Production-ready trading bot with:
    - Simplified indicators (4 core only)
    - Slippage modeling
    - Performance tracking
    - Position management
    """
    
    SYMBOLS = ['XAUUSD', 'BTCUSD']
    
    def __init__(self, 
                 telegram_token: str,
                 main_chat_id: str,
                 simple_chat_id: str,
                 account_balance: float = 500.0,
                 risk_percent: float = 2.0,
                 deriv_app_id: str = "1089",
                 enable_news_filter: bool = True,
                 enable_scalping: bool = True,
                 enable_day_trading: bool = True,
                 enable_swing_trading: bool = True):
        
        self.notifier = TelegramNotifier(telegram_token, main_chat_id, simple_chat_id)
        self.data_fetcher = DerivDataFetcher(deriv_app_id)
        self.news_monitor = NewsMonitor(buffer_minutes=60)
        
        self.trade_journal = TradeJournal()
        self.position_manager = PositionManager()
        
        self.account_balance = account_balance
        self.risk_percent = risk_percent
        self.enable_news_filter = enable_news_filter
        
        self.enable_scalping = enable_scalping
        self.enable_day_trading = enable_day_trading
        self.enable_swing_trading = enable_swing_trading
        
        self.recent_signals = deque(maxlen=20)
        self.news_alerts_sent = set()
        
        logger.info("=" * 80)
        logger.info("PRODUCTION TRADING BOT v13.0")
        logger.info("✅ Simplified indicators (EMA, RSI, ADX, ATR)")
        logger.info("✅ Slippage modeling")
        logger.info("✅ Performance tracking")
        logger.info("✅ Position management")
        logger.info(f"Balance: ${account_balance:.2f} | Risk: {risk_percent}%")
        logger.info("=" * 80)
    
    def analyze_symbol(self, symbol: str, signal_type: SignalType) -> Optional[TradeSignal]:
        """Analyze symbol with simplified scoring"""
        logger.info(f"\n{'='*60}")
        logger.info(f"ANALYZING {symbol} - {signal_type.value}")
        logger.info(f"{'='*60}")
        
        # ... existing code for market hours, session, news checks ...
        
        is_open, market_status = MarketHoursValidator.get_market_status(symbol)
        if not is_open:
            logger.info(f"⚪ {market_status}")
            return None
        
        session, liquidity = MarketHoursValidator.get_trading_session()
        logger.info(f"📍 {session} session (liquidity: {liquidity:.0f}%)")
        
        if signal_type == SignalType.SCALP and liquidity < 60:
            logger.info("❌ Liquidity too low for scalping")
            return None
        
        # ... existing code for data fetching and indicator calculation ...
        
        if signal_type == SignalType.SCALP:
            entry_tf, trend_tf = '5m', '15m'
            entry_count, trend_count = 200, 200
        elif signal_type == SignalType.DAY_TRADE:
            entry_tf, trend_tf = '15m', '1h'
            entry_count, trend_count = 300, 500
        else:
            entry_tf, trend_tf = '1h', '4h'
            entry_count, trend_count = 500, 200
        
        df_entry = self.data_fetcher.get_historical_data(symbol, entry_tf, entry_count)
        df_trend = self.data_fetcher.get_historical_data(symbol, trend_tf, trend_count)
        
        if df_entry is None or df_trend is None:
            return None
        
        df_entry = TechnicalIndicators.add_all_indicators(df_entry)
        df_trend = TechnicalIndicators.add_all_indicators(df_trend)
        
        # Determine action
        trend_entry = MarketStructureAnalyzer.determine_trend_direction(df_entry)
        
        action = None
        if trend_entry in [TrendDirection.STRONG_BULLISH, TrendDirection.BULLISH]:
            action = "BUY"
        elif trend_entry in [TrendDirection.STRONG_BEARISH, TrendDirection.BEARISH]:
            action = "SELL"
        else:
            logger.info("❌ No clear trend")
            return None
        
        logger.info(f"🎯 Calculating confidence (simplified scoring)...")
        confidence, reasons = SimplifiedSignalScorer.calculate_confidence(
            df_entry, df_trend, action, session, liquidity
        )
        
        logger.info(f"📊 Confidence Score: {confidence:.1f}%")
        
        min_conf = {
            SignalType.SCALP: SimplifiedSignalScorer.MIN_CONFIDENCE_SCALP,
            SignalType.DAY_TRADE: SimplifiedSignalScorer.MIN_CONFIDENCE_DAY,
            SignalType.SWING: SimplifiedSignalScorer.MIN_CONFIDENCE_SWING
        }[signal_type]
        
        if confidence < min_conf:
            logger.info(f"❌ Below {min_conf}% threshold")
            return None
        
        # Calculate trade levels
        last_candle = df_entry.iloc[-1]
        current_price = last_candle['close']
        atr = last_candle['atr']
        
        levels = RiskCalculator.calculate_trade_levels(
            current_price, atr, action, symbol, self.account_balance, signal_type
        )
        
        executed_entry, slippage_pips = ExecutionModel.apply_slippage(
            levels['entry'], action, symbol, signal_type, liquidity
        )
        spread_pips = ExecutionModel.get_spread(symbol, liquidity)
        
        logger.info(f"💰 Entry: {levels['entry']:.5f} → {executed_entry:.5f} (slippage: {slippage_pips:.1f}p, spread: {spread_pips:.1f}p)")
        
        # Adjust levels for slippage
        levels['entry'] = executed_entry
        
        lot_size, risk_amount = RiskCalculator.calculate_position_size(
            self.account_balance, self.risk_percent, levels['sl_pips'], symbol
        )
        
        # Create signal (similar to before but with slippage-adjusted entry)
        signal_id = f"{symbol}_{signal_type.value}_{int(datetime.now().timestamp())}"
        
        # ... rest of signal creation code ...
        
        quality = SignalQuality.EXCELLENT if confidence >= 90 else (
            SignalQuality.STRONG if confidence >= 80 else SignalQuality.GOOD
        )
        
        signal = TradeSignal(
            signal_id=signal_id,
            symbol=symbol,
            action=action,
            signal_type=signal_type,
            confidence=confidence,
            quality=quality,
            entry_price=levels['entry'],
            stop_loss=levels['stop_loss'],
            take_profit_1=levels['tp1'],
            take_profit_2=levels['tp2'],
            take_profit_3=levels['tp3'],
            breakeven_price=levels['breakeven'],
            nearest_support=None,
            nearest_resistance=None,
            key_level=None,
            stop_loss_pips=levels['sl_pips'],
            tp1_pips=levels['tp1_pips'],
            tp2_pips=levels['tp2_pips'],
            tp3_pips=levels['tp3_pips'],
            position_size=lot_size,
            risk_amount_usd=risk_amount,
            risk_reward_ratio=levels['rr_ratio'],
            timeframe=entry_tf,
            htf_timeframe=trend_tf,
            htf_trend=MarketStructureAnalyzer.determine_trend_direction(df_trend).value,
            market_phase="TRENDING",
            trading_session=session,
            strategy_components=reasons,
            technical_summary={},
            timestamp=datetime.now(pytz.UTC),
            valid_until=datetime.now(pytz.UTC) + timedelta(hours=2)
        )
        
        trade_record = TradeRecord(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            action=signal.action,
            signal_type=signal.signal_type.value,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            take_profit_3=signal.take_profit_3,
            position_size=signal.position_size,
            confidence=signal.confidence,
            timestamp=signal.timestamp.isoformat()
        )
        self.trade_journal.log_signal(trade_record)
        
        return signal
    
    def run_analysis_cycle(self) -> int:
        """Run analysis cycle with performance tracking"""
        logger.info("\n" + "=" * 80)
        logger.info("🚀 PRODUCTION TRADING BOT v13.0 - ANALYSIS CYCLE")
        logger.info(f"⏰ {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        stats = self.trade_journal.get_performance_stats(days=30)
        if stats['total_trades'] > 0:
            logger.info(f"📊 Last 30 days: {stats['total_trades']} trades, {stats['win_rate']:.1f}% win rate, ${stats['total_pnl_usd']:.2f} P&L")
        
        logger.info("=" * 80)
        
        signals_generated = 0
        
        for symbol in self.SYMBOLS:
            signal_types = []
            if self.enable_scalping:
                signal_types.append(SignalType.SCALP)
            if self.enable_day_trading:
                signal_types.append(SignalType.DAY_TRADE)
            if self.enable_swing_trading:
                signal_types.append(SignalType.SWING)
            
            for sig_type in signal_types:
                try:
                    signal = self.analyze_symbol(symbol, sig_type)
                    
                    if signal and not self.is_duplicate_signal(signal):
                        if self.notifier.send_signal(signal):
                            self.recent_signals.append(signal)
                            signals_generated += 1
                            logger.info(f"✅ Signal sent: {signal.signal_id}")
                
                except Exception as e:
                    logger.error(f"❌ Error: {e}", exc_info=True)
        
        logger.info(f"\n✅ CYCLE COMPLETE - {signals_generated} signals generated")
        return signals_generated
    
    def is_duplicate_signal(self, signal: TradeSignal) -> bool:
        """Check for duplicate signals"""
        for recent in self.recent_signals:
            if (recent.symbol == signal.symbol and 
                recent.action == signal.action and 
                recent.signal_type == signal.signal_type):
                time_diff = (signal.timestamp - recent.timestamp).total_seconds()
                cooldown = 3600 if signal.signal_type == SignalType.SCALP else 7200
                if time_diff < cooldown:
                    return True
        return False



def main():
    """Main entry point"""
    
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    main_chat_id = os.getenv('MAIN_CHAT_ID')
    simple_chat_id = os.getenv('SIMPLE_CHAT_ID')
    
    if not all([telegram_token, main_chat_id, simple_chat_id]):
        logger.error("❌ Missing required environment variables")
        return 1
    
    try:
        bot = ProductionTradingBot(
            telegram_token=telegram_token,
            main_chat_id=main_chat_id,
            simple_chat_id=simple_chat_id,
            account_balance=float(os.getenv('ACCOUNT_BALANCE', '500')),
            risk_percent=float(os.getenv('RISK_PERCENT', '2.0')),
            enable_scalping=os.getenv('ENABLE_SCALPING', 'true').lower() == 'true',
            enable_day_trading=os.getenv('ENABLE_DAY_TRADING', 'true').lower() == 'true',
            enable_swing_trading=os.getenv('ENABLE_SWING_TRADING', 'true').lower() == 'true'
        )
        
        signals_count = bot.run_analysis_cycle()
        
        logger.info(f"✅ Bot execution completed - {signals_count} signals")
        return 0
    
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
