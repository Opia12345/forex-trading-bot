"""
SYNTHETIC INDICES TRADING BOT v1.0
Specialized for Deriv Volatility Indices (V10, V25, V50, V75, V100)
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
import requests
import json
import asyncio
import websockets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('synthetic_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SignalQuality(Enum):
    """Signal quality rating"""
    EXCELLENT = "EXCELLENT"
    STRONG = "STRONG"


@dataclass
class TradeSignal:
    """Complete trade signal"""
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
    stop_loss_ticks: float
    tp1_ticks: float
    tp2_ticks: float
    tp3_ticks: float
    stake_usd: float
    risk_reward_ratio: float
    strategy_components: List[str]
    volatility_level: str
    timestamp: datetime


# ============================================================================
# TELEGRAM NOTIFIER
# ============================================================================

class TelegramNotifier:
    """Sends trading signals to main Telegram group"""
    
    def __init__(self, token: str, main_chat_id: str):
        self.token = token
        self.main_chat_id = main_chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_signal(self, signal: TradeSignal) -> bool:
        """Send signal to Telegram"""
        try:
            quality_emoji = "🌟" if signal.quality.value == "EXCELLENT" else "⭐"
            action_emoji = "🟢" if signal.action == "RISE" else "🔴"
            
            message = f"""
{quality_emoji} <b>SYNTHETIC SIGNAL</b> {quality_emoji}

{action_emoji} <b>{signal.action} {signal.symbol}</b>
💯 Confidence: {signal.confidence:.1f}%

<b>📍 ENTRY LEVELS</b>
💵 Entry: <code>{signal.entry_price:.2f}</code>
🛑 Stop Loss: <code>{signal.stop_loss:.2f}</code> (-{signal.stop_loss_ticks:.0f} ticks)
🎯 TP1: <code>{signal.take_profit_1:.2f}</code> (+{signal.tp1_ticks:.0f} ticks)
🎯 TP2: <code>{signal.take_profit_2:.2f}</code> (+{signal.tp2_ticks:.0f} ticks)
🎯 TP3: <code>{signal.take_profit_3:.2f}</code> (+{signal.tp3_ticks:.0f} ticks)

<b>💼 POSITION</b>
💰 Stake: ${signal.stake_usd:.2f}
📊 R:R Ratio: 1:{signal.risk_reward_ratio:.1f}
🔥 Volatility: {signal.volatility_level}

<b>✅ WHY THIS SIGNAL</b>
"""
            
            for reason in signal.strategy_components[:6]:
                message += f"• {reason}\n"
            
            message += f"""
<b>⚠️ MANAGEMENT</b>
• Close 50% at TP1
• Close 30% at TP2
• Trail 20% to TP3

<i>ID: {signal.signal_id}</i>
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
                logger.error(f"Telegram API error: {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False


# ============================================================================
# DERIV DATA FETCHER
# ============================================================================

class DerivDataFetcher:
    """Fetch synthetic indices data from Deriv"""
    
    SYMBOLS = {
        'V10': 'R_10',
        'V25': 'R_25',
        'V50': 'R_50',
        'V75': 'R_75',
        'V100': 'R_100'
    }
    
    TICK_SIZES = {
        'V10': 0.01,
        'V25': 0.01,
        'V50': 0.01,
        'V75': 0.01,
        'V100': 0.01
    }
    
    def __init__(self, app_id: str = "1089"):
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    
    async def _fetch_ticks_async(self, symbol: str, count: int = 1000) -> Optional[List]:
        """Fetch tick data"""
        try:
            async with websockets.connect(self.ws_url, ping_interval=30) as ws:
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
                    logger.error(f"API error: {data['error']}")
                    return None
                
                if 'history' in data and 'prices' in data['history']:
                    return data['history']['prices']
                
                return None
        
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return None
    
    def get_tick_data(self, symbol: str, count: int = 1000) -> Optional[pd.DataFrame]:
        """Get tick data as DataFrame"""
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
            
            if not prices or len(prices) < 100:
                logger.error(f"Insufficient tick data: {len(prices) if prices else 0}")
                return None
            
            df = pd.DataFrame({'price': prices})
            df['tick'] = range(len(df))
            
            logger.info(f"✅ Fetched {len(df)} ticks for {symbol}")
            return df
        
        except Exception as e:
            logger.error(f"Error getting tick data: {e}")
            return None
    
    def get_tick_size(self, symbol: str) -> float:
        """Get tick size for symbol"""
        return self.TICK_SIZES.get(symbol, 0.01)


# ============================================================================
# SYNTHETIC INDICATORS
# ============================================================================

class SyntheticIndicators:
    """Technical indicators for synthetic indices"""
    
    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, period: int = 20, std: float = 2.0) -> Tuple:
        """Bollinger Bands"""
        sma = series.rolling(window=period).mean()
        rolling_std = series.rolling(window=period).std()
        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        return upper, sma, lower
    
    @staticmethod
    def calculate_volatility(series: pd.Series, period: int = 20) -> pd.Series:
        """Rolling volatility"""
        returns = series.pct_change()
        volatility = returns.rolling(window=period).std() * 100
        return volatility
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all indicators"""
        df['ema_10'] = SyntheticIndicators.calculate_ema(df['price'], 10)
        df['ema_20'] = SyntheticIndicators.calculate_ema(df['price'], 20)
        df['ema_50'] = SyntheticIndicators.calculate_ema(df['price'], 50)
        df['ema_100'] = SyntheticIndicators.calculate_ema(df['price'], 100)
        
        df['rsi'] = SyntheticIndicators.calculate_rsi(df['price'], 14)
        
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = \
            SyntheticIndicators.calculate_bollinger_bands(df['price'], 20, 2.0)
        
        df['volatility'] = SyntheticIndicators.calculate_volatility(df['price'], 20)
        
        df = df.fillna(method='bfill')
        
        return df


# ============================================================================
# SIGNAL ANALYZER
# ============================================================================

class SignalAnalyzer:
    """Analyze signals for synthetic indices"""
    
    @staticmethod
    def analyze_trend(df: pd.DataFrame) -> Tuple[str, float, List[str]]:
        """Analyze trend direction"""
        last = df.iloc[-1]
        score = 0.0
        reasons = []
        
        # EMA alignment
        if last['ema_10'] > last['ema_20'] > last['ema_50'] > last['ema_100']:
            score += 30.0
            action = "RISE"
            reasons.append("✅ Perfect bullish EMA alignment")
        elif last['ema_10'] < last['ema_20'] < last['ema_50'] < last['ema_100']:
            score += 30.0
            action = "FALL"
            reasons.append("✅ Perfect bearish EMA alignment")
        elif last['ema_10'] > last['ema_20'] > last['ema_50']:
            score += 20.0
            action = "RISE"
            reasons.append("Strong bullish trend")
        elif last['ema_10'] < last['ema_20'] < last['ema_50']:
            score += 20.0
            action = "FALL"
            reasons.append("Strong bearish trend")
        elif last['ema_10'] > last['ema_20']:
            score += 10.0
            action = "RISE"
            reasons.append("Weak bullish trend")
        elif last['ema_10'] < last['ema_20']:
            score += 10.0
            action = "FALL"
            reasons.append("Weak bearish trend")
        else:
            return None, 0.0, ["No clear trend"]
        
        return action, score, reasons
    
    @staticmethod
    def analyze_momentum(df: pd.DataFrame, action: str) -> Tuple[float, List[str]]:
        """Analyze RSI momentum"""
        last = df.iloc[-1]
        score = 0.0
        reasons = []
        
        if action == "RISE":
            if 40 <= last['rsi'] <= 55:
                score = 25.0
                reasons.append(f"✅ RSI optimal ({last['rsi']:.1f})")
            elif 30 <= last['rsi'] < 40:
                score = 20.0
                reasons.append(f"RSI oversold bounce ({last['rsi']:.1f})")
            elif 55 < last['rsi'] < 65:
                score = 15.0
                reasons.append(f"RSI bullish ({last['rsi']:.1f})")
            elif last['rsi'] >= 70:
                return 0.0, [f"❌ RSI overbought ({last['rsi']:.1f})"]
            else:
                score = 8.0
                reasons.append(f"RSI neutral ({last['rsi']:.1f})")
        else:  # FALL
            if 45 <= last['rsi'] <= 60:
                score = 25.0
                reasons.append(f"✅ RSI optimal ({last['rsi']:.1f})")
            elif 60 < last['rsi'] <= 70:
                score = 20.0
                reasons.append(f"RSI overbought rejection ({last['rsi']:.1f})")
            elif 35 < last['rsi'] < 45:
                score = 15.0
                reasons.append(f"RSI bearish ({last['rsi']:.1f})")
            elif last['rsi'] <= 30:
                return 0.0, [f"❌ RSI oversold ({last['rsi']:.1f})"]
            else:
                score = 8.0
                reasons.append(f"RSI neutral ({last['rsi']:.1f})")
        
        return score, reasons
    
    @staticmethod
    def analyze_bollinger(df: pd.DataFrame, action: str) -> Tuple[float, List[str]]:
        """Analyze Bollinger Bands"""
        last = df.iloc[-1]
        score = 0.0
        reasons = []
        
        bb_position = (last['price'] - last['bb_lower']) / (last['bb_upper'] - last['bb_lower'])
        
        if action == "RISE":
            if bb_position < 0.3:
                score = 20.0
                reasons.append("✅ Price near lower BB (bounce setup)")
            elif 0.3 <= bb_position < 0.5:
                score = 15.0
                reasons.append("Price in lower half")
            elif bb_position > 0.8:
                return 0.0, ["❌ Price near upper BB"]
            else:
                score = 8.0
                reasons.append("Price mid-range")
        else:  # FALL
            if bb_position > 0.7:
                score = 20.0
                reasons.append("✅ Price near upper BB (rejection setup)")
            elif 0.5 < bb_position <= 0.7:
                score = 15.0
                reasons.append("Price in upper half")
            elif bb_position < 0.2:
                return 0.0, ["❌ Price near lower BB"]
            else:
                score = 8.0
                reasons.append("Price mid-range")
        
        return score, reasons
    
    @staticmethod
    def analyze_volatility(df: pd.DataFrame) -> Tuple[float, List[str], str]:
        """Analyze volatility level"""
        last = df.iloc[-1]
        recent = df.tail(50)
        avg_vol = recent['volatility'].mean()
        
        score = 0.0
        reasons = []
        
        if last['volatility'] > avg_vol * 1.5:
            score = 15.0
            vol_level = "VERY HIGH"
            reasons.append("🔥 Very high volatility")
        elif last['volatility'] > avg_vol * 1.2:
            score = 20.0
            vol_level = "HIGH"
            reasons.append("✅ High volatility (ideal)")
        elif last['volatility'] > avg_vol * 0.8:
            score = 12.0
            vol_level = "NORMAL"
            reasons.append("Normal volatility")
        else:
            score = 5.0
            vol_level = "LOW"
            reasons.append("⚠️ Low volatility")
        
        return score, reasons, vol_level
    
    @staticmethod
    def calculate_confidence(df: pd.DataFrame) -> Tuple[Optional[str], float, List[str], str]:
        """Calculate overall confidence"""
        all_reasons = []
        total_score = 0.0
        
        # 1. Trend (30 points)
        action, trend_score, trend_reasons = SignalAnalyzer.analyze_trend(df)
        if not action:
            return None, 0.0, trend_reasons, "NONE"
        
        total_score += trend_score
        all_reasons.extend(trend_reasons)
        
        if trend_score < 10.0:
            return None, 0.0, ["❌ Trend too weak"], "NONE"
        
        # 2. Momentum (25 points)
        momentum_score, momentum_reasons = SignalAnalyzer.analyze_momentum(df, action)
        if momentum_score == 0.0:
            return None, 0.0, momentum_reasons, "NONE"
        
        total_score += momentum_score
        all_reasons.extend(momentum_reasons)
        
        # 3. Bollinger (20 points)
        bb_score, bb_reasons = SignalAnalyzer.analyze_bollinger(df, action)
        if bb_score == 0.0:
            return None, 0.0, bb_reasons, "NONE"
        
        total_score += bb_score
        all_reasons.extend(bb_reasons)
        
        # 4. Volatility (15 points)
        vol_score, vol_reasons, vol_level = SignalAnalyzer.analyze_volatility(df)
        total_score += vol_score
        all_reasons.extend(vol_reasons)
        
        # 5. Recent price action (10 points)
        recent = df.tail(10)
        price_change = ((recent.iloc[-1]['price'] - recent.iloc[0]['price']) / recent.iloc[0]['price']) * 100
        
        if action == "RISE" and price_change > 0:
            total_score += 10.0
            all_reasons.append(f"Recent momentum +{price_change:.2f}%")
        elif action == "FALL" and price_change < 0:
            total_score += 10.0
            all_reasons.append(f"Recent momentum {price_change:.2f}%")
        else:
            total_score += 5.0
            all_reasons.append("Mixed recent action")
        
        return action, total_score, all_reasons, vol_level


# ============================================================================
# SYNTHETIC TRADING BOT
# ============================================================================

class SyntheticTradingBot:
    """Trading bot for synthetic indices"""
    
    SYMBOLS = ['V10', 'V25', 'V50', 'V75', 'V100']
    
    # Minimum 80% confidence for all symbols
    MIN_CONFIDENCE = 80.0
    
    def __init__(self, telegram_token: str, main_chat_id: str, stake_usd: float = 10.0):
        self.notifier = TelegramNotifier(telegram_token, main_chat_id)
        self.data_fetcher = DerivDataFetcher()
        self.stake_usd = stake_usd
        self.recent_signals = deque(maxlen=20)
        
        logger.info("=" * 80)
        logger.info("SYNTHETIC INDICES TRADING BOT v1.0")
        logger.info("Signals: 80%+ Confidence Only")
        logger.info("Symbols: V10, V25, V50, V75, V100")
        logger.info(f"Stake: ${stake_usd:.2f}")
        logger.info("=" * 80)
    
    def analyze_symbol(self, symbol: str) -> Optional[TradeSignal]:
        """Analyze symbol for trading signals"""
        logger.info(f"\n{'='*60}")
        logger.info(f"ANALYZING {symbol}")
        logger.info(f"{'='*60}")
        
        # Fetch tick data
        logger.info("📥 Fetching tick data...")
        df = self.data_fetcher.get_tick_data(symbol, count=1000)
        
        if df is None:
            logger.error("❌ Failed to fetch data")
            return None
        
        # Calculate indicators
        logger.info("🔧 Calculating indicators...")
        df = SyntheticIndicators.add_all_indicators(df)
        
        # Analyze for signals
        logger.info("🎯 Analyzing signal...")
        action, confidence, reasons, vol_level = SignalAnalyzer.calculate_confidence(df)
        
        if not action:
            logger.info("❌ No signal")
            return None
        
        logger.info(f"📊 Confidence: {confidence:.1f}%")
        
        if confidence < self.MIN_CONFIDENCE:
            logger.info(f"❌ Below {self.MIN_CONFIDENCE}% threshold")
            return None
        
        logger.info(f"✅ SIGNAL QUALIFIED ({confidence:.1f}%)")
        
        # Calculate trade levels
        current_price = df.iloc[-1]['price']
        tick_size = self.data_fetcher.get_tick_size(symbol)
        
        # Dynamic stop loss based on volatility
        volatility_multiplier = {
            'V10': 15,
            'V25': 20,
            'V50': 25,
            'V75': 30,
            'V100': 35
        }[symbol]
        
        if vol_level == "VERY HIGH":
            volatility_multiplier *= 1.5
        elif vol_level == "HIGH":
            volatility_multiplier *= 1.2
        
        sl_ticks = volatility_multiplier
        tp1_ticks = sl_ticks * 1.5
        tp2_ticks = sl_ticks * 2.5
        tp3_ticks = sl_ticks * 4.0
        
        if action == "RISE":
            entry = current_price
            stop_loss = entry - (sl_ticks * tick_size)
            tp1 = entry + (tp1_ticks * tick_size)
            tp2 = entry + (tp2_ticks * tick_size)
            tp3 = entry + (tp3_ticks * tick_size)
        else:  # FALL
            entry = current_price
            stop_loss = entry + (sl_ticks * tick_size)
            tp1 = entry - (tp1_ticks * tick_size)
            tp2 = entry - (tp2_ticks * tick_size)
            tp3 = entry - (tp3_ticks * tick_size)
        
        rr_ratio = tp3_ticks / sl_ticks if sl_ticks > 0 else 0
        
        # Determine quality based on confidence
        quality = SignalQuality.EXCELLENT if confidence >= 90 else SignalQuality.STRONG
        
        # Create signal
        signal_id = f"{symbol}_{int(datetime.now().timestamp())}"
        timestamp = datetime.now(pytz.UTC)
        
        signal = TradeSignal(
            signal_id=signal_id,
            symbol=symbol,
            action=action,
            confidence=confidence,
            quality=quality,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            stop_loss_ticks=sl_ticks,
            tp1_ticks=tp1_ticks,
            tp2_ticks=tp2_ticks,
            tp3_ticks=tp3_ticks,
            stake_usd=self.stake_usd,
            risk_reward_ratio=rr_ratio,
            strategy_components=reasons,
            volatility_level=vol_level,
            timestamp=timestamp
        )
        
        return signal
    
    def is_duplicate_signal(self, signal: TradeSignal) -> bool:
        """Check for duplicate signals"""
        for recent in self.recent_signals:
            if recent.symbol == signal.symbol and recent.action == signal.action:
                time_diff = (signal.timestamp - recent.timestamp).total_seconds()
                if time_diff < 1800:  # 30 minutes cooldown
                    return True
        return False
    
    def run_analysis_cycle(self) -> int:
        """Run analysis cycle"""
        logger.info("\n" + "=" * 80)
        logger.info("🚀 SYNTHETIC INDICES BOT - ANALYSIS CYCLE")
        logger.info(f"⏰ {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info("=" * 80)
        
        signals_generated = 0
        
        for symbol in self.SYMBOLS:
            try:
                signal = self.analyze_symbol(symbol)
                
                if signal and not self.is_duplicate_signal(signal):
                    if self.notifier.send_signal(signal):
                        self.recent_signals.append(signal)
                        signals_generated += 1
                        logger.info(f"✅ Signal sent: {signal.signal_id}")
            
            except Exception as e:
                logger.error(f"❌ Error analyzing {symbol}: {e}")
        
        logger.info(f"\n✅ CYCLE COMPLETE - {signals_generated} signals generated")
        return signals_generated


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    main_chat_id = os.getenv('MAIN_CHAT_ID')
    
    if not telegram_token or not main_chat_id:
        logger.error("❌ Missing TELEGRAM_BOT_TOKEN or MAIN_CHAT_ID")
        return 1
    
    try:
        # Handle STAKE_USD with proper default
        stake_usd_str = os.getenv('STAKE_USD', '10')
        stake_usd = float(stake_usd_str) if stake_usd_str else 10.0
        
        bot = SyntheticTradingBot(
            telegram_token=telegram_token,
            main_chat_id=main_chat_id,
            stake_usd=stake_usd
        )
        
        signals = bot.run_analysis_cycle()
        logger.info(f"✅ Execution complete - {signals} signals generated")
        return 0
    
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())