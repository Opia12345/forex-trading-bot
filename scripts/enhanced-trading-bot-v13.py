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
import requests
import json
import asyncio
import websockets

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


@dataclass
class NewsEvent:
    """Economic news event"""
    title: str
    country: str
    date: datetime
    impact: str
    currency: str
    actual: str = ""
    forecast: str = ""
    previous: str = ""


# ============================================================================
# TELEGRAM NOTIFIER (Fixed)
# ============================================================================

class TelegramNotifier:
    """
    Handles Telegram notifications for trading signals
    - Main chat: Detailed analysis with all indicators and levels
    - Simple chat: Clean, simple signal format
    """
    
    def __init__(self, token: str, main_chat_id: str, simple_chat_id: str):
        self.token = token
        self.main_chat = main_chat_id
        self.simple_chat = simple_chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_signal(self, signal) -> bool:
        """
        Send signal to BOTH chats:
        - Main chat: Full detailed analysis
        - Simple chat: Clean simple format
        """
        try:
            # Send detailed message to MAIN CHAT
            main_success = self._send_detailed_signal(signal, self.main_chat)
            
            # Send simple message to SIMPLE CHAT
            simple_success = self._send_simple_signal(signal, self.simple_chat)
            
            if main_success and simple_success:
                logger.info(f"✅ Signal sent to both chats: {signal.signal_id}")
                return True
            elif main_success:
                logger.warning(f"⚠️ Signal sent to main chat only: {signal.signal_id}")
                return True
            else:
                logger.error(f"❌ Failed to send signal: {signal.signal_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending signal: {e}")
            return False
    
    def _send_detailed_signal(self, signal, chat_id: str) -> bool:
        """Send detailed signal to main chat"""
        
        # Quality emoji
        quality_emoji = {
            "EXCELLENT": "🌟",
            "STRONG": "⭐",
            "GOOD": "✨"
        }.get(signal.quality.value, "✨")
        
        # Signal type emoji
        type_emoji = {
            "SCALP": "⚡",
            "DAY_TRADE": "📊",
            "SWING": "📈"
        }.get(signal.signal_type.value, "📊")
        
        message = f"""
{quality_emoji} <b>TRADING SIGNAL</b> {quality_emoji}

{type_emoji} <b>Type:</b> {signal.signal_type.value}
💰 <b>Symbol:</b> {signal.symbol}
🎯 <b>Action:</b> {signal.action}
📊 <b>Confidence:</b> {signal.confidence:.1f}%

<b>📍 ENTRY LEVELS</b>
💵 Entry: <code>{signal.entry_price:.5f}</code>
🛑 Stop Loss: <code>{signal.stop_loss:.5f}</code> ({signal.stop_loss_pips:.1f} pips)
🎯 TP1: <code>{signal.take_profit_1:.5f}</code> ({signal.tp1_pips:.1f} pips)
🎯 TP2: <code>{signal.take_profit_2:.5f}</code> ({signal.tp2_pips:.1f} pips)
🎯 TP3: <code>{signal.take_profit_3:.5f}</code> ({signal.tp3_pips:.1f} pips)
⚖️ Breakeven: <code>{signal.breakeven_price:.5f}</code>

<b>💼 POSITION SIZING</b>
📦 Lot Size: {signal.position_size:.2f}
💰 Risk: ${signal.risk_amount_usd:.2f}
📊 R:R Ratio: 1:{signal.risk_reward_ratio:.1f}

<b>📊 MARKET CONTEXT</b>
⏰ Session: {signal.trading_session}
📈 HTF Trend: {signal.htf_trend}
🎭 Phase: {signal.market_phase}
⏱️ Timeframe: {signal.timeframe} / {signal.htf_timeframe}

<b>✅ STRATEGY COMPONENTS</b>
"""
        
        # Add strategy reasons
        for reason in signal.strategy_components[:8]:  # Limit to 8 reasons
            message += f"• {reason}\n"
        
        message += f"""
<b>⚠️ RISK MANAGEMENT</b>
• Close 50% at TP1, move SL to breakeven
• Close 30% at TP2, activate trailing stop
• Let remaining 20% run to TP3
• Never risk more than allocated amount

<i>Signal ID: {signal.signal_id}</i>
<i>Valid until: {signal.valid_until.strftime('%H:%M UTC')}</i>
"""
        
        return self._send_message(chat_id, message)
    
    def _send_simple_signal(self, signal, chat_id: str) -> bool:
        """Send simple signal to simple chat"""
        
        # Action emoji
        action_emoji = "🟢" if signal.action == "BUY" else "🔴"
        
        message = f"""
{action_emoji} <b>{signal.action} {signal.symbol}</b>

💵 Entry: <code>{signal.entry_price:.5f}</code>
🛑 SL: <code>{signal.stop_loss:.5f}</code>
🎯 TP1: <code>{signal.take_profit_1:.5f}</code>
🎯 TP2: <code>{signal.take_profit_2:.5f}</code>
🎯 TP3: <code>{signal.take_profit_3:.5f}</code>

📊 Confidence: {signal.confidence:.0f}%
⚡ Type: {signal.signal_type.value}
💼 Size: {signal.position_size:.2f} lots
"""
        
        return self._send_message(chat_id, message)
    
    def _send_message(self, chat_id: str, message: str) -> bool:
        """Send message via Telegram API"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def send_news_alert(self, news_events: List[NewsEvent]) -> bool:
        """Send high-impact news alert to both chats"""
        if not news_events:
            return True
        
        message = f"""
⚠️ <b>HIGH IMPACT NEWS ALERT</b>

{news_events[0].title}

Trading may be restricted during this period.
"""
        
        main_sent = self._send_message(self.main_chat, message)
        simple_sent = self._send_message(self.simple_chat, message)
        
        return main_sent or simple_sent


# ============================================================================
# DERIV DATA FETCHER
# ============================================================================

class DerivDataFetcher:
    """Fetch real-time and historical data from Deriv API"""
    
    SYMBOL_MAP = {
        'XAUUSD': 'frxXAUUSD',
        'BTCUSD': 'cryBTCUSD'
    }
    
    TIMEFRAME_MAP = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400
    }
    
    def __init__(self, app_id: str = "1089"):
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    
    async def _fetch_candles_async(self, symbol: str, granularity: int, count: int) -> Optional[Dict]:
        """Fetch candles via WebSocket (async)"""
        try:
            async with websockets.connect(self.ws_url, ping_interval=30, close_timeout=10) as ws:
                end_time = int(datetime.now().timestamp())
                start_time = end_time - (count * granularity)
                
                request = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": count,
                    "end": "latest",
                    "start": start_time,
                    "style": "candles",
                    "granularity": granularity
                }
                
                await ws.send(json.dumps(request))
                response = await asyncio.wait_for(ws.recv(), timeout=30)
                data = json.loads(response)
                
                if 'error' in data:
                    logger.error(f"Deriv API error: {data['error']}")
                    return None
                
                return data
        
        except asyncio.TimeoutError:
            logger.error(f"WebSocket timeout for {symbol}")
            return None
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            return None
    
    def get_historical_data(self, symbol: str, timeframe: str = '1h', count: int = 500) -> Optional[pd.DataFrame]:
        """Fetch historical candles and return as DataFrame"""
        
        deriv_symbol = self.SYMBOL_MAP.get(symbol)
        if not deriv_symbol:
            logger.error(f"Invalid symbol: {symbol}")
            return None
        
        granularity = self.TIMEFRAME_MAP.get(timeframe)
        if not granularity:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    self._fetch_candles_async(deriv_symbol, granularity, count)
                )
            finally:
                loop.close()
            
            if not response or 'candles' not in response:
                logger.error(f"No candle data returned for {symbol}")
                return None
            
            candles = response['candles']
            
            min_candles = 50 if timeframe == '5m' else 100
            if not candles or len(candles) < min_candles:
                logger.error(f"Insufficient candle data: {len(candles) if candles else 0}")
                return None
            
            df = pd.DataFrame(candles)
            
            if 'epoch' in df.columns:
                df['timestamp'] = pd.to_datetime(df['epoch'], unit='s')
                df.set_index('timestamp', inplace=True)
            
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            if len(df) < min_candles:
                logger.error(f"Insufficient valid data after cleaning: {len(df)}")
                return None
            
            logger.info(f"✅ Fetched {len(df)} candles for {symbol} ({timeframe})")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching {symbol} data: {e}")
            return None


# ============================================================================
# NEWS MONITOR
# ============================================================================

class NewsMonitor:
    """Monitor high-impact economic news with caching"""
    
    def __init__(self, buffer_minutes: int = 60):
        self.forex_factory_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        self.buffer_minutes = buffer_minutes
        self.cache_duration = 1800  # 30 minutes
        self.last_fetch = None
        self.cached_events: List[NewsEvent] = []
    
    def fetch_news_events(self) -> List[NewsEvent]:
        """Fetch upcoming high-impact news events"""
        if self.last_fetch and (datetime.now() - self.last_fetch).seconds < self.cache_duration:
            return self.cached_events
        
        try:
            response = requests.get(self.forex_factory_url, timeout=15)
            if response.status_code != 200:
                logger.warning(f"News API returned status {response.status_code}")
                return self.cached_events
            
            data = response.json()
            events = []
            
            for item in data:
                try:
                    impact = item.get('impact', '').upper()
                    if impact != 'HIGH':
                        continue
                    
                    date_str = item.get('date', '')
                    if not date_str:
                        continue
                    
                    event_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S%z')
                    
                    events.append(NewsEvent(
                        title=item.get('title', 'Unknown'),
                        country=item.get('country', 'Unknown'),
                        date=event_date,
                        impact=impact,
                        currency=item.get('currency', ''),
                        actual=item.get('actual', ''),
                        forecast=item.get('forecast', ''),
                        previous=item.get('previous', '')
                    ))
                except Exception:
                    continue
            
            self.cached_events = events
            self.last_fetch = datetime.now()
            logger.info(f"✅ Fetched {len(events)} HIGH-impact news events")
            return events
        
        except Exception as e:
            logger.error(f"News fetch error: {e}")
            return self.cached_events
    
    def check_news_safety(self, symbol: str, signal_type: SignalType) -> Tuple[bool, List[NewsEvent]]:
        """Check if it's safe to trade (no news within buffer period)"""
        events = self.fetch_news_events()
        now = datetime.now(pytz.UTC)
        
        if signal_type == SignalType.SCALP:
            buffer_minutes = 30
        elif signal_type == SignalType.DAY_TRADE:
            buffer_minutes = 45
        else:
            buffer_minutes = 60
        
        buffer_end = now + timedelta(minutes=buffer_minutes)
        
        if symbol == 'XAUUSD':
            relevant_currencies = ['USD', 'EUR', 'GBP']
        elif symbol == 'BTCUSD':
            relevant_currencies = ['USD']
        else:
            return True, []
        
        upcoming_news = []
        for event in events:
            if event.currency in relevant_currencies:
                if now <= event.date <= buffer_end:
                    upcoming_news.append(event)
        
        is_safe = len(upcoming_news) == 0
        return is_safe, upcoming_news


# ============================================================================
# MARKET HOURS VALIDATOR
# ============================================================================

class MarketHoursValidator:
    """Validate trading hours and identify sessions"""
    
    @staticmethod
    def is_forex_open(dt: datetime = None) -> Tuple[bool, str]:
        """Check Forex market hours"""
        if dt is None:
            dt = datetime.now(pytz.UTC)
        
        weekday = dt.weekday()
        hour = dt.hour
        
        if weekday == 5:
            return False, "Forex closed (Saturday)"
        
        if weekday == 6:
            if hour < 22:
                return False, "Forex closed (Sunday before 22:00 UTC)"
            return True, "Forex open (Sunday evening)"
        
        if weekday == 4 and hour >= 22:
            return False, "Forex closed (Friday after 22:00 UTC)"
        
        return True, "Forex market open"
    
    @staticmethod
    def is_crypto_open(dt: datetime = None) -> Tuple[bool, str]:
        """Crypto markets are 24/7"""
        return True, "Crypto market open (24/7)"
    
    @staticmethod
    def get_market_status(symbol: str) -> Tuple[bool, str]:
        """Get market status for specific symbol"""
        if symbol == 'XAUUSD':
            return MarketHoursValidator.is_forex_open()
        elif symbol == 'BTCUSD':
            return MarketHoursValidator.is_crypto_open()
        return True, "Market status unknown"
    
    @staticmethod
    def get_trading_session(dt: datetime = None) -> Tuple[str, float]:
        """Get current trading session and liquidity score"""
        if dt is None:
            dt = datetime.now(pytz.UTC)
        
        hour = dt.hour
        
        if 0 <= hour < 8:
            return "Asian", 40.0
        
        if 8 <= hour < 13:
            return "London", 80.0
        
        if 13 <= hour < 16:
            return "London/NY Overlap", 100.0
        
        if 16 <= hour < 21:
            return "New York", 85.0
        
        return "After Hours", 30.0


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

class TechnicalIndicators:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX, +DI, -DI"""
        tr = TechnicalIndicators.calculate_atr(df, period=1)
        
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to DataFrame"""
        df['ema_9'] = TechnicalIndicators.calculate_ema(df['close'], 9)
        df['ema_21'] = TechnicalIndicators.calculate_ema(df['close'], 21)
        df['ema_50'] = TechnicalIndicators.calculate_ema(df['close'], 50)
        df['ema_200'] = TechnicalIndicators.calculate_ema(df['close'], 200)
        
        df['rsi'] = TechnicalIndicators.calculate_rsi(df['close'], 14)
        df['atr'] = TechnicalIndicators.calculate_atr(df, 14)
        df['adx'], df['plus_di'], df['minus_di'] = TechnicalIndicators.calculate_adx(df, 14)
        
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df


# ============================================================================
# MARKET STRUCTURE ANALYZER
# ============================================================================

class MarketStructureAnalyzer:
    """Analyze market structure and identify trends"""
    
    @staticmethod
    def determine_trend_direction(df: pd.DataFrame) -> TrendDirection:
        """Determine overall trend direction"""
        last = df.iloc[-1]
        
        ema_score = 0
        if last['ema_9'] > last['ema_21'] > last['ema_50']:
            ema_score = 3
        elif last['ema_9'] > last['ema_21']:
            ema_score = 2
        elif last['ema_9'] < last['ema_21'] < last['ema_50']:
            ema_score = -3
        elif last['ema_9'] < last['ema_21']:
            ema_score = -2
        
        if ema_score >= 2:
            return TrendDirection.STRONG_BULLISH if ema_score == 3 else TrendDirection.BULLISH
        elif ema_score <= -2:
            return TrendDirection.STRONG_BEARISH if ema_score == -3 else TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL


# ============================================================================
# RISK CALCULATOR
# ============================================================================

class RiskCalculator:
    """Calculate position sizing and risk management"""
    
    @staticmethod
    def calculate_position_size(balance: float, risk_percent: float, 
                                 sl_pips: float, symbol: str) -> Tuple[float, float]:
        """Calculate position size"""
        risk_amount = balance * (risk_percent / 100.0)
        pip_value = 10.0
        
        if sl_pips <= 0:
            return 0.01, risk_amount
        
        lot_size = risk_amount / (sl_pips * pip_value)
        
        if balance < 100:
            lot_size = max(0.01, min(lot_size, 0.05))
        elif balance < 500:
            lot_size = max(0.01, min(lot_size, 0.2))
        elif balance < 1000:
            lot_size = max(0.01, min(lot_size, 0.5))
        else:
            lot_size = max(0.01, min(lot_size, 2.0))
        
        lot_size = round(lot_size, 2)
        actual_risk = lot_size * sl_pips * pip_value
        
        return lot_size, actual_risk
    
    @staticmethod
    def calculate_trade_levels(current_price: float, atr: float, action: str,
                                symbol: str, balance: float, signal_type: SignalType) -> Dict:
        """Calculate trade levels based on signal type"""
        if signal_type == SignalType.SCALP:
            sl_multiplier = 1.0
            tp1_multiplier = 1.5
            tp2_multiplier = 2.5
            tp3_multiplier = 3.5
        elif signal_type == SignalType.DAY_TRADE:
            sl_multiplier = 1.5
            tp1_multiplier = 2.5
            tp2_multiplier = 4.0
            tp3_multiplier = 6.0
        else:  # SWING
            sl_multiplier = 2.0
            tp1_multiplier = 3.0
            tp2_multiplier = 5.0
            tp3_multiplier = 7.5
        
        if action == "BUY":
            entry = current_price
            stop_loss = entry - (sl_multiplier * atr)
            tp1 = entry + (tp1_multiplier * atr)
            tp2 = entry + (tp2_multiplier * atr)
            tp3 = entry + (tp3_multiplier * atr)
            breakeven = entry + (0.1 * (entry - stop_loss))
        else:
            entry = current_price
            stop_loss = entry + (sl_multiplier * atr)
            tp1 = entry - (tp1_multiplier * atr)
            tp2 = entry - (tp2_multiplier * atr)
            tp3 = entry - (tp3_multiplier * atr)
            breakeven = entry - (0.1 * (stop_loss - entry))
        
        pip_size = 0.01 if symbol == 'XAUUSD' else 1.0
        sl_pips = abs(entry - stop_loss) / pip_size
        tp1_pips = abs(tp1 - entry) / pip_size
        tp2_pips = abs(tp2 - entry) / pip_size
        tp3_pips = abs(tp3 - entry) / pip_size
        
        rr_ratio = tp3_pips / sl_pips if sl_pips > 0 else 0
        
        return {
            'entry': entry,
            'stop_loss': stop_loss,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'breakeven': breakeven,
            'sl_pips': sl_pips,
            'tp1_pips': tp1_pips,
            'tp2_pips': tp2_pips,
            'tp3_pips': tp3_pips,
            'rr_ratio': rr_ratio
        }


# ============================================================================
# SLIPPAGE & EXECUTION MODEL
# ============================================================================

class ExecutionModel:
    """Models realistic trade execution with slippage and spread"""
    
    @staticmethod
    def apply_slippage(price: float, action: str, symbol: str, 
                       signal_type, liquidity: float) -> Tuple[float, float]:
        """Apply realistic slippage based on signal type and liquidity"""
        if signal_type == SignalType.SCALP:
            base_slippage = 2.0
        elif signal_type == SignalType.DAY_TRADE:
            base_slippage = 1.5
        else:
            base_slippage = 1.0
        
        liquidity_factor = 1.0
        if liquidity < 60:
            liquidity_factor = 1.5
        elif liquidity >= 100:
            liquidity_factor = 0.7
        
        total_slippage_pips = base_slippage * liquidity_factor
        
        pip_size = 0.01 if symbol == 'XAUUSD' else 1.0
        slippage_price = total_slippage_pips * pip_size
        
        if action == "BUY":
            executed_price = price + slippage_price
        else:
            executed_price = price - slippage_price
        
        return executed_price, total_slippage_pips
    
    @staticmethod
    def get_spread(symbol: str, liquidity: float) -> float:
        """Get current spread in pips"""
        base_spread = {
            'XAUUSD': 3.0,
            'BTCUSD': 5.0
        }.get(symbol, 2.0)
        
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
            if last_entry['ema_9'] > last_entry['ema_21'] > last_entry['ema_50']:
                score += 15.0
                reasons.append("✅ Perfect EMA alignment (entry)")
            elif last_entry['ema_9'] > last_entry['ema_21']:
                score += 10.0
                reasons.append("EMA bullish (entry)")
            else:
                return 0.0, ["❌ No bullish EMA trend"]
            
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
        """Calculate confidence using only 4 core indicators"""
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
        
        # Market hours check
        is_open, market_status = MarketHoursValidator.get_market_status(symbol)
        if not is_open:
            logger.info(f"⚪ {market_status}")
            return None
        
        logger.info(f"✅ {market_status}")
        
        # Session check
        session, liquidity = MarketHoursValidator.get_trading_session()
        logger.info(f"📍 {session} session (liquidity: {liquidity:.0f}%)")
        
        if signal_type == SignalType.SCALP and liquidity < 60:
            logger.info("❌ Liquidity too low for scalping")
            return None
        
        # News check
        if self.enable_news_filter:
            is_safe, upcoming_news = self.news_monitor.check_news_safety(symbol, signal_type)
            if not is_safe:
                logger.warning(f"🚨 HIGH-IMPACT NEWS - BLOCKING")
                news_id = f"{upcoming_news[0].title}_{upcoming_news[0].date.isoformat()}"
                if news_id not in self.news_alerts_sent:
                    self.notifier.send_news_alert(upcoming_news)
                    self.news_alerts_sent.add(news_id)
                return None
            logger.info("✅ No high-impact news")
        
        # Timeframe selection
        if signal_type == SignalType.SCALP:
            entry_tf, trend_tf = '5m', '15m'
            entry_count, trend_count = 200, 200
        elif signal_type == SignalType.DAY_TRADE:
            entry_tf, trend_tf = '15m', '1h'
            entry_count, trend_count = 300, 500
        else:
            entry_tf, trend_tf = '1h', '4h'
            entry_count, trend_count = 500, 200
        
        logger.info(f"📥 Fetching {entry_tf} and {trend_tf} data...")
        df_entry = self.data_fetcher.get_historical_data(symbol, entry_tf, entry_count)
        df_trend = self.data_fetcher.get_historical_data(symbol, trend_tf, trend_count)
        
        if df_entry is None or df_trend is None:
            logger.error("❌ Failed to fetch data")
            return None
        
        # Calculate indicators
        logger.info("🔧 Calculating indicators...")
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
        
        logger.info(f"✅ SIGNAL QUALIFIED ({confidence:.1f}%)")
        
        # Calculate trade levels
        last_candle = df_entry.iloc[-1]
        current_price = last_candle['close']
        atr = last_candle['atr']
        
        levels = RiskCalculator.calculate_trade_levels(
            current_price, atr, action, symbol, self.account_balance, signal_type
        )
        
        # Apply slippage
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
        
        # Determine quality
        quality = SignalQuality.EXCELLENT if confidence >= 90 else (
            SignalQuality.STRONG if confidence >= 80 else SignalQuality.GOOD
        )
        
        # Create signal
        signal_id = f"{symbol}_{signal_type.value}_{int(datetime.now().timestamp())}"
        timestamp = datetime.now(pytz.UTC)
        
        validity_hours = {
            SignalType.SCALP: 1,
            SignalType.DAY_TRADE: 2,
            SignalType.SWING: 4
        }[signal_type]
        valid_until = timestamp + timedelta(hours=validity_hours)
        
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
            timestamp=timestamp,
            valid_until=valid_until
        )
        
        # Log to journal
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


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

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
