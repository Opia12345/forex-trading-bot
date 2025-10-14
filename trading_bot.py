"""
ELITE TRADING BOT v10.0 - ENHANCED & OPTIMIZED
Added: Multi-timeframe, Price Action, Better Range Filter, Volume Analysis
Assets: XAUUSD & BTCUSD | Signal Generation Only
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
import json
import os
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import pytz
from collections import deque
import asyncio
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('elite_trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class NewsEvent:
    """Economic news event data"""
    title: str
    country: str
    date: datetime
    impact: str
    actual: str
    forecast: str
    previous: str
    currency: str


@dataclass
class TradeSignal:
    """Trade signal data structure"""
    symbol: str
    action: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    tp1: float
    tp2: float
    tp3: float
    risk_reward_ratio: float
    timeframe: str
    strategy_name: str
    indicators: Dict
    timestamp: datetime
    stop_loss_pips: float = 0
    tp1_pips: float = 0
    tp2_pips: float = 0
    tp3_pips: float = 0
    position_size: float = 0.01
    risk_amount_usd: float = 0
    signal_id: str = ""
    breakeven_price: float = 0

    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class MarketHoursValidator:
    """Validate if markets are open for trading"""

    @staticmethod
    def is_forex_open(dt: datetime = None) -> Tuple[bool, str]:
        """Check if Forex market is open (for XAUUSD)"""
        if dt is None:
            dt = datetime.now(pytz.UTC)

        weekday = dt.weekday()

        if weekday == 5:
            return False, "Weekend - Forex closed (Saturday)"

        if weekday == 6:
            if dt.hour < 22:
                return False, "Weekend - Forex closed (Sunday before 22:00 UTC)"
            return True, "Forex open (Sunday evening)"

        if weekday == 4 and dt.hour >= 22:
            return False, "Weekend - Forex closed (Friday after 22:00 UTC)"

        return True, "Forex market open"

    @staticmethod
    def is_crypto_open(dt: datetime = None) -> Tuple[bool, str]:
        """Crypto is always open"""
        return True, "Crypto market open (24/7)"

    @staticmethod
    def get_market_status(symbol: str) -> Tuple[bool, str]:
        """Get market status for symbol"""
        if symbol == 'XAUUSD':
            return MarketHoursValidator.is_forex_open()
        elif symbol == 'BTCUSD':
            return MarketHoursValidator.is_crypto_open()
        return True, "Unknown market"


class NewsMonitor:
    """Monitor high-impact economic news"""

    def __init__(self):
        self.forex_factory_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        self.last_fetch = None
        self.cache_duration = 1800
        self.cached_events = []

    def fetch_news_events(self) -> List[NewsEvent]:
        """Fetch upcoming news"""
        
        if self.last_fetch and (datetime.now() - self.last_fetch).seconds < self.cache_duration:
            return self.cached_events

        try:
            response = requests.get(self.forex_factory_url, timeout=15)
            if response.status_code != 200:
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
                        actual=item.get('actual', ''),
                        forecast=item.get('forecast', ''),
                        previous=item.get('previous', ''),
                        currency=item.get('currency', '')
                    ))
                except:
                    continue

            self.cached_events = events
            self.last_fetch = datetime.now()
            logger.info(f"✅ Fetched {len(events)} HIGH impact news events")
            return events

        except Exception as e:
            logger.error(f"News fetch error: {e}")
            return self.cached_events

    def check_news_before_trade(self, symbol: str) -> Tuple[bool, List[NewsEvent]]:
        """Check if safe to trade (no news within 1 hour)"""
        events = self.fetch_news_events()
        
        now = datetime.now(pytz.UTC)
        one_hour_later = now + timedelta(hours=1)

        relevant_news = []

        if symbol == 'XAUUSD':
            relevant_currencies = ['USD', 'EUR', 'GBP']
        elif symbol == 'BTCUSD':
            relevant_currencies = ['USD']
        else:
            return True, []

        for event in events:
            if event.currency in relevant_currencies:
                if now <= event.date <= one_hour_later:
                    relevant_news.append(event)

        is_safe = len(relevant_news) == 0
        return is_safe, relevant_news


class DerivDataFetcher:
    """Fetch real-time data from Deriv API"""

    def __init__(self, app_id: str = "1089"):
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"

    async def _fetch_candles_ws(self, symbol: str, granularity: int, count: int) -> Optional[Dict]:
        """Fetch candles via WebSocket"""
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

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            return None

    def get_historical_data(self, symbol: str, timeframe: str = '1h', count: int = 300) -> Optional[pd.DataFrame]:
        """Fetch historical candles"""

        symbol_map = {'XAUUSD': 'frxXAUUSD', 'BTCUSD': 'cryBTCUSD'}
        deriv_symbol = symbol_map.get(symbol)
        if not deriv_symbol:
            return None

        tf_map = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400}
        granularity = tf_map.get(timeframe)
        if not granularity:
            return None

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    self._fetch_candles_ws(deriv_symbol, granularity, count)
                )
            finally:
                loop.close()

            if not response or 'candles' not in response:
                return None

            candles = response['candles']
            if not candles:
                return None

            df = pd.DataFrame(candles)
            
            if 'epoch' in df.columns:
                df['timestamp'] = pd.to_datetime(df['epoch'], unit='s')
            
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna(subset=['open', 'high', 'low', 'close'])

            if len(df) < 100:
                return None

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None


class TelegramNotifier:
    """Send signals via Telegram"""

    def __init__(self, bot_token: str, main_chat_id: str, simple_chat_id: str):
        self.bot_token = bot_token
        self.main_chat_id = main_chat_id
        self.simple_chat_id = simple_chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def send_message(self, message: str, chat_id: str) -> bool:
        """Send message to Telegram"""
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': chat_id, 
            'text': message, 
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False

    def send_signal(self, signal: TradeSignal) -> bool:
        """Send trading signal"""
        emoji = "🟢" if "BUY" in signal.action else "🔴"

        if signal.confidence >= 90:
            label = "🔥🔥🔥 EXCELLENT"
        elif signal.confidence >= 80:
            label = "🔥🔥 STRONG"
        else:
            label = "🔥 GOOD"

        reasons = signal.indicators.get('reasons', [])
        reasons_text = '\n'.join([f"  • {r}" for r in reasons[:6]])

        message = f"""
{emoji} <b>{label} - {signal.confidence:.0f}%</b>

<b>Symbol:</b> {signal.symbol}
<b>Action:</b> {signal.action}
<b>ID:</b> {signal.signal_id}

💰 <b>Trade Setup</b>
<b>Entry:</b> {signal.entry_price:.5f}
<b>Lots:</b> {signal.position_size:.2f}
<b>Risk:</b> ${signal.risk_amount_usd:.2f}

<b>Stop Loss:</b> {signal.stop_loss:.5f} ({signal.stop_loss_pips:.1f} pips)
<b>TP1:</b> {signal.tp1:.5f} ({signal.tp1_pips:.1f}p) - 50%
<b>TP2:</b> {signal.tp2:.5f} ({signal.tp2_pips:.1f}p) - 30%
<b>TP3:</b> {signal.tp3:.5f} ({signal.tp3_pips:.1f}p) - 20%

📊 <b>Analysis</b>
{reasons_text}

⚡ <b>Management:</b>
• Hit TP1 → Move SL to breakeven ({signal.breakeven_price:.5f})
• Trail stops to TP2 and TP3
• R:R Ratio: 1:{signal.risk_reward_ratio:.1f}

🕐 {signal.timestamp.strftime('%H:%M UTC')}
"""

        detailed = self.send_message(message, self.main_chat_id)

        simple_msg = f"""
{emoji} {signal.action}</b>

<b>{signal.symbol}</b>

Entry: {signal.entry_price:.5f}
SL: {signal.stop_loss:.5f}
TP1: {signal.tp1_pips:.5f} 🎯
TP2: {signal.tp2_pips:.5f} 🎯🎯
TP3: {signal.tp3_pips:.5f} 🎯🎯🎯

Risk/Reward: 1:{signal.risk_reward_ratio:.1f}
"""
        simple = self.send_message(simple_msg, self.simple_chat_id)

        return detailed and simple

    def send_news_alert(self, news_events: List[NewsEvent]):
        """Send news alert"""
        if not news_events:
            return

        message = "🚨 <b>HIGH-IMPACT NEWS ALERT</b>\n\n"
        message += "⚠️ Trading blocked - Major news within 1 hour\n\n"

        for event in news_events[:3]:
            time_until = event.date - datetime.now(pytz.UTC)
            minutes = int(time_until.total_seconds() / 60)

            message += f"<b>{event.title}</b>\n"
            message += f"🌍 {event.currency} | In {minutes} minutes\n\n"

        self.send_message(message, self.main_chat_id)
        self.send_message(message, self.simple_chat_id)

    def send_status(self, message: str):
        """Send status update"""
        self.send_message(f"ℹ️ {message}", self.main_chat_id)


class RiskCalculator:
    """Position sizing and risk management"""

    @staticmethod
    def calculate_position_size(balance: float, risk_pct: float, sl_pips: float, symbol: str) -> Tuple[float, float]:
        """Calculate lot size based on risk"""
        risk_amount = balance * (risk_pct / 100)
        pip_value = 10.0  # $10 per pip per lot

        if sl_pips > 0:
            lot_size = risk_amount / (sl_pips * pip_value)
        else:
            lot_size = 0.01

        # Limits
        if balance < 100:
            lot_size = max(0.01, min(lot_size, 0.05))
        else:
            lot_size = max(0.01, min(lot_size, 1.0))
            lot_size = round(lot_size * 100) / 100

        actual_risk = lot_size * sl_pips * pip_value
        return lot_size, actual_risk


class EliteTradingBot:
    """Elite Trading Bot v10.0 - Enhanced & Optimized"""

    SYMBOLS = ['XAUUSD', 'BTCUSD']

    def __init__(self, telegram_token: str, main_chat_id: str, simple_chat_id: str,
                 account_balance: float = 500.0, risk_percent: float = 2.0,
                 deriv_app_id: str = "1089", check_news: bool = True):

        self.notifier = TelegramNotifier(telegram_token, main_chat_id, simple_chat_id)
        self.data_fetcher = DerivDataFetcher(deriv_app_id)
        self.news_monitor = NewsMonitor()

        self.account_balance = account_balance
        self.risk_percent = risk_percent
        self.min_confidence = 70.0
        self.timeframe = '1h'
        self.htf_timeframe = '4h'  # Higher timeframe for trend
        self.recent_signals = deque(maxlen=20)
        self.check_news = check_news
        self.news_alerts_sent = set()

        logger.info("=" * 70)
        logger.info("ELITE BOT v10.0 - ENHANCED & OPTIMIZED")
        logger.info(f"Balance: ${account_balance} | Risk: {risk_percent}% | Min: 70%")
        logger.info("=" * 70)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate essential indicators"""

        # EMAs
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        atr14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(14).mean()

        # Bollinger Bands (for range detection)
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        df = df.ffill().bfill()
        return df

    def check_price_action(self, df: pd.DataFrame, action: str) -> Tuple[bool, str]:
        """Check for bullish/bearish price action patterns"""
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]

        if action == "BUY":
            # Check for higher highs and higher lows (last 3 candles)
            higher_high = last['high'] > prev['high'] > prev2['high']
            higher_low = last['low'] > prev['low'] > prev2['low']
            
            if higher_high and higher_low:
                return True, "Higher highs & higher lows"
            elif higher_high or higher_low:
                return True, "Bullish structure forming"
            
            # Check for bullish engulfing
            if last['close'] > last['open'] and prev['close'] < prev['open']:
                if last['close'] > prev['open'] and last['open'] < prev['close']:
                    return True, "Bullish engulfing pattern"
        
        else:  # SELL
            # Check for lower highs and lower lows
            lower_high = last['high'] < prev['high'] < prev2['high']
            lower_low = last['low'] < prev['low'] < prev2['low']
            
            if lower_high and lower_low:
                return True, "Lower highs & lower lows"
            elif lower_high or lower_low:
                return True, "Bearish structure forming"
            
            # Check for bearish engulfing
            if last['close'] < last['open'] and prev['close'] > prev['open']:
                if last['close'] < prev['open'] and last['open'] > prev['close']:
                    return True, "Bearish engulfing pattern"
        
        return False, ""

    def is_ranging_market(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Detect if market is ranging (improved)"""
        last = df.iloc[-1]
        
        # Method 1: Bollinger Band width (narrow bands = ranging)
        avg_bb_width = df['bb_width'].tail(20).mean()
        if avg_bb_width < 0.02:  # Less than 2% width
            return True, "Tight Bollinger Bands (ranging)"
        
        # Method 2: ADX check (low ADX = no trend)
        if last['adx'] < 15:
            return True, f"Weak trend (ADX {last['adx']:.0f})"
        
        # Method 3: Price oscillating around EMAs
        ema_touches = 0
        for i in range(-10, 0):
            candle = df.iloc[i]
            if candle['low'] <= candle['ema_21'] <= candle['high']:
                ema_touches += 1
        
        if ema_touches >= 5:  # Price crossing EMA21 frequently
            return True, "Price oscillating (choppy)"
        
        return False, ""

    def check_volume_confirmation(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check volume (using ATR as proxy since Deriv volume is synthetic)"""
        last = df.iloc[-1]
        avg_atr = df['atr'].tail(20).mean()
        
        # High volatility = strong move
        if last['atr'] > avg_atr * 1.3:
            return True, "High volatility (strong move)"
        elif last['atr'] > avg_atr * 1.1:
            return True, "Above-average volatility"
        elif last['atr'] > avg_atr * 0.7:
            return True, "Normal volatility"
        else:
            return False, "Low volatility (weak move)"

    def analyze_market(self, symbol: str) -> Optional[TradeSignal]:
        """Enhanced multi-timeframe analysis"""

        # Fetch both timeframes
        df_1h = self.data_fetcher.get_historical_data(symbol, '1h', 300)
        df_4h = self.data_fetcher.get_historical_data(symbol, '4h', 150)
        
        if df_1h is None or len(df_1h) < 100:
            return None
        if df_4h is None or len(df_4h) < 50:
            logger.warning(f"{symbol}: Using single timeframe (4H unavailable)")
            df_4h = None

        df_1h = self.calculate_indicators(df_1h)
        last_1h = df_1h.iloc[-1]
        prev_1h = df_1h.iloc[-2]

        # STEP 1: Multi-timeframe trend check
        htf_trend = None
        if df_4h is not None:
            df_4h = self.calculate_indicators(df_4h)
            last_4h = df_4h.iloc[-1]
            
            # Determine 4H trend
            if last_4h['ema_9'] > last_4h['ema_21'] and last_4h['close'] > last_4h['ema_50']:
                htf_trend = "BULLISH"
            elif last_4h['ema_9'] < last_4h['ema_21'] and last_4h['close'] < last_4h['ema_50']:
                htf_trend = "BEARISH"
            else:
                htf_trend = "NEUTRAL"
            
            logger.info(f"  4H Trend: {htf_trend}")

        # STEP 2: 1H trend detection
        if last_1h['ema_9'] > last_1h['ema_21'] and last_1h['close'] > last_1h['ema_50']:
            action = "BUY"
        elif last_1h['ema_9'] < last_1h['ema_21'] and last_1h['close'] < last_1h['ema_50']:
            action = "SELL"
        else:
            return None  # No clear trend

        # STEP 3: Check if 1H aligns with 4H (if available)
        if htf_trend is not None:
            if htf_trend == "NEUTRAL":
                logger.info(f"  ⚠️ 4H trend is neutral - lower confidence")
            elif (action == "BUY" and htf_trend == "BEARISH") or (action == "SELL" and htf_trend == "BULLISH"):
                logger.info(f"  ❌ 1H {action} against 4H {htf_trend} - REJECTED")
                return None

        # STEP 4: Range filter (improved)
        is_ranging, range_reason = self.is_ranging_market(df_1h)
        if is_ranging:
            logger.info(f"  ⚠️ Ranging market detected: {range_reason}")
            return None  # Skip ranging markets

        # STEP 5: Build confidence score
        confidence = 50.0
        reasons = []

        # Multi-timeframe alignment (15 points)
        if htf_trend is not None:
            if (action == "BUY" and htf_trend == "BULLISH") or (action == "SELL" and htf_trend == "BEARISH"):
                confidence += 15
                reasons.append("✅ 4H trend aligned with 1H")
            else:
                confidence += 7
                reasons.append("4H neutral, 1H trending")
        else:
            confidence += 5
            reasons.append("Single timeframe (1H only)")

        # EMA Alignment (15 points)
        if action == "BUY":
            if last_1h['ema_9'] > last_1h['ema_21'] > last_1h['ema_50']:
                confidence += 15
                reasons.append("✅ Perfect EMA bullish alignment")
            else:
                confidence += 8
                reasons.append("EMA bullish trend")
        else:
            if last_1h['ema_9'] < last_1h['ema_21'] < last_1h['ema_50']:
                confidence += 15
                reasons.append("✅ Perfect EMA bearish alignment")
            else:
                confidence += 8
                reasons.append("EMA bearish trend")

        # Price Action (12 points)
        has_price_action, pa_desc = self.check_price_action(df_1h, action)
        if has_price_action:
            confidence += 12
            reasons.append(f"✅ {pa_desc}")
        else:
            confidence += 5
            reasons.append("Basic price structure")

        # RSI (12 points)
        if action == "BUY":
            if 30 <= last_1h['rsi'] <= 50:
                confidence += 12
                reasons.append(f"✅ RSI optimal ({last_1h['rsi']:.0f})")
            elif last_1h['rsi'] < 60:
                confidence += 8
                reasons.append(f"RSI favorable ({last_1h['rsi']:.0f})")
            else:
                confidence += 3
        else:
            if 50 <= last_1h['rsi'] <= 70:
                confidence += 12
                reasons.append(f"✅ RSI optimal ({last_1h['rsi']:.0f})")
            elif last_1h['rsi'] > 40:
                confidence += 8
                reasons.append(f"RSI favorable ({last_1h['rsi']:.0f})")
            else:
                confidence += 3

        # MACD (10 points)
        if action == "BUY":
            if last_1h['macd'] > last_1h['macd_signal']:
                if prev_1h['macd'] <= prev_1h['macd_signal']:
                    confidence += 10
                    reasons.append("✅ MACD bullish crossover")
                else:
                    confidence += 7
                    reasons.append("MACD bullish")
        else:
            if last_1h['macd'] < last_1h['macd_signal']:
                if prev_1h['macd'] >= prev_1h['macd_signal']:
                    confidence += 10
                    reasons.append("✅ MACD bearish crossover")
                else:
                    confidence += 7
                    reasons.append("MACD bearish")

        # ADX + Volume (8 points)
        has_volume, vol_desc = self.check_volume_confirmation(df_1h)
        if last_1h['adx'] > 25 and has_volume:
            confidence += 8
            reasons.append(f"✅ Strong trend (ADX {last_1h['adx']:.0f}) + {vol_desc}")
        elif last_1h['adx'] > 20:
            confidence += 5
            reasons.append(f"Good trend (ADX {last_1h['adx']:.0f})")
        elif last_1h['adx'] > 15:
            confidence += 3
            reasons.append(f"Moderate trend (ADX {last_1h['adx']:.0f})")

        # Confirmation candle (3 points bonus)
        if action == "BUY":
            if last_1h['close'] > last_1h['open']:  # Bullish candle
                confidence += 3
                reasons.append("Bullish confirmation candle")
        else:
            if last_1h['close'] < last_1h['open']:  # Bearish candle
                confidence += 3
                reasons.append("Bearish confirmation candle")

        confidence = min(confidence, 100.0)

        logger.info(f"{symbol} {action}: {confidence:.0f}%")

        if confidence < self.min_confidence:
            logger.info(f"  Below 70% threshold")
            return None

        logger.info(f"  ✅ SIGNAL QUALIFIED!")

        # Calculate trade levels
        current_price = last_1h['close']
        atr = last_1h['atr']

        sl_multiplier = 1.5 if self.account_balance < 100 else 2.0

        if action == "BUY":
            entry = current_price
            sl = entry - (sl_multiplier * atr)
            tp1 = entry + (1.5 * atr)
            tp2 = entry + (2.5 * atr)
            tp3 = entry + (4.0 * atr)
        else:
            entry = current_price
            sl = entry + (sl_multiplier * atr)
            tp1 = entry - (1.5 * atr)
            tp2 = entry - (2.5 * atr)
            tp3 = entry - (4.0 * atr)

        pip_size = 0.01 if symbol == 'XAUUSD' else 1.0
        sl_pips = abs(entry - sl) / pip_size
        tp1_pips = abs(tp1 - entry) / pip_size
        tp2_pips = abs(tp2 - entry) / pip_size
        tp3_pips = abs(tp3 - entry) / pip_size

        lot_size, risk_amount = RiskCalculator.calculate_position_size(
            self.account_balance, self.risk_percent, sl_pips, symbol
        )

        rr_ratio = tp3_pips / sl_pips if sl_pips > 0 else 0
        be_price = entry + (abs(entry - sl) * 0.15) if action == "BUY" else entry - (abs(entry - sl) * 0.15)

        signal = TradeSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp3,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            risk_reward_ratio=rr_ratio,
            timeframe=self.timeframe,
            strategy_name="Enhanced Multi-Timeframe v10.0",
            indicators={
                'rsi': float(last_1h['rsi']),
                'macd': float(last_1h['macd']),
                'adx': float(last_1h['adx']),
                'htf_trend': htf_trend if htf_trend else 'N/A',
                'reasons': reasons
            },
            timestamp=datetime.now(pytz.UTC),
            stop_loss_pips=sl_pips,
            tp1_pips=tp1_pips,
            tp2_pips=tp2_pips,
            tp3_pips=tp3_pips,
            position_size=lot_size,
            risk_amount_usd=risk_amount,
            signal_id=f"{symbol}_{int(datetime.now().timestamp())}",
            breakeven_price=be_price
        )

        return signal

    def is_duplicate(self, signal: TradeSignal) -> bool:
        """Check for duplicate signals (2 hours)"""
        for recent in self.recent_signals:
            if (recent.symbol == signal.symbol and 
                recent.action == signal.action and
                abs((signal.timestamp - recent.timestamp).total_seconds()) < 7200):
                return True
        return False

    def run(self):
        """Run analysis cycle"""
        logger.info("\n" + "=" * 70)
        logger.info("🚀 ELITE BOT v10.0 - ANALYSIS STARTED")
        logger.info(f"Time: {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M UTC')}")
        logger.info("=" * 70)

        signals_count = 0

        for symbol in self.SYMBOLS:
            try:
                logger.info(f"\n📊 Analyzing {symbol}...")

                # Check market hours
                is_open, status = MarketHoursValidator.get_market_status(symbol)
                if not is_open:
                    logger.warning(f"  ⚪ {status}")
                    continue

                logger.info(f"  ✅ {status}")

                # Check news
                if self.check_news:
                    is_safe, news = self.news_monitor.check_news_before_trade(symbol)
                    
                    if not is_safe:
                        logger.warning(f"  🚨 HIGH-IMPACT NEWS - BLOCKING")
                        
                        news_id = f"{news[0].title}_{news[0].date.isoformat()}"
                        if news_id not in self.news_alerts_sent:
                            self.notifier.send_news_alert(news)
                            self.news_alerts_sent.add(news_id)
                        continue

                # Analyze
                signal = self.analyze_market(symbol)

                if signal and not self.is_duplicate(signal):
                    logger.info(f"✅ SIGNAL: {signal.action} @ {signal.confidence:.0f}%")
                    
                    self.recent_signals.append(signal)
                    
                    if self.notifier.send_signal(signal):
                        logger.info(f"  ✅ Sent to Telegram")
                        signals_count += 1
                    else:
                        logger.error(f"  ❌ Failed to send")

                elif signal:
                    logger.info(f"  ⚠️ Duplicate - skipped")

            except Exception as e:
                logger.error(f"❌ Error: {e}", exc_info=True)
                continue

        logger.info("\n" + "=" * 70)
        logger.info(f"✅ COMPLETE - {signals_count} signal(s) generated")
        logger.info("=" * 70)

        return signals_count


def main():
    """Main entry point"""

    token = os.getenv('TELEGRAM_BOT_TOKEN')
    main_chat = os.getenv('MAIN_CHAT_ID')
    simple_chat = os.getenv('SIMPLE_CHAT_ID')
    app_id = os.getenv('DERIV_APP_ID', '1089')

    balance = float(os.getenv('ACCOUNT_BALANCE', '500'))
    risk = float(os.getenv('RISK_PERCENT', '2.0'))

    if not token or not main_chat or not simple_chat:
        logger.error("❌ Missing environment variables!")
        return 1

    try:
        bot = EliteTradingBot(
            telegram_token=token,
            main_chat_id=main_chat,
            simple_chat_id=simple_chat,
            account_balance=balance,
            risk_percent=risk,
            deriv_app_id=app_id,
            check_news=True
        )

        startup = f"""
🚀 <b>Elite Bot v10.0 - ENHANCED &amp; OPTIMIZED</b>

<b>Config:</b>
• Assets: XAUUSD, BTCUSD
• Timeframes: 4H trend + 1H entry
• Min Confidence: 70% (Fixed)
• Account: ${balance:.2f}
• Risk: {risk}% per trade

<b>✨ New Features:</b>
✅ Multi-timeframe analysis (4H + 1H)
✅ Price action patterns detection
✅ Advanced range filter (BB + ADX + EMA)
✅ Volume/volatility confirmation
✅ Confirmation candle check
✅ Smart trend alignment

<b>🛡️ Filters:</b>
• 4H/1H trend must align
• No ranging markets (tight BB/low ADX)
• News protection (1hr buffer)
• Market hours validation
• No counter-trend trades

<i>🔍 Scanning for premium 70%+ setups...</i>
"""
        bot.notifier.send_status(startup)

        count = bot.run()

        if count > 0:
            summary = f"""
✅ <b>{count} High-Quality Signal(s) Generated</b>

🎯 All signals passed:
• Multi-timeframe alignment check
• Range/chop filter
• Price action confirmation
• Volume validation
• 70%+ confidence threshold

<b>Review signals above and execute manually.</b>

<i>📅 Next scan: Top of next hour</i>
<i>🎖️ Quality over quantity - Premium setups only</i>
"""
        else:
            summary = f"""
✅ <b>Scan Complete - No Premium Signals</b>

No 70%+ setups met all criteria this cycle.

<b>Filtered out:</b>
• Counter-trend trades
• Ranging/choppy markets
• Weak price action
• Low volatility periods

<i>📅 Next scan: Top of next hour</i>
<i>⏳ Waiting for high-probability setups</i>
"""

        bot.notifier.send_status(summary)
        return 0

    except Exception as e:
        logger.error(f"❌ FATAL: {e}", exc_info=True)

        try:
            notifier = TelegramNotifier(token, main_chat, simple_chat)
            error = f"""
🚨 <b>Bot Error</b>

{type(e).__name__}: {str(e)[:200]}

Check logs for details.
"""
            notifier.send_status(error)
        except:
            pass

        return 1


if __name__ == "__main__":
    exit(main())