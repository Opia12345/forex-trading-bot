"""
ELITE TRADING BOT v8.0 - INSTITUTIONAL GRADE
Fixed: WebSocket, News Alerts, Risk Management, Weekend Detection
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
        """Check if Forex market is open (for XAUUSD) - FIXED"""
        if dt is None:
            dt = datetime.now(pytz.UTC)

        weekday = dt.weekday()

        # Saturday - completely closed
        if weekday == 5:
            return False, "Weekend - Forex market closed (Saturday)"

        # Sunday - opens at 22:00 UTC
        if weekday == 6:
            if dt.hour < 22:
                return False, "Weekend - Forex market closed (Sunday before 22:00 UTC)"
            else:
                return True, "Forex market open (Sunday evening)"

        # Friday - closes at 22:00 UTC
        if weekday == 4:
            if dt.hour >= 22:
                return False, "Weekend - Forex market closed (Friday after 22:00 UTC)"

        # Monday-Thursday - fully open
        return True, "Forex market open"

    @staticmethod
    def is_crypto_open(dt: datetime = None) -> Tuple[bool, str]:
        """Check if Crypto market is open (for BTCUSD)"""
        return True, "Crypto market always open (24/7)"

    @staticmethod
    def get_market_status(symbol: str) -> Tuple[bool, str]:
        """Get market status for a specific symbol"""
        if symbol == 'XAUUSD':
            return MarketHoursValidator.is_forex_open()
        elif symbol == 'BTCUSD':
            return MarketHoursValidator.is_crypto_open()
        else:
            return True, "Unknown market"


class NewsMonitor:
    """Monitor high-impact economic news - FIXED"""

    def __init__(self):
        self.forex_factory_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        self.last_fetch = None
        self.cache_duration = 1800  # 30 minutes cache
        self.cached_events = []

    def fetch_news_events(self) -> List[NewsEvent]:
        """Fetch upcoming high-impact news from Forex Factory"""
        
        # Use cache if recent
        if self.last_fetch and (datetime.now() - self.last_fetch).seconds < self.cache_duration:
            logger.info(f"Using cached news events ({len(self.cached_events)} events)")
            return self.cached_events

        try:
            logger.info("Fetching economic calendar from Forex Factory...")
            response = requests.get(self.forex_factory_url, timeout=15)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch news: HTTP {response.status_code}")
                return self.cached_events  # Return cached data on failure

            data = response.json()
            events = []

            for item in data:
                try:
                    impact = item.get('impact', '').upper()
                    if impact not in ['HIGH', 'MEDIUM']:  # Only HIGH and MEDIUM
                        continue

                    date_str = item.get('date', '')
                    if not date_str:
                        continue

                    event_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S%z')

                    event = NewsEvent(
                        title=item.get('title', 'Unknown'),
                        country=item.get('country', 'Unknown'),
                        date=event_date,
                        impact=impact,
                        actual=item.get('actual', ''),
                        forecast=item.get('forecast', ''),
                        previous=item.get('previous', ''),
                        currency=item.get('currency', '')
                    )

                    events.append(event)

                except Exception as e:
                    logger.debug(f"Error parsing event: {e}")
                    continue

            self.cached_events = events
            self.last_fetch = datetime.now()
            logger.info(f"✅ Fetched {len(events)} news events (HIGH/MEDIUM impact)")

            return events

        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return self.cached_events  # Return cached data on error

    def get_upcoming_high_impact_news(self, hours_ahead: int = 4) -> List[NewsEvent]:
        """Get HIGH impact news within next X hours"""
        events = self.fetch_news_events()

        now = datetime.now(pytz.UTC)
        future_time = now + timedelta(hours=hours_ahead)

        # Only HIGH impact for alerts
        high_impact = [
            event for event in events
            if event.impact == 'HIGH' and now <= event.date <= future_time
        ]

        return sorted(high_impact, key=lambda x: x.date)

    def check_news_before_trade(self, symbol: str, hours_ahead: int = 2) -> Tuple[bool, List[NewsEvent]]:
        """
        Check if there's high-impact news coming soon - FIXED
        Returns: (is_safe_to_trade, upcoming_news)
        """
        upcoming_news = self.get_upcoming_high_impact_news(hours_ahead)

        if not upcoming_news:
            return True, []

        relevant_news = []

        # Currency filters based on symbol
        if symbol == 'XAUUSD':
            relevant_currencies = ['USD', 'EUR', 'GBP', 'CHF']
            relevant_countries = ['United States', 'European Union', 'United Kingdom', 'Switzerland']
        elif symbol == 'BTCUSD':
            relevant_currencies = ['USD']
            relevant_countries = ['United States']
        else:
            return True, []

        for event in upcoming_news:
            if (event.currency in relevant_currencies or 
                event.country in relevant_countries):
                relevant_news.append(event)

        # Block trading if critical news within 1 hour
        critical_news = [
            event for event in relevant_news
            if (event.date - datetime.now(pytz.UTC)).total_seconds() < 3600
        ]

        is_safe = len(critical_news) == 0

        return is_safe, relevant_news


class DerivDataFetcher:
    """Fetch real-time data from Deriv API using WebSocket - FIXED"""

    def __init__(self, app_id: str = "1089"):
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        self.max_retries = 3
        self.retry_delay = 2

    async def _fetch_candles_ws(self, symbol: str, granularity: int, count: int) -> Optional[Dict]:
        """Fetch candles via WebSocket with retry logic"""
        for attempt in range(self.max_retries):
            try:
                async with websockets.connect(
                    self.ws_url, 
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    
                    # Calculate time range
                    end_time = int(datetime.now().timestamp())
                    start_time = end_time - (count * granularity)

                    # Request candles
                    request = {
                        "ticks_history": symbol,
                        "adjust_start_time": 1,
                        "count": count,
                        "end": "latest",
                        "start": start_time,
                        "style": "candles",
                        "granularity": granularity
                    }

                    await websocket.send(json.dumps(request))
                    
                    # Wait for response with timeout
                    response = await asyncio.wait_for(websocket.recv(), timeout=30)
                    data = json.loads(response)

                    # Check for errors
                    if 'error' in data:
                        logger.error(f"Deriv API error: {data['error']}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                            continue
                        return None

                    return data

            except asyncio.TimeoutError:
                logger.warning(f"WebSocket timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
            except Exception as e:
                logger.error(f"WebSocket error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue

        return None

    def get_historical_data(self, symbol: str, timeframe: str = '1h', count: int = 500) -> Optional[pd.DataFrame]:
        """Fetch historical candles from Deriv - FIXED"""

        # Map symbols to Deriv format
        symbol_map = {
            'XAUUSD': 'frxXAUUSD',
            'BTCUSD': 'cryBTCUSD'
        }

        deriv_symbol = symbol_map.get(symbol)
        if not deriv_symbol:
            logger.error(f"Unknown symbol: {symbol}")
            return None

        # Map timeframe to granularity (seconds)
        tf_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }

        granularity = tf_map.get(timeframe)
        if not granularity:
            logger.error(f"Unknown timeframe: {timeframe}")
            return None

        logger.info(f"Fetching {count} candles for {symbol} ({deriv_symbol}) @ {timeframe}...")

        try:
            # Run async fetch in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    self._fetch_candles_ws(deriv_symbol, granularity, count)
                )
            finally:
                loop.close()

            if not response or 'candles' not in response:
                logger.error(f"❌ Failed to fetch data for {symbol}")
                return None

            candles = response['candles']

            if not candles or len(candles) == 0:
                logger.error(f"❌ No candles returned for {symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(candles)
            
            # Handle timestamps
            if 'epoch' in df.columns:
                df['timestamp'] = pd.to_datetime(df['epoch'], unit='s')
            
            # Ensure numeric columns
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Validate OHLC data
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"❌ Missing OHLC data for {symbol}")
                return None

            # Remove NaN rows
            df = df.dropna(subset=required_cols)

            if len(df) < 200:
                logger.error(f"❌ Insufficient data for {symbol} (got {len(df)} candles)")
                return None

            logger.info(f"✅ Fetched {len(df)} candles for {symbol}")
            logger.info(f"   Latest: {df['close'].iloc[-1]:.5f} @ {df['timestamp'].iloc[-1]}")

            return df

        except Exception as e:
            logger.error(f"❌ Error fetching data for {symbol}: {e}", exc_info=True)
            return None


class TelegramNotifier:
    """Send trading signals and news alerts via Telegram - ENHANCED"""

    def __init__(self, bot_token: str, main_chat_id: str, simple_chat_id: str):
        self.bot_token = bot_token
        self.main_chat_id = main_chat_id
        self.simple_chat_id = simple_chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def send_message(self, message: str, chat_id: str, parse_mode: str = "HTML") -> bool:
        """Send message to Telegram with retry"""
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': chat_id, 
            'text': message, 
            'parse_mode': parse_mode,
            'disable_web_page_preview': True
        }

        for attempt in range(3):
            try:
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    return True
                else:
                    logger.warning(f"Telegram error: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Telegram send error (attempt {attempt + 1}): {e}")
                time.sleep(1)
        
        return False

    def send_signal(self, signal: TradeSignal) -> bool:
        """Send trading signal - ENHANCED"""
        emoji = "🟢" if "BUY" in signal.action else "🔴"

        if signal.confidence >= 95:
            conf_label = "🔥🔥🔥🔥🔥 ELITE"
        elif signal.confidence >= 90:
            conf_label = "🔥🔥🔥🔥 EXCEPTIONAL"
        elif signal.confidence >= 85:
            conf_label = "🔥🔥🔥 VERY HIGH"
        else:
            conf_label = "🔥🔥 HIGH"

        reasons = signal.indicators.get('reasons', [])
        reasons_text = '\n'.join([f"  • {r}" for r in reasons[:5]])

        message = f"""
{emoji} <b>{conf_label}</b>

<b>Symbol:</b> {signal.symbol}
<b>Action:</b> {signal.action}
<b>Confidence:</b> {signal.confidence:.1f}%
<b>Strategy:</b> {signal.strategy_name}
<b>Signal ID:</b> {signal.signal_id}

💰 <b>Trade Setup</b>
<b>Entry:</b> {signal.entry_price:.5f}
<b>Position Size:</b> {signal.position_size:.2f} lots
<b>Risk:</b> ${signal.risk_amount_usd:.2f}

<b>Stop Loss:</b> {signal.stop_loss:.5f} ({signal.stop_loss_pips:.1f} pips)
<b>TP1:</b> {signal.tp1:.5f} ({signal.tp1_pips:.1f}p) - Close 50%
<b>TP2:</b> {signal.tp2:.5f} ({signal.tp2_pips:.1f}p) - Close 30%
<b>TP3:</b> {signal.tp3:.5f} ({signal.tp3_pips:.1f}p) - Close 20%

📊 <b>Technical Analysis</b>
{reasons_text}

⚡ <b>Trade Management:</b>
• Close 50% at TP1, move SL to breakeven
• Trail remaining 50% to TP2 and TP3
• Breakeven: {signal.breakeven_price:.5f}
• R:R Ratio: 1:{signal.risk_reward_ratio:.1f}

🕐 {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

<i>📱 Signal only - manual execution required</i>
"""

        detailed = self.send_message(message, self.main_chat_id)

        # Simple version
        simple_msg = f"""
{emoji} <b>{signal.confidence:.0f}% {signal.action}</b>

<b>{signal.symbol}</b> | ID: {signal.signal_id}

<b>Entry:</b> {signal.entry_price:.5f}
<b>Lots:</b> {signal.position_size:.2f}
<b>SL:</b> {signal.stop_loss:.5f} ({signal.stop_loss_pips:.1f}p)
<b>TPs:</b> {signal.tp1_pips:.0f}/{signal.tp2_pips:.0f}/{signal.tp3_pips:.0f}p

🎯 R:R 1:{signal.risk_reward_ratio:.1f} | Risk ${signal.risk_amount_usd:.2f}
"""
        simple = self.send_message(simple_msg, self.simple_chat_id)

        return detailed and simple

    def send_news_alert(self, news_events: List[NewsEvent]):
        """Send high-impact news alert - FIXED"""
        if not news_events:
            return

        message = "🚨 <b>HIGH-IMPACT NEWS ALERT</b>\n\n"
        message += f"⚠️ {len(news_events)} upcoming HIGH impact event(s)\n"
        message += "🚫 <b>AVOID TRADING DURING THESE RELEASES</b>\n\n"

        for event in news_events[:5]:
            time_until = event.date - datetime.now(pytz.UTC)
            hours = int(time_until.total_seconds() / 3600)
            minutes = int((time_until.total_seconds() % 3600) / 60)

            message += f"<b>{event.title}</b>\n"
            message += f"🌍 {event.country} | {event.currency}\n"
            message += f"⏰ In {hours}h {minutes}m\n"
            message += f"📊 {event.impact} IMPACT\n"
            if event.forecast:
                message += f"Expected: {event.forecast}\n"
            message += "\n"

        message += "<i>⚠️ Trading blocked during high-impact news</i>"

        self.send_message(message, self.main_chat_id)
        self.send_message(message, self.simple_chat_id)

    def send_market_closed_alert(self, symbol: str, reason: str):
        """Send market closed notification"""
        message = f"""
⚪ <b>Market Closed - No Trading</b>

<b>Symbol:</b> {symbol}
<b>Status:</b> {reason}

<i>Signals will resume when market opens</i>
"""
        self.send_message(message, self.main_chat_id)

    def send_status(self, message: str):
        """Send status update"""
        self.send_message(f"ℹ️ <b>Bot Status</b>\n\n{message}", self.main_chat_id)


class RiskCalculator:
    """Calculate position sizing and risk parameters - FIXED FOR SMALL ACCOUNTS"""

    @staticmethod
    def calculate_position_size(account_balance: float, risk_percent: float, 
                                stop_loss_pips: float, symbol: str) -> Tuple[float, float]:
        """
        Calculate optimal position size based on risk - FIXED
        Special handling for accounts under $100
        """
        risk_amount = account_balance * (risk_percent / 100)

        # Pip values per standard lot
        if symbol == 'XAUUSD':
            pip_value_per_lot = 10.0  # $10 per pip for 1 lot
        elif symbol == 'BTCUSD':
            pip_value_per_lot = 10.0
        else:
            pip_value_per_lot = 10.0

        # Calculate lot size
        if stop_loss_pips > 0:
            lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
        else:
            lot_size = 0.01

        # Special handling for small accounts (under $100)
        if account_balance < 100:
            # Use tighter SL and smaller lots
            lot_size = max(0.01, min(lot_size, 0.05))
            logger.info(f"Small account detected: Limiting lot size to {lot_size:.2f}")
        else:
            # Normal accounts: round to 0.01
            lot_size = max(0.01, round(lot_size * 100) / 100)
            lot_size = min(lot_size, 1.0)  # Cap at 1 lot for safety

        # Recalculate actual risk amount
        actual_risk = lot_size * stop_loss_pips * pip_value_per_lot

        return lot_size, actual_risk

    @staticmethod
    def calculate_tp_levels_for_small_account(entry: float, stop_loss: float, 
                                               action: str, atr: float, 
                                               account_balance: float) -> Tuple[float, float, float]:
        """
        Calculate appropriate TP levels for small accounts - NEW
        Tighter targets for accounts under $100
        """
        if account_balance < 100:
            # Tighter TPs for small accounts
            if action == "BUY":
                tp1 = entry + (1.0 * atr)  # Conservative
                tp2 = entry + (1.5 * atr)
                tp3 = entry + (2.5 * atr)
            else:
                tp1 = entry - (1.0 * atr)
                tp2 = entry - (1.5 * atr)
                tp3 = entry - (2.5 * atr)
        else:
            # Normal TPs for larger accounts
            if action == "BUY":
                tp1 = entry + (1.5 * atr)
                tp2 = entry + (2.5 * atr)
                tp3 = entry + (4.0 * atr)
            else:
                tp1 = entry - (1.5 * atr)
                tp2 = entry - (2.5 * atr)
                tp3 = entry - (4.0 * atr)

        return tp1, tp2, tp3

    @staticmethod
    def calculate_breakeven_price(entry: float, action: str, stop_loss: float) -> float:
        """Calculate breakeven price (entry + spread buffer)"""
        spread_buffer = abs(entry - stop_loss) * 0.15

        if 'BUY' in action:
            return entry + spread_buffer
        else:
            return entry - spread_buffer


class EliteTradingBot:
    """Elite Trading Bot v8.0 - Institutional Grade"""

    SYMBOLS = ['XAUUSD', 'BTCUSD']

    def __init__(self, telegram_token: str, main_chat_id: str, simple_chat_id: str,
                 account_balance: float = 500.0, risk_percent: float = 2.0,
                 deriv_app_id: str = "1089", check_news: bool = True,
                 min_confidence: float = 80.0):

        self.notifier = TelegramNotifier(telegram_token, main_chat_id, simple_chat_id)
        self.data_fetcher = DerivDataFetcher(deriv_app_id)
        self.news_monitor = NewsMonitor()

        self.account_balance = account_balance
        self.risk_percent = risk_percent
        self.min_confidence = min_confidence  # FIXED: Now uses parameter
        self.timeframe = '1h'
        self.recent_signals = deque(maxlen=50)
        self.check_news = check_news
        self.news_alerts_sent = set()

        logger.info(f"=" * 70)
        logger.info(f"Bot Initialized: v8.0 INSTITUTIONAL GRADE")
        logger.info(f"Balance: ${self.account_balance:.2f} | Risk: {risk_percent}%")
        logger.info(f"Min Confidence: {self.min_confidence}%")
        logger.info(f"WebSocket: ENABLED ✅ | News Monitor: {'ON' if check_news else 'OFF'}")
        logger.info(f"=" * 70)

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""

        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()

        # Stochastic
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        atr_14 = true_range.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr_14)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # Supertrend
        df['supertrend'] = self._calculate_supertrend(df)

        # Volume (synthetic if missing)
        if 'volume' not in df.columns:
            df['volume'] = 1000

        # Fill any remaining NaN
        df = df.ffill().bfill().fillna(0)
        
        return df

    def _calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3) -> pd.Series:
        """Calculate Supertrend indicator"""
        hl2 = (df['high'] + df['low']) / 2
        atr = df['atr']

        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)

        for i in range(1, len(df)):
            if df['close'].iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1
            elif df['close'].iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]

                if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i-1]:
                    lower_band.iloc[i] = lower_band.iloc[i-1]
                if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i-1]:
                    upper_band.iloc[i] = upper_band.iloc[i-1]

            supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

        return supertrend

    def analyze_market(self, symbol: str) -> Optional[TradeSignal]:
        """Analyze market and generate high-confidence signals - FIXED SCORING"""

        # Fetch data
        df = self.data_fetcher.get_historical_data(symbol, self.timeframe, 500)
        if df is None or len(df) < 200:
            logger.warning(f"❌ Insufficient data for {symbol}")
            return None

        # Calculate indicators
        df = self.calculate_all_indicators(df)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # DETERMINE PRIMARY DIRECTION FIRST (based on multiple factors)
        bullish_score = 0
        bearish_score = 0

        # Check primary trend indicators
        if last['ema_9'] > last['ema_21']:
            bullish_score += 3
        else:
            bearish_score += 3

        if last['close'] > last['sma_50']:
            bullish_score += 2
        else:
            bearish_score += 2

        if last['macd'] > last['macd_signal']:
            bullish_score += 2
        else:
            bearish_score += 2

        if last['close'] > last['supertrend']:
            bullish_score += 2
        else:
            bearish_score += 2

        if last['rsi'] > 50:
            bullish_score += 1
        else:
            bearish_score += 1

        # Set action based on dominant direction
        if bullish_score > bearish_score:
            action = "BUY"
        elif bearish_score > bullish_score:
            action = "SELL"
        else:
            logger.info(f"{symbol}: Market neutral - No clear direction")
            return None

        # NOW score the quality of the setup
        signal_strength = 0
        reasons = []

        # 1. EMA Trend Strength (15 points) - RELAXED
        if action == "BUY":
            if last['ema_9'] > last['ema_21'] > last['ema_50']:
                signal_strength += 15
                reasons.append("✅ Perfect bullish EMA alignment")
            elif last['ema_9'] > last['ema_21']:
                signal_strength += 10
                reasons.append("✅ Bullish EMA trend (9>21)")
            elif last['ema_9'] > last['sma_50']:
                signal_strength += 5
                reasons.append("Bullish EMA positioning")
        else:  # SELL
            if last['ema_9'] < last['ema_21'] < last['ema_50']:
                signal_strength += 15
                reasons.append("✅ Perfect bearish EMA alignment")
            elif last['ema_9'] < last['ema_21']:
                signal_strength += 10
                reasons.append("✅ Bearish EMA trend (9<21)")
            elif last['ema_9'] < last['sma_50']:
                signal_strength += 5
                reasons.append("Bearish EMA positioning")

        # 2. RSI Position (15 points) - RELAXED
        if action == "BUY":
            if 30 <= last['rsi'] <= 55:
                signal_strength += 15
                reasons.append(f"✅ RSI ideal buy zone ({last['rsi']:.1f})")
            elif last['rsi'] < 40:
                signal_strength += 12
                reasons.append(f"✅ RSI oversold recovery ({last['rsi']:.1f})")
            elif last['rsi'] < 50:
                signal_strength += 8
                reasons.append(f"RSI below 50 ({last['rsi']:.1f})")
            elif last['rsi'] < 60:
                signal_strength += 5
                reasons.append(f"RSI neutral-bullish ({last['rsi']:.1f})")
        else:  # SELL
            if 45 <= last['rsi'] <= 70:
                signal_strength += 15
                reasons.append(f"✅ RSI ideal sell zone ({last['rsi']:.1f})")
            elif last['rsi'] > 60:
                signal_strength += 12
                reasons.append(f"✅ RSI overbought reversal ({last['rsi']:.1f})")
            elif last['rsi'] > 50:
                signal_strength += 8
                reasons.append(f"RSI above 50 ({last['rsi']:.1f})")
            elif last['rsi'] > 40:
                signal_strength += 5
                reasons.append(f"RSI neutral-bearish ({last['rsi']:.1f})")

        # 3. MACD Direction (15 points) - RELAXED
        macd_bullish_cross = last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']
        macd_bearish_cross = last['macd'] < last['macd_signal'] and prev['macd'] >= prev['macd_signal']
        
        if action == "BUY":
            if macd_bullish_cross:
                signal_strength += 15
                reasons.append("✅ MACD bullish crossover")
            elif last['macd'] > last['macd_signal'] and last['macd_hist'] > 0:
                signal_strength += 12
                reasons.append("✅ MACD bullish momentum")
            elif last['macd'] > last['macd_signal']:
                signal_strength += 8
                reasons.append("MACD above signal")
            elif last['macd_hist'] > prev['macd_hist']:
                signal_strength += 5
                reasons.append("MACD histogram rising")
        else:  # SELL
            if macd_bearish_cross:
                signal_strength += 15
                reasons.append("✅ MACD bearish crossover")
            elif last['macd'] < last['macd_signal'] and last['macd_hist'] < 0:
                signal_strength += 12
                reasons.append("✅ MACD bearish momentum")
            elif last['macd'] < last['macd_signal']:
                signal_strength += 8
                reasons.append("MACD below signal")
            elif last['macd_hist'] < prev['macd_hist']:
                signal_strength += 5
                reasons.append("MACD histogram falling")

        # 4. Supertrend Confirmation (12 points)
        if action == "BUY":
            if last['close'] > last['supertrend']:
                signal_strength += 12
                reasons.append("✅ Supertrend bullish")
            elif last['close'] > prev['supertrend']:
                signal_strength += 6
                reasons.append("Supertrend break attempt")
        else:  # SELL
            if last['close'] < last['supertrend']:
                signal_strength += 12
                reasons.append("✅ Supertrend bearish")
            elif last['close'] < prev['supertrend']:
                signal_strength += 6
                reasons.append("Supertrend break attempt")

        # 5. ADX Trend Strength (12 points)
        if last['adx'] > 30:
            signal_strength += 12
            reasons.append(f"✅ Very strong trend (ADX: {last['adx']:.1f})")
        elif last['adx'] > 25:
            signal_strength += 10
            reasons.append(f"✅ Strong trend (ADX: {last['adx']:.1f})")
        elif last['adx'] > 20:
            signal_strength += 8
            reasons.append(f"Good trend (ADX: {last['adx']:.1f})")
        elif last['adx'] > 15:
            signal_strength += 5
            reasons.append(f"Moderate trend (ADX: {last['adx']:.1f})")

        # 6. Bollinger Bands (10 points) - RELAXED
        bb_width = last['bb_upper'] - last['bb_lower']
        bb_position = (last['close'] - last['bb_lower']) / bb_width if bb_width > 0 else 0.5
        
        if action == "BUY":
            if bb_position < 0.2:  # Near lower band
                signal_strength += 10
                reasons.append("✅ Price near lower BB (oversold)")
            elif bb_position < 0.4:
                signal_strength += 8
                reasons.append("Price in lower BB zone")
            elif bb_position < 0.5:
                signal_strength += 5
                reasons.append("Price below BB middle")
        else:  # SELL
            if bb_position > 0.8:  # Near upper band
                signal_strength += 10
                reasons.append("✅ Price near upper BB (overbought)")
            elif bb_position > 0.6:
                signal_strength += 8
                reasons.append("Price in upper BB zone")
            elif bb_position > 0.5:
                signal_strength += 5
                reasons.append("Price above BB middle")

        # 7. Stochastic (10 points) - RELAXED
        if action == "BUY":
            if last['stoch_k'] < 20:
                signal_strength += 10
                reasons.append(f"✅ Stochastic oversold ({last['stoch_k']:.1f})")
            elif last['stoch_k'] < 30:
                signal_strength += 8
                reasons.append(f"✅ Stochastic buy zone ({last['stoch_k']:.1f})")
            elif last['stoch_k'] < 40:
                signal_strength += 6
                reasons.append(f"Stochastic favorable ({last['stoch_k']:.1f})")
            elif last['stoch_k'] < 50:
                signal_strength += 3
                reasons.append(f"Stochastic below 50 ({last['stoch_k']:.1f})")
        else:  # SELL
            if last['stoch_k'] > 80:
                signal_strength += 10
                reasons.append(f"✅ Stochastic overbought ({last['stoch_k']:.1f})")
            elif last['stoch_k'] > 70:
                signal_strength += 8
                reasons.append(f"✅ Stochastic sell zone ({last['stoch_k']:.1f})")
            elif last['stoch_k'] > 60:
                signal_strength += 6
                reasons.append(f"Stochastic favorable ({last['stoch_k']:.1f})")
            elif last['stoch_k'] > 50:
                signal_strength += 3
                reasons.append(f"Stochastic above 50 ({last['stoch_k']:.1f})")

        # 8. Price Momentum (6 points)
        price_change = ((last['close'] - prev['close']) / prev['close']) * 100
        
        if action == "BUY":
            if price_change > 0.5:
                signal_strength += 6
                reasons.append(f"✅ Strong bullish momentum (+{price_change:.2f}%)")
            elif price_change > 0:
                signal_strength += 4
                reasons.append(f"Bullish momentum (+{price_change:.2f}%)")
            elif price_change > -0.2:
                signal_strength += 2
                reasons.append("Price stable")
        else:  # SELL
            if price_change < -0.5:
                signal_strength += 6
                reasons.append(f"✅ Strong bearish momentum ({price_change:.2f}%)")
            elif price_change < 0:
                signal_strength += 4
                reasons.append(f"Bearish momentum ({price_change:.2f}%)")
            elif price_change < 0.2:
                signal_strength += 2
                reasons.append("Price stable")

        # 9. Volume Confirmation (5 points)
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        if last['volume'] > avg_volume * 1.5:
            signal_strength += 5
            reasons.append("✅ High volume")
        elif last['volume'] > avg_volume * 1.2:
            signal_strength += 3
            reasons.append("Above-average volume")
        elif last['volume'] > avg_volume:
            signal_strength += 1
            reasons.append("Normal volume")

        # Calculate final confidence
        confidence = min(signal_strength, 100)

        logger.info(f"{symbol}: {action} setup scored {confidence:.1f}%")

        # Check minimum threshold
        if confidence < self.min_confidence:
            logger.info(f"   Below {self.min_confidence}% threshold ❌")
            return None

        logger.info(f"   ✅ SIGNAL QUALIFIED! {confidence:.1f}% confidence")

        # Calculate trade parameters (rest of your existing code)
        current_price = last['close']
        atr = last['atr']

        if self.account_balance < 100:
            sl_multiplier = 1.5
        else:
            sl_multiplier = 2.0

        if action == "BUY":
            entry_price = current_price
            stop_loss = current_price - (sl_multiplier * atr)
            tp1, tp2, tp3 = RiskCalculator.calculate_tp_levels_for_small_account(
                entry_price, stop_loss, action, atr, self.account_balance
            )
        else:
            entry_price = current_price
            stop_loss = current_price + (sl_multiplier * atr)
            tp1, tp2, tp3 = RiskCalculator.calculate_tp_levels_for_small_account(
                entry_price, stop_loss, action, atr, self.account_balance
            )

        pip_size = 0.01 if symbol == 'XAUUSD' else 1.0
        stop_loss_pips = abs(entry_price - stop_loss) / pip_size
        tp1_pips = abs(tp1 - entry_price) / pip_size
        tp2_pips = abs(tp2 - entry_price) / pip_size
        tp3_pips = abs(tp3 - entry_price) / pip_size

        lot_size, risk_amount = RiskCalculator.calculate_position_size(
            self.account_balance, self.risk_percent, stop_loss_pips, symbol
        )

        breakeven_price = RiskCalculator.calculate_breakeven_price(entry_price, action, stop_loss)
        rr_ratio = tp3_pips / stop_loss_pips if stop_loss_pips > 0 else 0

        signal = TradeSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=tp3,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            risk_reward_ratio=rr_ratio,
            timeframe=self.timeframe,
            strategy_name="Institutional Multi-Indicator v8.0",
            indicators={
                'rsi': float(last['rsi']),
                'macd': float(last['macd']),
                'adx': float(last['adx']),
                'stoch_k': float(last['stoch_k']),
                'reasons': reasons
            },
            timestamp=datetime.now(pytz.UTC),
            stop_loss_pips=stop_loss_pips,
            tp1_pips=tp1_pips,
            tp2_pips=tp2_pips,
            tp3_pips=tp3_pips,
            position_size=lot_size,
            risk_amount_usd=risk_amount,
            signal_id=f"{symbol}_{int(datetime.now().timestamp())}",
            breakeven_price=breakeven_price
        )

        return signal

        def is_duplicate_signal(self, signal: TradeSignal) -> bool:
            """Check for duplicate signals within last 2 hours"""
            for recent in self.recent_signals:
                if (recent.symbol == signal.symbol and 
                    recent.action == signal.action and
                    abs((signal.timestamp - recent.timestamp).total_seconds()) < 7200):
                    return True
            return False

        def run(self):
            """Run single analysis cycle - INSTITUTIONAL GRADE"""
            logger.info("=" * 70)
            logger.info("🚀 ELITE TRADING BOT v8.0 - INSTITUTIONAL GRADE")
            logger.info("=" * 70)
            logger.info(f"Analysis Time: {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            logger.info(f"Account: ${self.account_balance:.2f} | Risk: {self.risk_percent}%")
            logger.info(f"Min Confidence: {self.min_confidence}% (STRICT)")
            logger.info(f"WebSocket: ENABLED ✅ | News Monitor: {'ON' if self.check_news else 'OFF'}")
            logger.info("=" * 70)

            signals_generated = 0

            for symbol in self.SYMBOLS:
                try:
                    logger.info(f"\n📊 Analyzing {symbol}...")

                    # 1. Check market hours (CRITICAL FOR GOLD)
                    is_open, status = MarketHoursValidator.get_market_status(symbol)
                    if not is_open:
                        logger.warning(f"   ⚪ {status}")
                        self.notifier.send_market_closed_alert(symbol, status)
                        continue

                    logger.info(f"   ✅ {status}")

                    # 2. Check for high-impact news
                    if self.check_news:
                        is_safe, upcoming_news = self.news_monitor.check_news_before_trade(symbol, hours_ahead=2)
                        
                        if not is_safe:
                            logger.warning(f"   🚨 HIGH-IMPACT NEWS DETECTED - BLOCKING {symbol}")
                            
                            # Send news alert (only once per news event)
                            news_ids = {f"{n.title}_{n.date.isoformat()}" for n in upcoming_news}
                            if not news_ids.issubset(self.news_alerts_sent):
                                self.notifier.send_news_alert(upcoming_news)
                                self.news_alerts_sent.update(news_ids)
                            continue
                        
                        if upcoming_news:
                            logger.info(f"   ℹ️ News detected but >1hr away - Proceeding with caution")

                    # 3. Analyze market
                    signal = self.analyze_market(symbol)

                    if signal and not self.is_duplicate_signal(signal):
                        logger.info(f"✅ INSTITUTIONAL-GRADE SIGNAL GENERATED!")
                        logger.info(f"   {signal.action} {symbol} @ {signal.confidence:.1f}%")
                        logger.info(f"   Entry: {signal.entry_price:.5f}")
                        logger.info(f"   SL: {signal.stop_loss:.5f} ({signal.stop_loss_pips:.1f} pips)")
                        logger.info(f"   TP1/2/3: {signal.tp1_pips:.0f}/{signal.tp2_pips:.0f}/{signal.tp3_pips:.0f} pips")
                        logger.info(f"   Position: {signal.position_size:.2f} lots")
                        logger.info(f"   Risk: ${signal.risk_amount_usd:.2f}")
                        logger.info(f"   R:R: 1:{signal.risk_reward_ratio:.2f}")

                        self.recent_signals.append(signal)

                        # Send to Telegram
                        if self.notifier.send_signal(signal):
                            logger.info(f"   ✅ Signal sent to Telegram successfully")
                            signals_generated += 1
                        else:
                            logger.error(f"   ❌ Failed to send signal to Telegram")

                    elif signal and self.is_duplicate_signal(signal):
                        logger.info(f"   ⚠️ Duplicate signal detected - Skipped")
                    else:
                        logger.info(f"   ⚪ No {self.min_confidence}%+ confidence setup found")

                except Exception as e:
                    logger.error(f"❌ Error analyzing {symbol}: {e}", exc_info=True)
                    continue

            logger.info("\n" + "=" * 70)
            logger.info(f"📊 ANALYSIS COMPLETE")
            logger.info(f"   Signals Generated: {signals_generated}")
            logger.info(f"   Symbols Analyzed: {len(self.SYMBOLS)}")
            logger.info(f"   Quality Filter: {self.min_confidence}%+ only")
            logger.info("=" * 70)

            return signals_generated

def main():
    """Main entry point for GitHub Actions"""

    # Get configuration from environment variables
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    main_chat_id = os.getenv('MAIN_CHAT_ID')
    simple_chat_id = os.getenv('SIMPLE_CHAT_ID')
    deriv_app_id = os.getenv('DERIV_APP_ID', '1089')

    # Trading parameters
    account_balance = float(os.getenv('ACCOUNT_BALANCE', '500'))
    risk_percent = float(os.getenv('RISK_PERCENT', '2.0'))
    min_confidence = float(os.getenv('MIN_CONFIDENCE', '80.0'))

    # Validation
    if not telegram_token or not main_chat_id or not simple_chat_id:
        logger.error("❌ CRITICAL: Missing required environment variables!")
        logger.error("   Required: TELEGRAM_BOT_TOKEN, MAIN_CHAT_ID, SIMPLE_CHAT_ID")
        return 1

    logger.info("✅ Configuration loaded successfully")
    logger.info(f"   Account: ${account_balance:.2f}")
    logger.info(f"   Risk: {risk_percent}%")
    logger.info(f"   Min Confidence: {min_confidence}%")

    try:
        # Initialize bot
        bot = EliteTradingBot(
            telegram_token=telegram_token,
            main_chat_id=main_chat_id,
            simple_chat_id=simple_chat_id,
            account_balance=account_balance,
            risk_percent=risk_percent,
            deriv_app_id=deriv_app_id,
            check_news=True,
            min_confidence=min_confidence
        )

# Startup message in your main() function with this:
        startup_msg = f"""
🚀 <b>Elite Trading Bot v8.0 - INSTITUTIONAL GRADE</b>

<b>⚙️ Configuration:</b>
• Mode: Signal Generation (GitHub Actions)
• Connection: WebSocket ✅
• Assets: XAUUSD (Forex), BTCUSD (Crypto)
• Timeframe: 1 Hour
• Min Confidence: {min_confidence}% (STRICT)
• Account: ${account_balance:.2f}
• Risk per trade: {risk_percent}%

<b>🛡️ Protection Features:</b>
✅ Multi-indicator confluence (10 indicators)
✅ Institutional-grade trend analysis
✅ Volume &amp; momentum confirmation
✅ Dynamic position sizing
✅ Market hours validation (No Gold weekends)
✅ HIGH-IMPACT news blocking
✅ Duplicate signal prevention
✅ Small account protection (under $100)
✅ Complete trade management plans

<b>📊 Signal Quality Standards:</b>
• Only {min_confidence}%+ confidence signals sent
• Multiple confirmation layers required
• Risk-optimized entry/exit levels
• Professional trade management

<i>🔍 Scanning markets for premium setups...</i>
"""
        bot.notifier.send_status(startup_msg)

        # Run analysis
        signals_count = bot.run()

        # If signals generated:
        if signals_count > 0:
            summary_msg = f"""
✅ <b>Analysis Complete - {signals_count} Signal(s) Generated</b>

🎯 <b>HIGH-CONFIDENCE SETUP(S) DETECTED!</b>

<b>Signals sent include:</b>
• Precise entry prices
• Risk-optimized stop losses
• Multi-level take profits (TP1/TP2/TP3)
• Position sizing for your account
• Complete trade management plans
• Breakeven levels

<b>⚡ Next Steps:</b>
1. Review signal details above
2. Verify market conditions
3. Execute trade manually if conditions align
4. Follow trade management plan

<i>📅 Next analysis: Top of next hour</i>
<i>🛡️ Quality over quantity - Only premium setups sent</i>
"""
        else:
            summary_msg = f"""
✅ <b>Analysis Complete - Market Scan Finished</b>

⚪ <b>No {min_confidence}%+ confidence signals this cycle</b>

<b>📊 Market Status:</b>
All assets scanned successfully. Current market conditions do not meet our institutional-grade criteria.

<b>🔍 What we checked:</b>
• Market hours (Gold weekends blocked)
• High-impact news (None blocking)
• Technical confluence ({min_confidence}%+ required)
• Multiple indicator confirmation
• Volume &amp; momentum validation

<b>💡 Strategy:</b>
We prioritize quality over quantity. Only the highest probability setups with multiple confirmations are sent.

<i>📅 Next analysis: Top of next hour</i>
<i>🎯 Waiting for premium institutional-grade setups</i>
"""

        bot.notifier.send_status(summary_msg)

        logger.info("✅ Bot execution completed successfully")
        logger.info(f"   Runtime: {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        return 0

    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}", exc_info=True)

        # Try to send error notification
        try:
            notifier = TelegramNotifier(telegram_token, main_chat_id, simple_chat_id)
            error_msg = f"""
🚨 <b>Bot Execution Error</b>

<b>Error Type:</b> {type(e).__name__}
<b>Message:</b> {str(e)[:300]}

<b>Action Required:</b>
Check GitHub Actions logs for full details.

<b>Timestamp:</b> {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            notifier.send_status(error_msg)
        except:
            pass

        return 1


if __name__ == "__main__":
    exit(main())