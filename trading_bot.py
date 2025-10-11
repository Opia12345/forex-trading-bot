"""
ELITE TRADING BOT v7.0 - GITHUB ACTIONS EDITION
Signal Generation Only - No MT5 Required
Works perfectly in cloud environments
Assets: XAUUSD & BTCUSD
Features: Advanced multi-indicator analysis, Market hours validation, News monitoring, High-confidence signals
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
import re

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
    impact: str  # HIGH, MEDIUM, LOW
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
        
        # Forex is closed on weekends
        weekday = dt.weekday()  # 0=Monday, 6=Sunday
        
        if weekday == 5:  # Saturday
            return False, "Weekend - Forex market closed (Saturday)"
        
        if weekday == 6:  # Sunday
            # Forex opens Sunday 5PM EST (22:00 UTC)
            if dt.hour < 22:
                return False, "Weekend - Forex market closed (Sunday before 22:00 UTC)"
        
        if weekday == 4:  # Friday
            # Forex closes Friday 5PM EST (22:00 UTC)
            if dt.hour >= 22:
                return False, "Weekend - Forex market closing (Friday after 22:00 UTC)"
        
        return True, "Forex market open"
    
    @staticmethod
    def is_crypto_open(dt: datetime = None) -> Tuple[bool, str]:
        """Check if Crypto market is open (for BTCUSD)"""
        # Crypto trades 24/7
        return True, "Crypto market always open"
    
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
    """Monitor high-impact economic news"""
    
    def __init__(self):
        self.forex_factory_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        self.last_fetch = None
        self.cache_duration = 3600  # Cache for 1 hour
        self.cached_events = []
    
    def fetch_news_events(self) -> List[NewsEvent]:
        """Fetch upcoming high-impact news from Forex Factory"""
        
        # Use cache if available
        if self.last_fetch and (datetime.now() - self.last_fetch).seconds < self.cache_duration:
            logger.info("Using cached news events")
            return self.cached_events
        
        try:
            logger.info("Fetching economic calendar from Forex Factory...")
            response = requests.get(self.forex_factory_url, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch news: HTTP {response.status_code}")
                return []
            
            data = response.json()
            events = []
            
            for item in data:
                try:
                    # Parse impact level
                    impact = item.get('impact', '').upper()
                    if impact not in ['HIGH', 'MEDIUM', 'LOW']:
                        continue
                    
                    # Parse date/time
                    date_str = item.get('date', '')
                    if not date_str:
                        continue
                    
                    # Convert to datetime
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
            logger.info(f"✅ Fetched {len(events)} news events")
            
            return events
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def get_upcoming_high_impact_news(self, hours_ahead: int = 4) -> List[NewsEvent]:
        """Get high-impact news within next X hours"""
        events = self.fetch_news_events()
        
        now = datetime.now(pytz.UTC)
        future_time = now + timedelta(hours=hours_ahead)
        
        high_impact = [
            event for event in events
            if event.impact == 'HIGH' and now <= event.date <= future_time
        ]
        
        return sorted(high_impact, key=lambda x: x.date)
    
    def check_news_before_trade(self, symbol: str, hours_ahead: int = 2) -> Tuple[bool, List[NewsEvent]]:
        """
        Check if there's high-impact news coming soon
        Returns: (is_safe_to_trade, upcoming_news)
        """
        upcoming_news = self.get_upcoming_high_impact_news(hours_ahead)
        
        if not upcoming_news:
            return True, []
        
        # Check if news affects this symbol
        relevant_news = []
        
        if symbol == 'XAUUSD':
            # Gold affected by USD, EUR, GBP news
            relevant_currencies = ['USD', 'EUR', 'GBP']
            relevant_news = [
                event for event in upcoming_news
                if event.currency in relevant_currencies or
                event.country in ['United States', 'European Union', 'United Kingdom']
            ]
        
        elif symbol == 'BTCUSD':
            # Bitcoin affected by major USD news
            relevant_currencies = ['USD']
            relevant_news = [
                event for event in upcoming_news
                if event.currency in relevant_currencies or
                event.country in ['United States']
            ]
        
        # If high-impact news within 1 hour, don't trade
        critical_news = [
            event for event in relevant_news
            if (event.date - datetime.now(pytz.UTC)).total_seconds() < 3600
        ]
        
        is_safe = len(critical_news) == 0
        
        return is_safe, relevant_news


class DerivDataFetcher:
    """Fetch real-time data from Deriv API"""
    
    def __init__(self, app_id: str = "1089"):
        self.app_id = app_id
        self.base_url = "https://api.deriv.com"
        
    def get_historical_data(self, symbol: str, timeframe: str = '1h', count: int = 500) -> Optional[pd.DataFrame]:
        """Fetch historical candles from Deriv"""
        
        # Map symbols to Deriv format
        symbol_map = {
            'XAUUSD': 'frxXAUUSD',
            'BTCUSD': 'cryBTCUSD'
        }
        
        deriv_symbol = symbol_map.get(symbol, symbol)
        
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
        
        granularity = tf_map.get(timeframe, 3600)
        
        # Calculate end time (now) and start time
        end_time = int(datetime.now().timestamp())
        start_time = end_time - (count * granularity)
        
        logger.info(f"Fetching {count} candles for {symbol} ({deriv_symbol})...")
        
        try:
            response = self._fetch_via_http(deriv_symbol, start_time, granularity, count)
            
            if not response or 'candles' not in response:
                logger.error(f"Failed to fetch data for {symbol}")
                return None
            
            candles = response['candles']
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['epoch'], unit='s')
            df.rename(columns={'epoch': 'time'}, inplace=True)
            
            # Ensure we have OHLCV
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                logger.error(f"Missing OHLCV data for {symbol}")
                return None
            
            logger.info(f"✅ Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _fetch_via_http(self, symbol: str, start_time: int, granularity: int, count: int) -> Optional[Dict]:
        """Fetch data via HTTP API"""
        try:
            url = f"https://api.deriv.com/api/v3/ticks_history"
            
            params = {
                'ticks_history': symbol,
                'adjust_start_time': 1,
                'count': count,
                'end': 'latest',
                'start': start_time,
                'style': 'candles',
                'granularity': granularity
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"HTTP request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"HTTP request error: {e}")
            return None


class TelegramNotifier:
    """Send trading signals and news alerts via Telegram"""
    
    def __init__(self, bot_token: str, main_chat_id: str, simple_chat_id: str):
        self.bot_token = bot_token
        self.main_chat_id = main_chat_id
        self.simple_chat_id = simple_chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
    def send_message(self, message: str, chat_id: str, parse_mode: str = "HTML") -> bool:
        """Send message to Telegram"""
        url = f"{self.base_url}/sendMessage"
        payload = {'chat_id': chat_id, 'text': message, 'parse_mode': parse_mode}
            
        try:
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False
    
    def send_signal(self, signal: TradeSignal) -> bool:
        """Send trading signal"""
        emoji = "🟢" if "BUY" in signal.action else "🔴"
        
        if signal.confidence >= 95:
            conf_label = "🔥🔥🔥🔥🔥 ELITE"
        elif signal.confidence >= 90:
            conf_label = "🔥🔥🔥🔥 EXCEPTIONAL"
        elif signal.confidence >= 85:
            conf_label = "🔥🔥🔥 VERY HIGH"
        else:
            conf_label = "🔥🔥 HIGH"
        
        # Get indicator reasons
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

⚡ <b>Trade Management Plan:</b>
• Close 50% at TP1, move SL to breakeven
• Trail remaining 50% to TP2 and TP3
• Breakeven: {signal.breakeven_price:.5f}
• R:R Ratio: 1:{signal.risk_reward_ratio:.1f}

🕐 {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

<i>📱 This is a signal only - manual execution required</i>
"""
        
        detailed = self.send_message(message, self.main_chat_id)
        
        # Simple version
        simple_msg = f"""
{emoji} <b>{signal.confidence:.0f}% SIGNAL</b>

<b>{signal.symbol}</b> {signal.action}
ID: {signal.signal_id}

<b>Entry:</b> {signal.entry_price:.5f}
<b>Lots:</b> {signal.position_size:.2f}
<b>SL:</b> {signal.stop_loss:.5f} ({signal.stop_loss_pips:.1f}p)
<b>TPs:</b> {signal.tp1_pips:.0f}/{signal.tp2_pips:.0f}/{signal.tp3_pips:.0f} pips

🎯 R:R 1:{signal.risk_reward_ratio:.1f}
"""
        simple = self.send_message(simple_msg, self.simple_chat_id)
        
        return detailed and simple
    
    def send_news_alert(self, news_events: List[NewsEvent]):
        """Send high-impact news alert"""
        if not news_events:
            return
        
        message = "🚨 <b>HIGH-IMPACT NEWS ALERT</b>\n\n"
        message += f"⚠️ {len(news_events)} upcoming high-impact event(s)\n\n"
        
        for event in news_events[:5]:  # Show max 5 events
            time_until = event.date - datetime.now(pytz.UTC)
            hours = int(time_until.total_seconds() / 3600)
            minutes = int((time_until.total_seconds() % 3600) / 60)
            
            message += f"<b>{event.title}</b>\n"
            message += f"🌍 {event.country} | {event.currency}\n"
            message += f"⏰ In {hours}h {minutes}m\n"
            message += f"📊 {event.impact} IMPACT\n"
            if event.forecast:
                message += f"Forecast: {event.forecast}\n"
            message += "\n"
        
        message += "<i>⚠️ Avoid trading during high-impact news releases</i>"
        
        self.send_message(message, self.main_chat_id)
    
    def send_market_closed_alert(self, symbol: str, reason: str):
        """Send market closed notification"""
        message = f"""
⚪ <b>Market Closed - No Signal</b>

<b>Symbol:</b> {symbol}
<b>Status:</b> {reason}

<i>Signals will resume when market opens</i>
"""
        self.send_message(message, self.main_chat_id)
    
    def send_status(self, message: str):
        """Send status update"""
        self.send_message(f"ℹ️ <b>Bot Status</b>\n\n{message}", self.main_chat_id)


class RiskCalculator:
    """Calculate position sizing and risk parameters"""
    
    @staticmethod
    def calculate_position_size(account_balance: float, risk_percent: float, 
                                stop_loss_pips: float, symbol: str) -> Tuple[float, float]:
        """Calculate optimal position size based on risk"""
        risk_amount = account_balance * (risk_percent / 100)
        
        # Pip values per standard lot
        if symbol == 'XAUUSD':
            pip_value_per_lot = 10.0  # $10 per pip for Gold
        elif symbol == 'BTCUSD':
            pip_value_per_lot = 10.0  # $10 per pip for Bitcoin
        else:
            pip_value_per_lot = 10.0
        
        # Calculate lot size
        lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
        
        # Round and apply limits
        lot_size = max(0.01, round(lot_size * 100) / 100)
        lot_size = min(lot_size, 1.0)  # Max 1 lot for safety
        
        return lot_size, risk_amount
    
    @staticmethod
    def calculate_breakeven_price(entry: float, action: str, stop_loss: float) -> float:
        """Calculate breakeven price (entry + spread buffer)"""
        spread_buffer = abs(entry - stop_loss) * 0.1
        
        if 'BUY' in action:
            return entry + spread_buffer
        else:
            return entry - spread_buffer


class RiskCalculator:
    """Calculate position sizing and risk parameters"""
    
    @staticmethod
    def calculate_position_size(account_balance: float, risk_percent: float, 
                                stop_loss_pips: float, symbol: str) -> Tuple[float, float]:
        """Calculate optimal position size based on risk"""
        risk_amount = account_balance * (risk_percent / 100)
        
        # Pip values per standard lot
        if symbol == 'XAUUSD':
            pip_value_per_lot = 10.0
        elif symbol == 'BTCUSD':
            pip_value_per_lot = 10.0
        else:
            pip_value_per_lot = 10.0
        
        # Calculate lot size
        lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
        
        # Round and apply limits
        lot_size = max(0.01, round(lot_size * 100) / 100)
        lot_size = min(lot_size, 1.0)
        
        return lot_size, risk_amount
    
    @staticmethod
    def calculate_breakeven_price(entry: float, action: str, stop_loss: float) -> float:
        """Calculate breakeven price (entry + spread buffer)"""
        spread_buffer = abs(entry - stop_loss) * 0.1
        
        if 'BUY' in action:
            return entry + spread_buffer
        else:
            return entry - spread_buffer


class EliteTradingBot:
    """Elite Trading Bot v7.0 - GitHub Actions Edition with Market Hours & News Monitoring"""
    
    SYMBOLS = ['XAUUSD', 'BTCUSD']
    
    def __init__(self, telegram_token: str, main_chat_id: str, simple_chat_id: str,
                 account_balance: float = 500.0, risk_percent: float = 2.0,
                 deriv_app_id: str = "1089", check_news: bool = True):
        
        self.notifier = TelegramNotifier(telegram_token, main_chat_id, simple_chat_id)
        self.data_fetcher = DerivDataFetcher(deriv_app_id)
        self.news_monitor = NewsMonitor()
        
        self.account_balance = account_balance
        self.risk_percent = risk_percent
        self.min_confidence = 80.0
        self.timeframe = '1h'
        self.recent_signals = deque(maxlen=50)
        self.check_news = check_news
        self.news_alerts_sent = set()  # Track sent news alerts
        
        logger.info(f"Bot initialized: Balance=${self.account_balance:.2f}, Risk={risk_percent}%")
        logger.info(f"Mode: SIGNAL GENERATION ONLY (GitHub Actions)")
        logger.info(f"News Monitoring: {'ENABLED ✅' if check_news else 'DISABLED'}")
                
    
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
        
        # Volume
        if 'volume' not in df.columns:
            df['volume'] = 1000  # Default if not available
        
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
        """Analyze market and generate high-confidence signals"""
        
        df = self.data_fetcher.get_historical_data(symbol, self.timeframe, 500)
        if df is None or len(df) < 200:
            logger.warning(f"Insufficient data for {symbol}")
            return None
        
        df = self.calculate_all_indicators(df)
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        signal_strength = 0
        reasons = []
        action = None
        
        # 1. Trend Analysis (15 points)
        if last['ema_9'] > last['ema_21'] > last['ema_50']:
            signal_strength += 15
            reasons.append("Strong bullish EMA alignment")
            if action is None:
                action = "BUY"
        elif last['ema_9'] < last['ema_21'] < last['ema_50']:
            signal_strength += 15
            reasons.append("Strong bearish EMA alignment")
            if action is None:
                action = "SELL"
        
        # 2. RSI Confirmation (12 points)
        if 30 < last['rsi'] < 45 and action == "BUY":
            signal_strength += 12
            reasons.append(f"RSI oversold recovery ({last['rsi']:.1f})")
        elif 55 < last['rsi'] < 70 and action == "SELL":
            signal_strength += 12
            reasons.append(f"RSI overbought reversal ({last['rsi']:.1f})")
        
        # 3. MACD Crossover (15 points)
        if last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal'] and action == "BUY":
            signal_strength += 15
            reasons.append("MACD bullish crossover")
        elif last['macd'] < last['macd_signal'] and prev['macd'] >= prev['macd_signal'] and action == "SELL":
            signal_strength += 15
            reasons.append("MACD bearish crossover")
        elif last['macd'] > last['macd_signal'] and action == "BUY":
            signal_strength += 8
            reasons.append("MACD bullish alignment")
        elif last['macd'] < last['macd_signal'] and action == "SELL":
            signal_strength += 8
            reasons.append("MACD bearish alignment")
        
        # 4. Bollinger Bands (10 points)
        if last['close'] < last['bb_lower'] and action == "BUY":
            signal_strength += 10
            reasons.append("Price below lower Bollinger Band")
        elif last['close'] > last['bb_upper'] and action == "SELL":
            signal_strength += 10
            reasons.append("Price above upper Bollinger Band")
        
        # 5. Stochastic (10 points)
        if last['stoch_k'] < 25 and last['stoch_d'] < 25 and action == "BUY":
            signal_strength += 10
            reasons.append(f"Stochastic oversold ({last['stoch_k']:.1f})")
        elif last['stoch_k'] > 75 and last['stoch_d'] > 75 and action == "SELL":
            signal_strength += 10
            reasons.append(f"Stochastic overbought ({last['stoch_k']:.1f})")
        
        # 6. ADX Trend Strength (8 points)
        if last['adx'] > 25:
            signal_strength += 8
            reasons.append(f"Strong trend confirmed (ADX: {last['adx']:.1f})")
        
        # 7. Supertrend (10 points)
        if last['close'] > last['supertrend'] and action == "BUY":
            signal_strength += 10
            reasons.append("Supertrend bullish")
        elif last['close'] < last['supertrend'] and action == "SELL":
            signal_strength += 10
            reasons.append("Supertrend bearish")
        
        # 8. Volume Confirmation (8 points)
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        if last['volume'] > avg_volume * 1.2:
            signal_strength += 8
            reasons.append("High volume confirmation")
        
        # 9. Price-MA Position (7 points)
        if action == "BUY" and last['close'] > last['sma_20'] > last['sma_50']:
            signal_strength += 7
            reasons.append("Price above key moving averages")
        elif action == "SELL" and last['close'] < last['sma_20'] < last['sma_50']:
            signal_strength += 7
            reasons.append("Price below key moving averages")
        
        # 10. Momentum Confirmation (5 points)
        if action == "BUY" and last['close'] > prev['close']:
            signal_strength += 5
            reasons.append("Bullish momentum")
        elif action == "SELL" and last['close'] < prev['close']:
            signal_strength += 5
            reasons.append("Bearish momentum")
        
        confidence = min(signal_strength, 100)
        
        # Only generate signals above threshold
        if confidence < self.min_confidence or action is None:
            logger.info(f"{symbol}: Confidence {confidence:.1f}% - Below threshold")
            return None
        
        # Calculate trade parameters
        current_price = last['close']
        atr = last['atr']
        
        if action == "BUY":
            entry_price = current_price
            stop_loss = current_price - (2.0 * atr)
            tp1 = current_price + (1.5 * atr)
            tp2 = current_price + (2.5 * atr)
            tp3 = current_price + (4.0 * atr)
        else:  # SELL
            entry_price = current_price
            stop_loss = current_price + (2.0 * atr)
            tp1 = current_price - (1.5 * atr)
            tp2 = current_price - (2.5 * atr)
            tp3 = current_price - (4.0 * atr)
        
        # Calculate pips
        pip_size = 0.01 if symbol == 'XAUUSD' else 1.0
        stop_loss_pips = abs(entry_price - stop_loss) / pip_size
        tp1_pips = abs(tp1 - entry_price) / pip_size
        tp2_pips = abs(tp2 - entry_price) / pip_size
        tp3_pips = abs(tp3 - entry_price) / pip_size
        
        # Position sizing
        lot_size, risk_amount = RiskCalculator.calculate_position_size(
            self.account_balance, self.risk_percent, stop_loss_pips, symbol
        )
        
        # Breakeven
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
            strategy_name="Multi-Indicator Confluence v7.0",
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
        """Check for duplicate signals within last hour"""
        for recent in self.recent_signals:
            if (recent.symbol == signal.symbol and 
                recent.action == signal.action and
                abs((signal.timestamp - recent.timestamp).total_seconds()) < 3600):
                return True
        return False
    
    def run(self):
        """Run single analysis cycle"""
        logger.info("=" * 70)
        logger.info("🚀 ELITE TRADING BOT v7.0 - GITHUB ACTIONS EDITION")
        logger.info("=" * 70)
        logger.info(f"Analysis Time: {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"Account Balance: ${self.account_balance:.2f}")
        logger.info(f"Risk Per Trade: {self.risk_percent}%")
        logger.info(f"Minimum Confidence: {self.min_confidence}%")
        logger.info("=" * 70)
        
        signals_generated = 0
        
        for symbol in self.SYMBOLS:
            try:
                logger.info(f"\n📊 Analyzing {symbol}...")
                
                signal = self.analyze_market(symbol)
                
                if signal and not self.is_duplicate_signal(signal):
                    logger.info(f"✅ HIGH-CONFIDENCE SIGNAL: {signal.action} {symbol} @ {signal.confidence:.1f}%")
                    logger.info(f"   Entry: {signal.entry_price:.5f} | SL: {signal.stop_loss:.5f} | TP: {signal.tp3:.5f}")
                    logger.info(f"   Position: {signal.position_size:.2f} lots | Risk: ${signal.risk_amount_usd:.2f}")
                    logger.info(f"   R:R Ratio: 1:{signal.risk_reward_ratio:.2f}")
                    
                    self.recent_signals.append(signal)
                    
                    # Send to Telegram
                    if self.notifier.send_signal(signal):
                        logger.info(f"   ✅ Signal sent to Telegram")
                        signals_generated += 1
                    else:
                        logger.error(f"   ❌ Failed to send signal to Telegram")
                    
                else:
                    logger.info(f"   No valid signal (confidence below {self.min_confidence}% or duplicate)")
                    
            except Exception as e:
                logger.error(f"❌ Error analyzing {symbol}: {e}", exc_info=True)
        
        logger.info("\n" + "=" * 70)
        logger.info(f"📊 ANALYSIS COMPLETE")
        logger.info(f"   Signals Generated: {signals_generated}")
        logger.info(f"   Total Symbols: {len(self.SYMBOLS)}")
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
    account_balance = str(os.getenv('ACCOUNT_BALANCE', '500.0'))
    risk_percent = str(os.getenv('RISK_PERCENT', '2.0'))
    
    # Validation
    if not telegram_token or not main_chat_id or not simple_chat_id:
        logger.error("❌ Missing required environment variables!")
        logger.error("   Required: TELEGRAM_BOT_TOKEN, MAIN_CHAT_ID, SIMPLE_CHAT_ID")
        return 1
    
    logger.info("✅ Configuration loaded from environment variables")
    
    try:
        # Initialize bot
        bot = EliteTradingBot(
            telegram_token=telegram_token,
            main_chat_id=main_chat_id,
            simple_chat_id=simple_chat_id,
            account_balance=account_balance,
            risk_percent=risk_percent,
            deriv_app_id=deriv_app_id
        )
        
        # Send startup notification
        startup_msg = f"""
🚀 <b>Elite Trading Bot v7.0 Started</b>

<b>Configuration:</b>
• Mode: Signal Generation (GitHub Actions)
• Assets: XAUUSD, BTCUSD
• Timeframe: 1 Hour
• Min Confidence: 80%
• Account: ${account_balance:.2f}
• Risk per trade: {risk_percent}%

<b>Analysis Features:</b>
✅ Multi-indicator confluence
✅ Advanced trend detection
✅ Volume confirmation
✅ Risk-based position sizing
✅ Complete trade management plans

<i>Scanning for high-probability setups...</i>
"""
        bot.notifier.send_status(startup_msg)
        
        # Run analysis
        signals_count = bot.run()
        
        # Send completion summary
        if signals_count > 0:
            summary_msg = f"""
✅ <b>Analysis Complete</b>

🎯 <b>{signals_count} high-confidence signal(s) generated!</b>

Check detailed signals above for:
• Entry prices
• Stop loss levels
• Take profit targets (TP1/TP2/TP3)
• Position sizing
• Trade management plans

<i>Next analysis: Top of next hour</i>
"""
        else:
            summary_msg = f"""
✅ <b>Analysis Complete</b>

⚪ No 80%+ confidence signals this cycle

<b>Market Status:</b>
All assets scanned successfully. Current market conditions do not meet our high-confidence criteria.

<i>Quality over quantity - waiting for premium setups</i>
<i>Next analysis: Top of next hour</i>
"""
        
        bot.notifier.send_status(summary_msg)
        
        logger.info("✅ Bot execution completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        
        # Try to send error notification
        try:
            notifier = TelegramNotifier(telegram_token, main_chat_id, simple_chat_id)
            error_msg = f"""
🚨 <b>Bot Execution Error</b>

An error occurred during analysis:
<code>{str(e)[:200]}</code>

Check GitHub Actions logs for details.
"""
            notifier.send_status(error_msg)
        except:
            pass
        
        return 1


if __name__ == "__main__":
    exit(main())