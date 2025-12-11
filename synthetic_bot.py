"""
BOOM & CRASH: CORRECTED VERSION
Using BOOM500/CRASH500 (confirmed working) and BOOM300N/CRASH300N
"""

import os
import sys
import logging
import json
from typing import Optional, List, Tuple, Dict
from datetime import datetime, time, timezone
from collections import deque
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import requests
import asyncio
import websockets

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('boom_crash_working.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    """CORRECTED: Using working symbol codes"""
    
    # OPTION 1: Use 500s (confirmed working, slightly more volatile but tradeable)
    SYMBOLS = {
        'CRASH_500': 'CRASH500',
        'BOOM_500': 'BOOM500',
    }
    
    # OPTION 2: Uncomment to use 300s (use BOOM300N/CRASH300N - note the "N")
    # SYMBOLS = {
    #     'CRASH_300': 'CRASH300N',
    #     'BOOM_300': 'BOOM300N',
    # }
    
    # Trading sessions (UTC)
    LONDON_START = time(8, 0)
    LONDON_END = time(12, 0)
    NY_START = time(13, 0)
    NY_END = time(17, 0)
    OVERLAP_START = time(13, 0)
    OVERLAP_END = time(16, 0)
    
    # Risk Management (Conservative)
    RISK_PER_TRADE = 0.5
    MAX_DAILY_TRADES = 5
    MAX_DAILY_LOSS = 2.0
    MAX_CONCURRENT_TRADES = 2
    
    # Strategy Parameters
    MIN_CONFIDENCE = 70
    SCALP_TARGET_PIPS = 20  # Adjusted for 500 indices
    STOP_LOSS_PIPS = 40     # Adjusted for 500 indices
    
    # Support/Resistance
    LOOKBACK_PERIODS = 100
    TOUCH_THRESHOLD = 0.002
    
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    CHAT_ID = os.getenv('MAIN_CHAT_ID', '')

@dataclass
class Signal:
    signal_id: str
    symbol: str
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    setup_type: str
    timestamp: datetime
    reasoning: List[str] = field(default_factory=list)

class DataFetcher:
    """Data fetcher with correct symbol codes"""
    
    SYMBOL_MAP = {
        'BOOM_300': 'BOOM300N',    # Corrected: Added "N"
        'BOOM_500': 'BOOM500',      # Confirmed working
        'BOOM_600': 'BOOM600',
        'BOOM_900': 'BOOM900',
        'BOOM_1000': 'BOOM1000',
        'CRASH_300': 'CRASH300N',   # Corrected: Added "N"
        'CRASH_500': 'CRASH500',    # Confirmed working
        'CRASH_600': 'CRASH600',
        'CRASH_900': 'CRASH900',
        'CRASH_1000': 'CRASH1000',
    }
    
    def __init__(self, app_id: str = "1089"):
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    
    async def _fetch_async(self, symbol: str, granularity: int = 60, count: int = 300) -> Optional[List]:
        try:
            async with websockets.connect(self.ws_url, ping_interval=30) as ws:
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
                
                if 'candles' in data:
                    return data.get('candles')
                elif 'error' in data:
                    logger.error(f"API Error for {symbol}: {data['error'].get('message')}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def get_candles(self, symbol: str) -> Optional[pd.DataFrame]:
        deriv_symbol = self.SYMBOL_MAP.get(symbol)
        if not deriv_symbol:
            logger.error(f"Unknown symbol: {symbol}")
            return None
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            candles = loop.run_until_complete(self._fetch_async(deriv_symbol))
            
            if not candles:
                logger.error(f"No candles returned for {symbol} ({deriv_symbol})")
                return None
            
            df = pd.DataFrame(candles)
            df = df.rename(columns={
                'open': 'open', 'high': 'high',
                'low': 'low', 'close': 'close',
                'epoch': 'time'
            })
            
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            logger.info(f"✅ Loaded {len(df)} candles for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error processing candles: {e}", exc_info=True)
            return None

class Indicators:
    """Technical indicators"""
    
    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['sma_100'] = df['close'].rolling(window=100).mean()
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = pd.Series(true_range).rolling(window=14).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df.dropna(inplace=True)
        return df

class SupportResistance:
    """Support and Resistance detection"""
    
    @staticmethod
    def find_levels(df: pd.DataFrame, lookback: int = 100) -> Dict[str, List[float]]:
        recent = df.tail(lookback)
        
        highs = []
        lows = []
        
        for i in range(2, len(recent) - 2):
            if (recent.iloc[i]['high'] > recent.iloc[i-1]['high'] and 
                recent.iloc[i]['high'] > recent.iloc[i-2]['high'] and
                recent.iloc[i]['high'] > recent.iloc[i+1]['high'] and
                recent.iloc[i]['high'] > recent.iloc[i+2]['high']):
                highs.append(recent.iloc[i]['high'])
            
            if (recent.iloc[i]['low'] < recent.iloc[i-1]['low'] and 
                recent.iloc[i]['low'] < recent.iloc[i-2]['low'] and
                recent.iloc[i]['low'] < recent.iloc[i+1]['low'] and
                recent.iloc[i]['low'] < recent.iloc[i+2]['low']):
                lows.append(recent.iloc[i]['low'])
        
        resistance_levels = SupportResistance._cluster_levels(highs) if highs else []
        support_levels = SupportResistance._cluster_levels(lows) if lows else []
        
        return {
            'resistance': resistance_levels[:3],
            'support': support_levels[:3]
        }
    
    @staticmethod
    def _cluster_levels(levels: List[float], threshold: float = 0.005) -> List[float]:
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clustered.append(np.mean(current_cluster))
        return clustered
    
    @staticmethod
    def is_near_level(price: float, level: float, threshold: float = 0.002) -> bool:
        return abs(price - level) / level < threshold

class Strategy:
    """Proven Boom & Crash strategies"""
    
    @staticmethod
    def analyze_crash(df: pd.DataFrame) -> Tuple[Optional[str], float, List[str], str]:
        if len(df) < 100:
            return None, 0.0, ["Insufficient data"], ""
        
        current = df.iloc[-1]
        reasoning = []
        score = 0.0
        setup_type = ""
        
        levels = SupportResistance.find_levels(df, Config.LOOKBACK_PERIODS)
        trend = "up" if current['ema_20'] > current['ema_50'] else "down"
        
        # SETUP 1: SELL at resistance
        for resistance in levels.get('resistance', []):
            if SupportResistance.is_near_level(current['close'], resistance, Config.TOUCH_THRESHOLD):
                score += 50
                reasoning.append(f"✅ Near resistance: {resistance:.2f}")
                
                if current['rsi'] > 60:
                    score += 20
                    reasoning.append(f"✅ RSI overbought: {current['rsi']:.0f}")
                
                if trend == "down":
                    score += 20
                    reasoning.append("✅ Downtrend confirmed")
                
                setup_type = "resistance_sell"
                
                if score >= Config.MIN_CONFIDENCE:
                    return "SELL", score, reasoning, setup_type
        
        # SETUP 2: BUY at support
        if trend == "up":
            for support in levels.get('support', []):
                if SupportResistance.is_near_level(current['close'], support, Config.TOUCH_THRESHOLD):
                    score += 50
                    reasoning.append(f"✅ Near support: {support:.2f}")
                    
                    if current['rsi'] < 40:
                        score += 20
                        reasoning.append(f"✅ RSI oversold: {current['rsi']:.0f}")
                    
                    score += 20
                    reasoning.append("✅ Uptrend confirmed")
                    
                    setup_type = "support_buy"
                    
                    if score >= Config.MIN_CONFIDENCE:
                        return "BUY", score, reasoning, setup_type
        
        return None, score, reasoning, setup_type
    
    @staticmethod
    def analyze_boom(df: pd.DataFrame) -> Tuple[Optional[str], float, List[str], str]:
        if len(df) < 100:
            return None, 0.0, ["Insufficient data"], ""
        
        current = df.iloc[-1]
        reasoning = []
        score = 0.0
        setup_type = ""
        
        levels = SupportResistance.find_levels(df, Config.LOOKBACK_PERIODS)
        trend = "up" if current['ema_20'] > current['ema_50'] else "down"
        
        # SETUP 1: BUY at support
        for support in levels.get('support', []):
            if SupportResistance.is_near_level(current['close'], support, Config.TOUCH_THRESHOLD):
                score += 50
                reasoning.append(f"✅ Near support: {support:.2f}")
                
                if current['rsi'] < 40:
                    score += 20
                    reasoning.append(f"✅ RSI oversold: {current['rsi']:.0f}")
                
                if trend == "up":
                    score += 20
                    reasoning.append("✅ Uptrend confirmed")
                
                setup_type = "support_buy"
                
                if score >= Config.MIN_CONFIDENCE:
                    return "BUY", score, reasoning, setup_type
        
        # SETUP 2: SELL at resistance
        if trend == "down":
            for resistance in levels.get('resistance', []):
                if SupportResistance.is_near_level(current['close'], resistance, Config.TOUCH_THRESHOLD):
                    score += 50
                    reasoning.append(f"✅ Near resistance: {resistance:.2f}")
                    
                    if current['rsi'] > 60:
                        score += 20
                        reasoning.append(f"✅ RSI overbought: {current['rsi']:.0f}")
                    
                    score += 20
                    reasoning.append("✅ Downtrend confirmed")
                    
                    setup_type = "resistance_sell"
                    
                    if score >= Config.MIN_CONFIDENCE:
                        return "SELL", score, reasoning, setup_type
        
        return None, score, reasoning, setup_type
    
    @staticmethod
    def calculate_levels(df: pd.DataFrame, action: str) -> Dict:
        current = df.iloc[-1]
        price = current['close']
        
        pip_value = 0.01
        
        if action == "BUY":
            sl = price - (Config.STOP_LOSS_PIPS * pip_value)
            tp = price + (Config.SCALP_TARGET_PIPS * pip_value)
        else:
            sl = price + (Config.STOP_LOSS_PIPS * pip_value)
            tp = price - (Config.SCALP_TARGET_PIPS * pip_value)
        
        risk = abs(price - sl)
        reward = abs(tp - price)
        rr = reward / risk if risk > 0 else 0
        
        return {
            'entry': price,
            'sl': sl,
            'tp': tp,
            'rr': rr
        }

class RiskManager:
    def __init__(self):
        self.daily_trades = 0
        self.daily_loss = 0.0
        self.active_trades = 0
        self.consecutive_losses = 0
        self.last_reset = datetime.now(timezone.utc).date()
    
    def reset_daily(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset:
            self.daily_trades = 0
            self.daily_loss = 0.0
            self.last_reset = today
            logger.info("📅 Daily reset")
    
    def can_trade(self) -> Tuple[bool, str]:
        self.reset_daily()
        
        if self.daily_trades >= Config.MAX_DAILY_TRADES:
            return False, f"Daily limit: {Config.MAX_DAILY_TRADES} trades"
        
        if self.daily_loss >= Config.MAX_DAILY_LOSS:
            return False, f"Daily loss limit: {Config.MAX_DAILY_LOSS}%"
        
        if self.active_trades >= Config.MAX_CONCURRENT_TRADES:
            return False, f"Max concurrent trades: {Config.MAX_CONCURRENT_TRADES}"
        
        if self.consecutive_losses >= 3:
            return False, "3 consecutive losses - paused"
        
        return True, "OK"

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.enabled = token and chat_id
    
    def send_signal(self, signal: Signal) -> bool:
        if not self.enabled:
            logger.warning("⚠️  Telegram not configured")
            return False
        
        try:
            emoji = "💥" if "BOOM" in signal.symbol else "💫"
            action_emoji = "🔴 SHORT" if signal.action == "SELL" else "🟢 LONG"
            
            message = f"""
{emoji} <b>{signal.symbol}</b> - {signal.setup_type.upper()}

{action_emoji}
📊 Confidence: <b>{signal.confidence:.0f}%</b>
🎯 Setup: <b>{signal.setup_type.replace('_', ' ').title()}</b>

<b>TRADE DETAILS</b>
Entry: <code>{signal.entry_price:.2f}</code>
SL: <code>{signal.stop_loss:.2f}</code> ({Config.STOP_LOSS_PIPS} pips)
TP: <code>{signal.take_profit:.2f}</code> ({Config.SCALP_TARGET_PIPS} pips)

<b>REASONING</b>
{chr(10).join(['• ' + r for r in signal.reasoning])}

<b>EXECUTION</b>
• Enter NOW at market
• Set SL/TP immediately
• Don't hold through spikes
• Exit at TP (scalp quick profits)

<i>{signal.timestamp.strftime('%H:%M:%S UTC')}</i>
"""
            url = f"{self.base_url}/sendMessage"
            response = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False

class TradingBot:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.risk_manager = RiskManager()
        self.notifier = TelegramNotifier(Config.TELEGRAM_BOT_TOKEN, Config.CHAT_ID)
        self.processed_signals = deque(maxlen=20)
    
    def is_trading_time(self) -> Tuple[bool, str]:
        now = datetime.now(timezone.utc).time()
        
        if Config.OVERLAP_START <= now <= Config.OVERLAP_END:
            return True, "OVERLAP (BEST)"
        if Config.LONDON_START <= now <= Config.LONDON_END:
            return True, "LONDON"
        if Config.NY_START <= now <= Config.NY_END:
            return True, "NEW YORK"
        
        return True, "24/7"
    
    def run(self):
        logger.info("=" * 70)
        logger.info("🎯 BOOM & CRASH: CORRECTED VERSION")
        logger.info("=" * 70)
        
        in_session, session_name = self.is_trading_time()
        logger.info(f"📊 Session: {session_name}")
        logger.info(f"🕐 Time: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
        
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.warning(f"⛔ {reason}")
            return
        
        signals_found = 0
        
        for symbol in Config.SYMBOLS.keys():
            logger.info(f"\n{'='*50}")
            logger.info(f"Analyzing {symbol}...")
            logger.info(f"{'='*50}")
            
            try:
                df = self.fetcher.get_candles(symbol)
                
                if df is None or df.empty:
                    logger.warning(f"❌ No data for {symbol}")
                    continue
                
                df = Indicators.add_all(df)
                
                if df.empty:
                    logger.warning(f"❌ Empty after indicators")
                    continue
                
                current = df.iloc[-1]
                levels = SupportResistance.find_levels(df)
                
                logger.info(f"📊 Price: {current['close']:.2f}")
                logger.info(f"📊 RSI: {current['rsi']:.0f}")
                logger.info(f"📊 Trend: {'UP' if current['ema_20'] > current['ema_50'] else 'DOWN'}")
                logger.info(f"📊 Support: {[f'{s:.2f}' for s in levels['support']]}")
                logger.info(f"📊 Resistance: {[f'{r:.2f}' for r in levels['resistance']]}")
                
                if "CRASH" in symbol:
                    action, confidence, reasoning, setup_type = Strategy.analyze_crash(df)
                else:
                    action, confidence, reasoning, setup_type = Strategy.analyze_boom(df)
                
                if not action:
                    logger.info(f"❌ No setup found (Score: {confidence:.0f}%)")
                    continue
                
                logger.info(f"✅ Setup found: {action} ({setup_type}) @ {confidence:.0f}%")
                
                sig_key = f"{symbol}_{action}_{int(datetime.now().timestamp() / 1800)}"
                if sig_key in self.processed_signals:
                    logger.info(f"⚠️  Duplicate signal (30min cooldown)")
                    continue
                
                levels_dict = Strategy.calculate_levels(df, action)
                logger.info(f"📊 Entry: {levels_dict['entry']:.2f}, SL: {levels_dict['sl']:.2f}, TP: {levels_dict['tp']:.2f}")
                
                signal = Signal(
                    signal_id=f"{symbol}-{int(datetime.now().timestamp())}",
                    symbol=symbol,
                    action=action,
                    entry_price=levels_dict['entry'],
                    stop_loss=levels_dict['sl'],
                    take_profit=levels_dict['tp'],
                    confidence=confidence,
                    setup_type=setup_type,
                    timestamp=datetime.now(timezone.utc),
                    reasoning=reasoning
                )
                
                logger.info(f"🚀 SENDING SIGNAL: {symbol} {action}")
                if self.notifier.send_signal(signal):
                    self.processed_signals.append(sig_key)
                    self.risk_manager.daily_trades += 1
                    self.risk_manager.active_trades += 1
                    signals_found += 1
                    logger.info(f"✅ Signal sent! Daily trades: {self.risk_manager.daily_trades}/{Config.MAX_DAILY_TRADES}")
                else:
                    logger.error("❌ Failed to send signal")
                
            except Exception as e:
                logger.error(f"❌ Error analyzing {symbol}: {e}", exc_info=True)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 SCAN COMPLETE: {signals_found} signal(s) found")
        logger.info(f"{'='*70}\n")

if __name__ == "__main__":
    if not os.getenv('TELEGRAM_BOT_TOKEN') or not os.getenv('MAIN_CHAT_ID'):
        print("⚠️  WARNING: Telegram credentials not set")
        print("Set TELEGRAM_BOT_TOKEN and MAIN_CHAT_ID environment variables")
        print("Bot will run but won't send notifications\n")
    
    print("""
╔══════════════════════════════════════════════════════════╗
║      BOOM & CRASH: CORRECTED WORKING VERSION             ║
║      Symbol codes fixed: Using BOOM500/CRASH500          ║
╚══════════════════════════════════════════════════════════╝

✅ SYMBOL CODES CORRECTED:
   • Boom 300:  BOOM300N  (note the "N")
   • Crash 300: CRASH300N (note the "N")
   • Boom 500:  BOOM500   ✓ Currently active
   • Crash 500: CRASH500  ✓ Currently active

📊 CURRENT CONFIGURATION:
   • Trading: BOOM500 & CRASH500 (confirmed working)
   • Target: 20 pips | Stop: 40 pips
   • Risk: 0.5% per trade
   • Max trades: 5/day

💡 TO USE BOOM300N/CRASH300N INSTEAD:
   Edit Config.SYMBOLS in the script (line 43-47)

⏰ RECOMMENDED SCHEDULE:
   Run every 5-15 minutes during London/NY overlap (13:00-16:00 UTC)

🎯 NEXT STEPS:
   1. Test on DEMO account for 30 days
   2. Track all trades in a spreadsheet
   3. Only go live after proving profitability
   4. Never risk more than 0.5% per trade

Starting bot...
    """)
    
    bot = TradingBot()
    bot.run()