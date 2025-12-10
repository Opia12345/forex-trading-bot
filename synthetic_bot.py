"""
BOOM & CRASH: REALISTIC VERSION
NO STOP LOSSES - Manual exit strategy based on Deriv's actual mechanics
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
        logging.FileHandler('boom_crash_realistic.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    """Realistic configuration for Boom/Crash trading"""
    
    # ONLY Crash 500 and Boom 500
    SYMBOLS = {
        'CRASH_500': 'CRASH500',
        'BOOM_500': 'BOOM500',
    }
    
    # Trading sessions (UTC)
    LONDON_START = time(8, 0)
    LONDON_END = time(12, 0)
    NY_START = time(13, 0)
    NY_END = time(17, 0)
    OVERLAP_START = time(13, 0)
    OVERLAP_END = time(16, 0)
    
    # Risk Management
    RISK_PER_TRADE = 0.5  # For position sizing only
    MAX_DAILY_TRADES = 5
    MAX_DAILY_LOSS = 2.0
    MAX_CONCURRENT_TRADES = 1  # One at a time!
    
    # Strategy Parameters
    MIN_CONFIDENCE = 70
    
    # EXIT STRATEGY (since no SL/TP available)
    # Exit after X candles OR when condition is met
    MAX_HOLD_CANDLES = 5  # Exit after 5 minutes max (scalping)
    PROFIT_TARGET_PERCENT = 0.3  # Exit at 0.3% profit
    MAX_LOSS_PERCENT = 0.6  # Mental stop - exit at 0.6% loss
    
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
    exit_target: float  # Target price
    mental_stop: float  # Mental stop price
    confidence: float
    setup_type: str
    timestamp: datetime
    reasoning: List[str] = field(default_factory=list)
    hold_duration: str = "5 candles max"

class DataFetcher:
    """Data fetcher with correct symbol codes"""
    
    SYMBOL_MAP = {
        'BOOM_300': 'BOOM300N',
        'BOOM_500': 'BOOM500',
        'BOOM_600': 'BOOM600',
        'BOOM_900': 'BOOM900',
        'BOOM_1000': 'BOOM1000',
        'CRASH_300': 'CRASH300N',
        'CRASH_500': 'CRASH500',
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
    """Realistic Boom & Crash strategies - NO SL/TP"""
    
    @staticmethod
    def analyze_crash(df: pd.DataFrame) -> Tuple[Optional[str], float, List[str], str]:
        """
        CRASH: Expect sudden DROPS (spikes down)
        - SELL near resistance (ride the crash down)
        - BUY at support during uptrend (catch the bounce)
        """
        if len(df) < 100:
            return None, 0.0, ["Insufficient data"], ""
        
        current = df.iloc[-1]
        reasoning = []
        score = 0.0
        setup_type = ""
        
        levels = SupportResistance.find_levels(df, Config.LOOKBACK_PERIODS)
        trend = "up" if current['ema_20'] > current['ema_50'] else "down"
        
        # SETUP 1: SELL at resistance (crash likely)
        for resistance in levels.get('resistance', []):
            if SupportResistance.is_near_level(current['close'], resistance, Config.TOUCH_THRESHOLD):
                score += 50
                reasoning.append(f"✅ Price near resistance: {resistance:.2f}")
                
                if current['rsi'] > 65:
                    score += 20
                    reasoning.append(f"✅ RSI overbought: {current['rsi']:.0f} (crash likely)")
                
                if trend == "down":
                    score += 20
                    reasoning.append("✅ Downtrend - crashes more likely")
                
                setup_type = "resistance_sell"
                
                if score >= Config.MIN_CONFIDENCE:
                    return "SELL", score, reasoning, setup_type
        
        # SETUP 2: BUY at support in uptrend (bounce after crash)
        if trend == "up":
            for support in levels.get('support', []):
                if SupportResistance.is_near_level(current['close'], support, Config.TOUCH_THRESHOLD):
                    score += 50
                    reasoning.append(f"✅ Price near support: {support:.2f}")
                    
                    if current['rsi'] < 35:
                        score += 20
                        reasoning.append(f"✅ RSI oversold: {current['rsi']:.0f} (bounce likely)")
                    
                    score += 20
                    reasoning.append("✅ Uptrend - buy the dip after crash")
                    
                    setup_type = "support_buy"
                    
                    if score >= Config.MIN_CONFIDENCE:
                        return "BUY", score, reasoning, setup_type
        
        return None, score, reasoning, setup_type
    
    @staticmethod
    def analyze_boom(df: pd.DataFrame) -> Tuple[Optional[str], float, List[str], str]:
        """
        BOOM: Expect sudden SPIKES (up)
        - BUY at support (ride the boom up)
        - SELL at resistance during downtrend (catch the drop)
        """
        if len(df) < 100:
            return None, 0.0, ["Insufficient data"], ""
        
        current = df.iloc[-1]
        reasoning = []
        score = 0.0
        setup_type = ""
        
        levels = SupportResistance.find_levels(df, Config.LOOKBACK_PERIODS)
        trend = "up" if current['ema_20'] > current['ema_50'] else "down"
        
        # SETUP 1: BUY at support (boom likely)
        for support in levels.get('support', []):
            if SupportResistance.is_near_level(current['close'], support, Config.TOUCH_THRESHOLD):
                score += 50
                reasoning.append(f"✅ Price near support: {support:.2f}")
                
                if current['rsi'] < 35:
                    score += 20
                    reasoning.append(f"✅ RSI oversold: {current['rsi']:.0f} (boom likely)")
                
                if trend == "up":
                    score += 20
                    reasoning.append("✅ Uptrend - booms more likely")
                
                setup_type = "support_buy"
                
                if score >= Config.MIN_CONFIDENCE:
                    return "BUY", score, reasoning, setup_type
        
        # SETUP 2: SELL at resistance in downtrend
        if trend == "down":
            for resistance in levels.get('resistance', []):
                if SupportResistance.is_near_level(current['close'], resistance, Config.TOUCH_THRESHOLD):
                    score += 50
                    reasoning.append(f"✅ Price near resistance: {resistance:.2f}")
                    
                    if current['rsi'] > 65:
                        score += 20
                        reasoning.append(f"✅ RSI overbought: {current['rsi']:.0f}")
                    
                    score += 20
                    reasoning.append("✅ Downtrend - sell the rally")
                    
                    setup_type = "resistance_sell"
                    
                    if score >= Config.MIN_CONFIDENCE:
                        return "SELL", score, reasoning, setup_type
        
        return None, score, reasoning, setup_type
    
    @staticmethod
    def calculate_exit_levels(df: pd.DataFrame, action: str) -> Dict:
        """Calculate manual exit targets (no SL/TP available)"""
        current = df.iloc[-1]
        price = current['close']
        
        if action == "BUY":
            exit_target = price * (1 + Config.PROFIT_TARGET_PERCENT / 100)
            mental_stop = price * (1 - Config.MAX_LOSS_PERCENT / 100)
        else:  # SELL
            exit_target = price * (1 - Config.PROFIT_TARGET_PERCENT / 100)
            mental_stop = price * (1 + Config.MAX_LOSS_PERCENT / 100)
        
        return {
            'entry': price,
            'exit_target': exit_target,
            'mental_stop': mental_stop,
            'profit_percent': Config.PROFIT_TARGET_PERCENT,
            'loss_percent': Config.MAX_LOSS_PERCENT
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
            action_emoji = "🔴 SELL" if signal.action == "SELL" else "🟢 BUY"
            
            message = f"""
{emoji} <b>{signal.symbol}</b> - {signal.setup_type.upper()}

{action_emoji}
📊 Confidence: <b>{signal.confidence:.0f}%</b>

<b>⚠️ NO STOP LOSS AVAILABLE - MANUAL EXIT REQUIRED</b>

<b>ENTRY & TARGETS</b>
Entry: <code>{signal.entry_price:.2f}</code>
Target: <code>{signal.exit_target:.2f}</code> (+{Config.PROFIT_TARGET_PERCENT}%)
Stop: <code>{signal.mental_stop:.2f}</code> (-{Config.MAX_LOSS_PERCENT}%)

<b>EXIT STRATEGY</b>
✅ Exit at target ({Config.PROFIT_TARGET_PERCENT}% profit)
✅ Exit after {signal.hold_duration}
⛔ Exit if price hits mental stop (-{Config.MAX_LOSS_PERCENT}%)
⛔ Exit if setup invalidated

<b>REASONING</b>
{chr(10).join(['• ' + r for r in signal.reasoning])}

<b>⚠️ CRITICAL REMINDERS</b>
• Boom/Crash have NO automated SL/TP
• You MUST watch and exit manually
• Don't hold through adverse spikes
• Set phone alerts for targets
• Exit quickly on {signal.hold_duration}

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
        logger.info("🎯 BOOM & CRASH: REALISTIC VERSION (NO SL/TP)")
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
                
                levels_dict = Strategy.calculate_exit_levels(df, action)
                logger.info(f"📊 Entry: {levels_dict['entry']:.2f}")
                logger.info(f"📊 Target: {levels_dict['exit_target']:.2f} (+{levels_dict['profit_percent']}%)")
                logger.info(f"📊 Mental Stop: {levels_dict['mental_stop']:.2f} (-{levels_dict['loss_percent']}%)")
                
                signal = Signal(
                    signal_id=f"{symbol}-{int(datetime.now().timestamp())}",
                    symbol=symbol,
                    action=action,
                    entry_price=levels_dict['entry'],
                    exit_target=levels_dict['exit_target'],
                    mental_stop=levels_dict['mental_stop'],
                    confidence=confidence,
                    setup_type=setup_type,
                    timestamp=datetime.now(timezone.utc),
                    reasoning=reasoning,
                    hold_duration=f"{Config.MAX_HOLD_CANDLES} minutes"
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
║   BOOM & CRASH: REALISTIC VERSION                        ║
║   CRASH 500 & BOOM 500 ONLY                              ║
║   NO STOP LOSSES - Manual Exit Strategy                  ║
╚══════════════════════════════════════════════════════════╝

⚠️  CRITICAL: BOOM/CRASH INDICES HAVE NO SL/TP SUPPORT!

📊 TRADING ONLY:
   • Crash 500 Index (CRASH500)
   • Boom 500 Index (BOOM500)

📊 EXIT STRATEGY:
   • Target: +0.3% profit (quick scalp)
   • Mental Stop: -0.6% loss (cut losses fast)
   • Time limit: Exit after 5 minutes max
   • Watch manually - set phone alerts!

🎯 HOW TO TRADE:
   1. Get signal notification
   2. Enter immediately at market
   3. Watch price movement
   4. Exit at target OR after 5 minutes
   5. If losing -0.6%, exit immediately!

⚠️  YOU MUST WATCH EVERY TRADE:
   • No automated exits available
   • Spikes can happen suddenly
   • Set price alerts on your phone
   • Don't leave trades unattended
   • One trade at a time only

💡 RECOMMENDED PLATFORMS:
   • Deriv MT5 mobile (for alerts)
   • Deriv DTrader (for quick exits)
   • Keep browser/app open during trades

⏰ SCHEDULE:
   Run every 5 minutes during London/NY overlap

Starting bot...
    """)
    
    bot = TradingBot()
    bot.run()