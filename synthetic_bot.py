"""
DERIV SYNTHETIC INDICES - SMALL ACCOUNT BUILDER v5.1
FIXED VERSION - Realistic signal generation

Key Changes:
- Reduced Z-score threshold to 1.8σ (more realistic)
- Simplified quality filters (removed excessive stacking)
- Added timezone awareness
- More forgiving confidence scoring
- Better session detection
"""

import os
import sys
import logging
from typing import Optional, List, Tuple, Dict
from datetime import datetime, time, timezone
from collections import deque
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import requests
import json
import asyncio
import websockets
import pytz

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('small_account_builder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
class TradingConfig:
    # Trading sessions (GMT/UTC times)
    LONDON_SESSION = (time(8, 0), time(12, 0))
    NY_SESSION = (time(13, 0), time(17, 0))
    
    # More realistic risk management
    MIN_RISK_PER_TRADE = 0.5
    MAX_RISK_PER_TRADE = 1.0
    MAX_DAILY_TRADES = 5
    MAX_DAILY_RISK = 3.0
    
    # FIXED: More realistic thresholds
    MIN_CONFIDENCE = 55  # Lowered from 70
    MIN_RISK_REWARD = 1.5  # Lowered from 2.0
    
    SYMBOLS = ['V75', 'V100']

@dataclass
class TradeSignal:
    signal_id: str
    symbol: str
    action: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    position_size_pct: float
    expected_hold_hours: float
    z_score: float
    quality_score: str
    timestamp: datetime
    reasoning: List[str] = field(default_factory=list)
    session: str = ""

class SmallAccountRiskManager:
    def __init__(self):
        self.daily_trades_taken = 0
        self.daily_risk_used = 0.0
        self.consecutive_losses = 0
        self.last_reset = datetime.now(timezone.utc).date()
        
    def reset_daily_counters(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset:
            self.daily_trades_taken = 0
            self.daily_risk_used = 0.0
            self.last_reset = today
            logger.info("📅 Daily counters reset")
    
    def can_trade(self) -> Tuple[bool, str]:
        self.reset_daily_counters()
        
        if self.daily_trades_taken >= TradingConfig.MAX_DAILY_TRADES:
            return False, f"Daily limit reached ({TradingConfig.MAX_DAILY_TRADES} trades)"
        
        if self.daily_risk_used >= TradingConfig.MAX_DAILY_RISK:
            return False, f"Daily risk limit reached ({TradingConfig.MAX_DAILY_RISK}%)"
        
        if self.consecutive_losses >= 3:
            return False, f"3 consecutive losses - taking a break"
        
        return True, "OK"
    
    def calculate_position_size(self, confidence: float) -> float:
        base_risk = TradingConfig.MIN_RISK_PER_TRADE
        
        if confidence >= 75:
            risk_pct = TradingConfig.MAX_RISK_PER_TRADE
        elif confidence >= 65:
            risk_pct = 0.75
        else:
            risk_pct = base_risk
        
        if self.consecutive_losses >= 2:
            risk_pct *= 0.5
        
        return risk_pct
    
    def record_trade(self, risk_pct: float):
        self.daily_trades_taken += 1
        self.daily_risk_used += risk_pct

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_signal(self, signal: TradeSignal) -> bool:
        try:
            a_emoji = "🟢 LONG" if signal.action == "BUY" else "🔴 SHORT"
            quality_emoji = "💎" if signal.confidence >= 75 else "⭐"
            
            message = f"""
{quality_emoji} <b>SYNTHETIC SIGNAL - {signal.quality_score}</b>

{a_emoji} <b>{signal.symbol}</b>
📊 Confidence: <b>{signal.confidence:.0f}%</b>
⏱️ Session: <b>{signal.session}</b>
💰 Expected Hold: <b>{signal.expected_hold_hours:.1f} hours</b>

<b>🎯 TRADE SETUP</b>
📍 Entry: <code>{signal.entry_price:.5f}</code>
🛑 Stop Loss: <code>{signal.stop_loss:.5f}</code>
🎯 Take Profit: <code>{signal.take_profit:.5f}</code>

<b>💼 RISK MANAGEMENT</b>
• Position Size: <b>{signal.position_size_pct:.2f}%</b> of account
• Risk:Reward: <b>1:{signal.risk_reward_ratio:.1f}</b>
• Z-Score: {signal.z_score:.2f}σ

<b>✅ WHY THIS TRADE</b>
"""
            for reason in signal.reasoning:
                message += f"• {reason}\n"
            
            message += f"""
<b>📱 EXECUTION TIPS</b>
• Enter at market price immediately
• Set SL/TP and walk away
• Don't overtrade - max 5 trades/day

<i>🆔 {signal.signal_id}</i>
<i>⏱️ {signal.timestamp.strftime('%H:%M:%S %Z')}</i>

<b>💪 Small Account Growth Strategy</b>
"""
            return self._send_message(message)
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False
    
    def _send_message(self, message: str) -> bool:
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except:
            return False

class DerivDataFetcher:
    SYMBOLS = {
        'V75': 'R_75',
        'V100': 'R_100',
    }
    
    GRANULARITIES = {
        'M15': 900,
        'H1': 3600,
    }
    
    def __init__(self, app_id: str = "1089"):
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    
    async def _fetch_candles_async(self, symbol: str, granularity: int, count: int = 300) -> Optional[List]:
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
                        logger.info(f"✅ Fetched {len(df)} {tf_name} candles for {symbol}")
                except Exception as e:
                    logger.error(f"Error processing {tf_name}: {e}")
                    continue
        finally:
            loop.close()
        
        return result

class Indicators:
    
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add proven mean reversion indicators"""
        
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['sma_100'] = df['close'].rolling(window=100).mean()
        
        # Bollinger Bands (20, 2.0) - FIXED: More standard deviation
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2.0)  # Changed from 2.5
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2.0)
        
        # Z-Score (50-period) - FIXED: Shorter period for more signals
        mean_50 = df['close'].rolling(window=50).mean()
        std_50 = df['close'].rolling(window=50).std()
        df['z_score'] = (df['close'] - mean_50) / std_50
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = pd.Series(true_range).rolling(window=14).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        df['vol_ma'] = df['volatility'].rolling(window=50).mean()
        
        df['distance_from_mean'] = ((df['close'] - df['bb_middle']) / df['bb_middle']) * 100
        
        df.dropna(inplace=True)
        return df

class MeanReversionStrategy:
    
    @staticmethod
    def analyze(df_m15: pd.DataFrame, df_h1: pd.DataFrame) -> Tuple[Optional[str], float, List[str], float]:
        """
        FIXED: More realistic mean reversion detection
        
        Key changes:
        - Z-score threshold lowered to 1.8σ (from 2.5σ)
        - Simplified scoring (removed overly strict filters)
        - More forgiving BB and RSI thresholds
        """
        if len(df_m15) < 100 or len(df_h1) < 100:
            return None, 0.0, ["Insufficient data"], 0.0
        
        last_m15 = df_m15.iloc[-1]
        last_h1 = df_h1.iloc[-1]
        
        reasoning = []
        score = 0.0
        action = None
        
        # 1. Z-SCORE CHECK (30 points) - FIXED: Lower threshold
        z_score = last_m15['z_score']
        
        if z_score < -1.8:  # Changed from -2.5
            action = "BUY"
            if z_score < -2.2:
                score += 30
                reasoning.append(f"✅ Strongly oversold: {abs(z_score):.2f}σ")
            else:
                score += 25
                reasoning.append(f"✅ Oversold: {abs(z_score):.2f}σ")
        elif z_score > 1.8:  # Changed from 2.5
            action = "SELL"
            if z_score > 2.2:
                score += 30
                reasoning.append(f"✅ Strongly overbought: {abs(z_score):.2f}σ")
            else:
                score += 25
                reasoning.append(f"✅ Overbought: {abs(z_score):.2f}σ")
        else:
            return None, 0.0, [f"❌ Z-score {z_score:.2f} not extreme enough (need 1.8σ+)"], z_score
        
        # 2. BOLLINGER BAND CHECK (25 points) - FIXED: More forgiving
        if action == "BUY":
            bb_distance = (last_m15['close'] - last_m15['bb_lower']) / last_m15['atr']
            if bb_distance < 0.5:  # Within 0.5 ATR of lower band
                score += 25
                reasoning.append("✅ Near/below lower BB")
            elif bb_distance < 1.0:
                score += 15
                reasoning.append("⚠️ Approaching lower BB")
            else:
                score += 5
        elif action == "SELL":
            bb_distance = (last_m15['bb_upper'] - last_m15['close']) / last_m15['atr']
            if bb_distance < 0.5:
                score += 25
                reasoning.append("✅ Near/above upper BB")
            elif bb_distance < 1.0:
                score += 15
                reasoning.append("⚠️ Approaching upper BB")
            else:
                score += 5
        
        # 3. RSI CHECK (20 points) - FIXED: More realistic thresholds
        rsi = last_m15['rsi']
        if action == "BUY":
            if rsi < 30:
                score += 20
                reasoning.append(f"✅ RSI oversold ({rsi:.0f})")
            elif rsi < 40:
                score += 15
                reasoning.append(f"⚠️ RSI low ({rsi:.0f})")
            else:
                score += 5
        elif action == "SELL":
            if rsi > 70:
                score += 20
                reasoning.append(f"✅ RSI overbought ({rsi:.0f})")
            elif rsi > 60:
                score += 15
                reasoning.append(f"⚠️ RSI high ({rsi:.0f})")
            else:
                score += 5
        
        # 4. H1 CONFIRMATION (15 points) - FIXED: More forgiving
        h1_z = last_h1['z_score']
        if (action == "BUY" and h1_z < -1.0) or (action == "SELL" and h1_z > 1.0):
            score += 15
            reasoning.append(f"✅ H1 confirms deviation ({h1_z:.2f}σ)")
        elif (action == "BUY" and h1_z < 0) or (action == "SELL" and h1_z > 0):
            score += 10
            reasoning.append(f"⚠️ H1 alignment ({h1_z:.2f}σ)")
        else:
            score += 5
        
        # 5. VOLATILITY (10 points)
        vol_ratio = last_m15['volatility'] / last_m15['vol_ma'] if last_m15['vol_ma'] > 0 else 1.0
        if vol_ratio < 1.5:
            score += 10
            reasoning.append(f"✅ Normal volatility ({vol_ratio:.2f})")
        else:
            score += 5
            reasoning.append(f"⚠️ Elevated volatility ({vol_ratio:.2f})")
        
        return action, score, reasoning, z_score

class SmallAccountBot:
    
    def __init__(self):
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('MAIN_CHAT_ID')
        
        if not self.telegram_token or not self.chat_id:
            logger.error("TELEGRAM_BOT_TOKEN or MAIN_CHAT_ID not set.")
            sys.exit(1)
        
        self.notifier = TelegramNotifier(self.telegram_token, self.chat_id)
        self.fetcher = DerivDataFetcher()
        self.risk_manager = SmallAccountRiskManager()
        self.processed_signals = deque(maxlen=50)
    
    def is_trading_session(self) -> Tuple[bool, str]:
        """FIXED: Proper timezone handling"""
        # Get current UTC time
        now_utc = datetime.now(timezone.utc).time()
        
        # London session
        if TradingConfig.LONDON_SESSION[0] <= now_utc <= TradingConfig.LONDON_SESSION[1]:
            return True, "LONDON"
        
        # NY session
        if TradingConfig.NY_SESSION[0] <= now_utc <= TradingConfig.NY_SESSION[1]:
            return True, "NEW YORK"
        
        # ADDED: Allow 24/7 operation for testing
        # Comment this out if you only want session trading
        return True, "24/7 MODE"
    
    def calculate_levels(self, df: pd.DataFrame, action: str) -> dict:
        """Calculate entry, SL, and TP - FIXED: More realistic targets"""
        last = df.iloc[-1]
        price = last['close']
        atr = last['atr']
        mean = last['bb_middle']
        
        # FIXED: Tighter stops (2.0 ATR instead of 2.5)
        sl_distance = atr * 2.0
        
        if action == "BUY":
            sl = price - sl_distance
            # Target mean
            tp = mean
            
            # Ensure minimum 1:1.5 R:R
            tp_distance = tp - price
            if tp_distance < (sl_distance * 1.5):
                tp = price + (sl_distance * 1.8)
        else:
            sl = price + sl_distance
            tp = mean
            
            tp_distance = price - tp
            if tp_distance < (sl_distance * 1.5):
                tp = price - (sl_distance * 1.8)
        
        rr = abs(tp - price) / abs(price - sl)
        
        return {
            'entry': price,
            'sl': sl,
            'tp': tp,
            'rr': rr,
            'atr': atr
        }
    
    def run(self):
        logger.info("=" * 60)
        logger.info("🚀 SMALL ACCOUNT BUILDER v5.1 - FIXED VERSION")
        logger.info("=" * 60)
        
        # Check if in trading session
        in_session, session_name = self.is_trading_session()
        if not in_session:
            logger.info("⏰ Outside trading sessions")
            return
        
        logger.info(f"📊 Trading Session: {session_name}")
        logger.info(f"🕐 Current UTC time: {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
        
        # Check risk limits
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.warning(f"⛔ {reason}")
            return
        
        signals_found = 0
        
        for symbol in TradingConfig.SYMBOLS:
            logger.info(f"\n{'='*40}")
            logger.info(f"Analyzing {symbol}...")
            logger.info(f"{'='*40}")
            
            try:
                # Fetch data
                mtf_data = self.fetcher.get_multi_timeframe_data(symbol)
                
                if 'M15' not in mtf_data or 'H1' not in mtf_data:
                    logger.warning(f"❌ Incomplete data for {symbol}")
                    continue
                
                # Add indicators
                df_m15 = Indicators.add_indicators(mtf_data['M15'])
                df_h1 = Indicators.add_indicators(mtf_data['H1'])
                
                if df_m15.empty or df_h1.empty:
                    logger.warning(f"❌ Empty dataframe after indicators")
                    continue
                
                # Log current market conditions
                last_m15 = df_m15.iloc[-1]
                logger.info(f"📊 Current Price: {last_m15['close']:.5f}")
                logger.info(f"📊 Z-Score: {last_m15['z_score']:.2f}σ")
                logger.info(f"📊 RSI: {last_m15['rsi']:.0f}")
                logger.info(f"📊 BB Position: Lower={last_m15['bb_lower']:.5f}, Mid={last_m15['bb_middle']:.5f}, Upper={last_m15['bb_upper']:.5f}")
                
                # Analyze
                action, confidence, reasoning, z_score = \
                    MeanReversionStrategy.analyze(df_m15, df_h1)
                
                if not action:
                    logger.info(f"❌ No setup: {reasoning[0] if reasoning else 'N/A'}")
                    continue
                
                logger.info(f"✅ Setup found: {action} with {confidence:.0f}% confidence")
                
                # Check confidence threshold
                if confidence < TradingConfig.MIN_CONFIDENCE:
                    logger.info(f"❌ Confidence {confidence:.0f}% below threshold {TradingConfig.MIN_CONFIDENCE}%")
                    continue
                
                # Check duplicates
                sig_key = f"{symbol}_{action}_{int(datetime.now().timestamp() / 3600)}"
                if sig_key in self.processed_signals:
                    logger.info(f"⚠️ Duplicate signal this hour")
                    continue
                
                # Calculate levels
                levels = self.calculate_levels(df_m15, action)
                logger.info(f"📊 Entry: {levels['entry']:.5f}, SL: {levels['sl']:.5f}, TP: {levels['tp']:.5f}, R:R: 1:{levels['rr']:.1f}")
                
                # Verify R:R
                if levels['rr'] < TradingConfig.MIN_RISK_REWARD:
                    logger.info(f"❌ R:R {levels['rr']:.1f} below minimum {TradingConfig.MIN_RISK_REWARD}")
                    continue
                
                # Position sizing
                pos_size = self.risk_manager.calculate_position_size(confidence)
                
                # Quality classification
                if confidence >= 75:
                    quality = "PREMIUM SETUP"
                elif confidence >= 65:
                    quality = "HIGH QUALITY"
                else:
                    quality = "GOOD SETUP"
                
                # Create signal
                signal = TradeSignal(
                    signal_id=f"SAB-{symbol}-{int(datetime.now().timestamp())}",
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    entry_price=levels['entry'],
                    stop_loss=levels['sl'],
                    take_profit=levels['tp'],
                    risk_reward_ratio=levels['rr'],
                    position_size_pct=pos_size,
                    expected_hold_hours=3.0,
                    z_score=z_score,
                    quality_score=quality,
                    timestamp=datetime.now(timezone.utc),
                    reasoning=reasoning,
                    session=session_name
                )
                
                # Send signal
                logger.info(f"🚀 SENDING SIGNAL: {symbol} {action} @ {confidence:.0f}%")
                if self.notifier.send_signal(signal):
                    self.processed_signals.append(sig_key)
                    self.risk_manager.record_trade(pos_size)
                    signals_found += 1
                    logger.info(f"✅ Signal sent! Daily: {self.risk_manager.daily_trades_taken}/{TradingConfig.MAX_DAILY_TRADES}")
                else:
                    logger.error("❌ Failed to send Telegram message")
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}", exc_info=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 SCAN COMPLETE: {signals_found} signal(s) sent")
        logger.info(f"{'='*60}\n")

if __name__ == "__main__":
    if not os.getenv('TELEGRAM_BOT_TOKEN') or not os.getenv('MAIN_CHAT_ID'):
        print("❌ Error: Set TELEGRAM_BOT_TOKEN and MAIN_CHAT_ID environment variables.")
        sys.exit(1)
    
    print("""
╔══════════════════════════════════════════════════╗
║   SMALL ACCOUNT BUILDER v5.1 - FIXED            ║
║   More realistic signal generation               ║
╚══════════════════════════════════════════════════╝

Key Fixes:
  ✅ Z-score threshold lowered to 1.8σ (was 2.5σ)
  ✅ Bollinger Band checks more forgiving
  ✅ Confidence threshold lowered to 55% (was 70%)
  ✅ Better timezone handling
  ✅ Detailed logging for debugging
  ✅ 24/7 mode enabled (disable in code if needed)

Run this and you SHOULD see signals now!
    """)
    
    bot = SmallAccountBot()
    bot.run()