"""
TELEGRAM NOTIFIER - Fixed Implementation
Sends detailed signals to main chat and simple alerts to simple chat
"""

import requests
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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
    
    def send_performance_update(self, stats: dict) -> bool:
        """Send performance statistics update to main chat"""
        
        message = f"""
📊 <b>PERFORMANCE UPDATE</b>

<b>Last 30 Days:</b>
• Total Trades: {stats['total_trades']}
• Wins: {stats['wins']} | Losses: {stats['losses']}
• Win Rate: {stats['win_rate']:.1f}%
• Total P&L: ${stats['total_pnl_usd']:.2f}
• Avg Confidence: {stats['avg_confidence']:.1f}%
• Expectancy: {stats['expectancy']:.2f} pips

• Avg Win: {stats['avg_win_pips']:.1f} pips
• Avg Loss: {stats['avg_loss_pips']:.1f} pips
"""
        
        return self._send_message(self.main_chat, message)
    
    def send_error_alert(self, error_message: str) -> bool:
        """Send error alert to main chat"""
        message = f"⚠️ <b>BOT ERROR</b>\n\n<code>{error_message}</code>"
        return self._send_message(self.main_chat, message)
    
    def send_news_alert(self, news_event: str) -> bool:
        """Send high-impact news alert to both chats"""
        message = f"""
⚠️ <b>HIGH IMPACT NEWS ALERT</b>

{news_event}

Trading may be restricted during this period.
"""
        
        main_sent = self._send_message(self.main_chat, message)
        simple_sent = self._send_message(self.simple_chat, message)
        
        return main_sent or simple_sent