"""
TRADE JOURNAL SYSTEM
Tracks all signals and outcomes for performance analysis
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd
from pathlib import Path

@dataclass
class TradeRecord:
    """Individual trade record"""
    signal_id: str
    symbol: str
    action: str
    signal_type: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    position_size: float
    confidence: float
    timestamp: str
    
    # Outcome fields (filled later)
    outcome: Optional[str] = None  # "WIN", "LOSS", "BREAKEVEN", "PARTIAL"
    exit_price: Optional[float] = None
    pnl_pips: Optional[float] = None
    pnl_usd: Optional[float] = None
    exit_timestamp: Optional[str] = None
    notes: Optional[str] = None


class TradeJournal:
    """SQLite-based trade journal for performance tracking"""
    
    def __init__(self, db_path: str = "trade_journal.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                signal_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit_1 REAL NOT NULL,
                take_profit_2 REAL NOT NULL,
                take_profit_3 REAL NOT NULL,
                position_size REAL NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                outcome TEXT,
                exit_price REAL,
                pnl_pips REAL,
                pnl_usd REAL,
                exit_timestamp TEXT,
                notes TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                total_signals INTEGER,
                wins INTEGER,
                losses INTEGER,
                breakeven INTEGER,
                win_rate REAL,
                total_pnl_usd REAL,
                avg_confidence REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_signal(self, trade: TradeRecord):
        """Log a new trading signal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.signal_id, trade.symbol, trade.action, trade.signal_type,
            trade.entry_price, trade.stop_loss, trade.take_profit_1,
            trade.take_profit_2, trade.take_profit_3, trade.position_size,
            trade.confidence, trade.timestamp, trade.outcome, trade.exit_price,
            trade.pnl_pips, trade.pnl_usd, trade.exit_timestamp, trade.notes
        ))
        
        conn.commit()
        conn.close()
    
    def update_trade_outcome(self, signal_id: str, outcome: str, exit_price: float,
                            pnl_pips: float, pnl_usd: float, notes: str = ""):
        """Update trade with outcome"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE trades 
            SET outcome = ?, exit_price = ?, pnl_pips = ?, pnl_usd = ?, 
                exit_timestamp = ?, notes = ?
            WHERE signal_id = ?
        """, (outcome, exit_price, pnl_pips, pnl_usd, 
              datetime.now().isoformat(), notes, signal_id))
        
        conn.commit()
        conn.close()
    
    def get_performance_stats(self, days: int = 30) -> Dict:
        """Get performance statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all completed trades
        cursor.execute("""
            SELECT outcome, pnl_pips, pnl_usd, confidence, signal_type
            FROM trades
            WHERE outcome IS NOT NULL
            AND timestamp > datetime('now', '-' || ? || ' days')
        """, (days,))
        
        trades = cursor.fetchall()
        conn.close()
        
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_win_pips": 0.0,
                "avg_loss_pips": 0.0,
                "total_pnl_usd": 0.0,
                "avg_confidence": 0.0
            }
        
        wins = [t for t in trades if t[0] == "WIN"]
        losses = [t for t in trades if t[0] == "LOSS"]
        
        win_rate = (len(wins) / len(trades)) * 100 if trades else 0
        avg_win_pips = sum(t[1] for t in wins) / len(wins) if wins else 0
        avg_loss_pips = sum(t[1] for t in losses) / len(losses) if losses else 0
        total_pnl = sum(t[2] for t in trades)
        avg_confidence = sum(t[3] for t in trades) / len(trades)
        
        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "avg_win_pips": avg_win_pips,
            "avg_loss_pips": avg_loss_pips,
            "total_pnl_usd": total_pnl,
            "avg_confidence": avg_confidence,
            "expectancy": (avg_win_pips * (win_rate/100)) + (avg_loss_pips * ((100-win_rate)/100))
        }
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Get recent trades"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        trades = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return trades
    
    def export_to_csv(self, filepath: str = "trade_history.csv"):
        """Export all trades to CSV"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC", conn)
        conn.close()
        
        df.to_csv(filepath, index=False)
        return filepath
