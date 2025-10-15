"""
POSITION MANAGER
Handles trailing stops, partial exits, and dynamic stop management
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    """Current state of an open position"""
    signal_id: str
    symbol: str
    action: str
    entry_price: float
    current_price: float
    stop_loss: float
    breakeven_price: float
    tp1: float
    tp2: float
    tp3: float
    position_size: float
    
    # Position management state
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    moved_to_breakeven: bool = False
    trailing_active: bool = False
    current_position_size: float = None
    
    def __post_init__(self):
        if self.current_position_size is None:
            self.current_position_size = self.position_size


class PositionManager:
    """
    Manages open positions with dynamic stop management
    
    Features:
    - Move SL to breakeven when TP1 hits
    - Partial exits at each TP level
    - Trailing stops after TP2
    - Risk-free trading after breakeven
    """
    
    def __init__(self, trailing_distance_atr: float = 1.5):
        self.trailing_distance_atr = trailing_distance_atr
        self.open_positions: List[PositionState] = []
    
    def add_position(self, position: PositionState):
        """Add a new position to manage"""
        self.open_positions.append(position)
        logger.info(f"📊 Managing position: {position.signal_id}")
    
    def update_position(self, signal_id: str, current_price: float, atr: float) -> List[str]:
        """
        Update position with current price and return actions to take
        
        Returns list of actions: ["MOVE_SL_BREAKEVEN", "CLOSE_50%", "TRAIL_STOP", etc.]
        """
        position = self._find_position(signal_id)
        if not position:
            return []
        
        position.current_price = current_price
        actions = []
        
        # Check if we should close the position (SL or TP3 hit)
        if self._check_stop_loss_hit(position):
            actions.append("CLOSE_ALL_STOP_LOSS")
            self._remove_position(signal_id)
            return actions
        
        if self._check_tp3_hit(position):
            actions.append("CLOSE_ALL_TP3")
            self._remove_position(signal_id)
            return actions
        
        # Check TP levels and manage position
        if not position.tp1_hit and self._check_tp1_hit(position):
            position.tp1_hit = True
            actions.append("CLOSE_50%_TP1")
            position.current_position_size *= 0.5
            logger.info(f"✅ TP1 hit for {signal_id} - Closing 50%")
        
        if position.tp1_hit and not position.moved_to_breakeven:
            position.stop_loss = position.breakeven_price
            position.moved_to_breakeven = True
            actions.append("MOVE_SL_BREAKEVEN")
            logger.info(f"🛡️ Moved SL to breakeven for {signal_id}")
        
        if not position.tp2_hit and self._check_tp2_hit(position):
            position.tp2_hit = True
            actions.append("CLOSE_30%_TP2")
            position.current_position_size *= 0.6  # 30% of remaining 50%
            position.trailing_active = True
            logger.info(f"✅ TP2 hit for {signal_id} - Closing 30%, activating trail")
        
        # Trailing stop logic (after TP2)
        if position.trailing_active:
            new_sl = self._calculate_trailing_stop(position, atr)
            if new_sl != position.stop_loss:
                position.stop_loss = new_sl
                actions.append(f"TRAIL_STOP_TO_{new_sl:.5f}")
                logger.info(f"📈 Trailing stop updated for {signal_id}: {new_sl:.5f}")
        
        return actions
    
    def _find_position(self, signal_id: str) -> Optional[PositionState]:
        """Find position by signal ID"""
        for pos in self.open_positions:
            if pos.signal_id == signal_id:
                return pos
        return None
    
    def _remove_position(self, signal_id: str):
        """Remove closed position"""
        self.open_positions = [p for p in self.open_positions if p.signal_id != signal_id]
        logger.info(f"🔒 Position closed: {signal_id}")
    
    def _check_stop_loss_hit(self, position: PositionState) -> bool:
        """Check if stop loss was hit"""
        if position.action == "BUY":
            return position.current_price <= position.stop_loss
        else:
            return position.current_price >= position.stop_loss
    
    def _check_tp1_hit(self, position: PositionState) -> bool:
        """Check if TP1 was hit"""
        if position.action == "BUY":
            return position.current_price >= position.tp1
        else:
            return position.current_price <= position.tp1
    
    def _check_tp2_hit(self, position: PositionState) -> bool:
        """Check if TP2 was hit"""
        if position.action == "BUY":
            return position.current_price >= position.tp2
        else:
            return position.current_price <= position.tp2
    
    def _check_tp3_hit(self, position: PositionState) -> bool:
        """Check if TP3 was hit"""
        if position.action == "BUY":
            return position.current_price >= position.tp3
        else:
            return position.current_price <= position.tp3
    
    def _calculate_trailing_stop(self, position: PositionState, atr: float) -> float:
        """Calculate trailing stop level"""
        trailing_distance = atr * self.trailing_distance_atr
        
        if position.action == "BUY":
            # Trail below current price
            new_sl = position.current_price - trailing_distance
            # Only move SL up, never down
            return max(new_sl, position.stop_loss)
        else:
            # Trail above current price
            new_sl = position.current_price + trailing_distance
            # Only move SL down, never up
            return min(new_sl, position.stop_loss)
    
    def get_position_summary(self, signal_id: str) -> Optional[Dict]:
        """Get current position summary"""
        position = self._find_position(signal_id)
        if not position:
            return None
        
        pnl_pips = self._calculate_pnl_pips(position)
        
        return {
            "signal_id": position.signal_id,
            "symbol": position.symbol,
            "action": position.action,
            "entry_price": position.entry_price,
            "current_price": position.current_price,
            "stop_loss": position.stop_loss,
            "pnl_pips": pnl_pips,
            "tp1_hit": position.tp1_hit,
            "tp2_hit": position.tp2_hit,
            "at_breakeven": position.moved_to_breakeven,
            "trailing_active": position.trailing_active,
            "remaining_size": position.current_position_size
        }
    
    def _calculate_pnl_pips(self, position: PositionState) -> float:
        """Calculate current P&L in pips"""
        pip_size = 0.01 if position.symbol == 'XAUUSD' else 1.0
        
        if position.action == "BUY":
            pnl = (position.current_price - position.entry_price) / pip_size
        else:
            pnl = (position.entry_price - position.current_price) / pip_size
        
        return pnl
