"""
Position Manager
Tracks current holdings and suggests jumps between roller coasters
"""
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from config import POSITIONS_FILE, BEAT_INDEX_TARGET
from strategy.roller_coaster import RollerCoaster


@dataclass
class Position:
    """A single position in the portfolio"""
    ticker: str
    shares: float
    entry_price: float
    entry_date: str
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    roller_coaster_status: Optional[str] = None
    notes: str = ""


@dataclass
class JumpRecord:
    """Record of a jump between positions"""
    id: str
    from_ticker: str
    to_ticker: str
    jump_date: str
    from_price: float
    to_price: float
    shares_moved: float
    reason: str
    outcome: Optional[str] = None  # Filled later when resolved


class PositionManager:
    """
    Manages portfolio positions and tracks jumps

    Core functionality:
    - Track current positions
    - Suggest jumps based on roller coaster analysis
    - Record jump history for learning
    - Calculate portfolio performance vs index
    """

    def __init__(self, positions_file: Path = POSITIONS_FILE):
        self.positions_file = positions_file
        self.positions: dict[str, Position] = {}
        self.jump_history: list[JumpRecord] = []
        self.roller_coaster = RollerCoaster()
        self._load()

    def _load(self):
        """Load positions from file"""
        if self.positions_file.exists():
            try:
                with open(self.positions_file) as f:
                    data = json.load(f)

                    for pos_data in data.get("positions", []):
                        pos = Position(**pos_data)
                        self.positions[pos.ticker] = pos

                    for jump_data in data.get("jump_history", []):
                        self.jump_history.append(JumpRecord(**jump_data))
            except (json.JSONDecodeError, TypeError):
                self.positions = {}
                self.jump_history = []

    def _save(self):
        """Save positions to file"""
        data = {
            "updated_at": datetime.now().isoformat(),
            "positions": [asdict(p) for p in self.positions.values()],
            "jump_history": [asdict(j) for j in self.jump_history],
            "summary": self.get_portfolio_summary(),
        }
        with open(self.positions_file, "w") as f:
            json.dump(data, f, indent=2)

    def add_position(
        self,
        ticker: str,
        shares: float,
        price: float,
        notes: str = ""
    ) -> Position:
        """Add or update a position"""
        if ticker in self.positions:
            # Update existing
            pos = self.positions[ticker]
            # Average in
            total_cost = (pos.shares * pos.entry_price) + (shares * price)
            total_shares = pos.shares + shares
            pos.shares = total_shares
            pos.entry_price = total_cost / total_shares
            pos.notes = notes if notes else pos.notes
        else:
            # New position
            pos = Position(
                ticker=ticker,
                shares=shares,
                entry_price=price,
                entry_date=datetime.now().isoformat(),
                notes=notes,
            )
            self.positions[ticker] = pos

        self._save()
        return pos

    def close_position(self, ticker: str, price: float) -> Optional[dict]:
        """Close a position and record P&L"""
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        pnl = (price - pos.entry_price) * pos.shares
        pnl_pct = ((price - pos.entry_price) / pos.entry_price) * 100

        result = {
            "ticker": ticker,
            "shares": pos.shares,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "holding_days": (datetime.now() - datetime.fromisoformat(pos.entry_date)).days,
        }

        del self.positions[ticker]
        self._save()
        return result

    def execute_jump(
        self,
        from_ticker: str,
        to_ticker: str,
        from_price: float,
        to_price: float,
        reason: str
    ) -> Optional[JumpRecord]:
        """
        Execute a jump from one position to another

        Sells from_ticker and buys to_ticker with the proceeds
        """
        if from_ticker not in self.positions:
            return None

        from_pos = self.positions[from_ticker]
        proceeds = from_pos.shares * from_price
        new_shares = proceeds / to_price

        # Record the jump
        jump = JumpRecord(
            id=f"J{len(self.jump_history) + 1:05d}",
            from_ticker=from_ticker,
            to_ticker=to_ticker,
            jump_date=datetime.now().isoformat(),
            from_price=from_price,
            to_price=to_price,
            shares_moved=from_pos.shares,
            reason=reason,
        )
        self.jump_history.append(jump)

        # Close old position
        del self.positions[from_ticker]

        # Open new position
        self.add_position(to_ticker, new_shares, to_price, f"Jumped from {from_ticker}")

        self._save()
        return jump

    def update_prices(self, prices: dict[str, float]):
        """Update current prices and calculate unrealized P&L"""
        for ticker, price in prices.items():
            if ticker in self.positions:
                pos = self.positions[ticker]
                pos.current_price = price
                pos.unrealized_pnl = (price - pos.entry_price) * pos.shares
                pos.unrealized_pnl_pct = ((price - pos.entry_price) / pos.entry_price) * 100

        self._save()

    def update_roller_coaster_status(self):
        """Update roller coaster status for all positions"""
        for ticker in self.positions:
            status = self.roller_coaster.analyze_ticker(ticker)
            self.positions[ticker].roller_coaster_status = status.position

        self._save()

    def get_jump_suggestions(self, watchlist: list[str]) -> list[dict]:
        """
        Get suggested jumps for current positions

        Args:
            watchlist: Tickers to consider jumping to

        Returns:
            List of suggested jumps with reasoning
        """
        suggestions = []

        for ticker in self.positions:
            status = self.roller_coaster.analyze_ticker(ticker)

            # Only suggest jumps if at peak or falling
            if status.recommendation == "sell_to_jump":
                candidates = self.roller_coaster.find_jump_candidates(ticker, watchlist)

                for candidate in candidates[:3]:  # Top 3 per position
                    suggestions.append({
                        "from": ticker,
                        "from_status": status.position,
                        "to": candidate["ticker"],
                        "to_status": candidate["status"].position,
                        "score": candidate["score"],
                        "reason": candidate["reason"],
                        "from_reasoning": status.reasoning,
                    })

        # Sort by score
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return suggestions

    def get_portfolio_summary(self) -> dict:
        """Get portfolio summary"""
        if not self.positions:
            return {
                "total_value": 0,
                "total_cost": 0,
                "total_pnl": 0,
                "total_pnl_pct": 0,
                "positions_count": 0,
                "beat_target": BEAT_INDEX_TARGET,
            }

        total_value = sum(
            p.current_price * p.shares if p.current_price else p.entry_price * p.shares
            for p in self.positions.values()
        )
        total_cost = sum(p.entry_price * p.shares for p in self.positions.values())
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

        return {
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "positions_count": len(self.positions),
            "beat_target": BEAT_INDEX_TARGET,
            "positions": {
                ticker: {
                    "shares": p.shares,
                    "entry": p.entry_price,
                    "current": p.current_price,
                    "pnl_pct": round(p.unrealized_pnl_pct, 2) if p.current_price else 0,
                    "status": p.roller_coaster_status,
                }
                for ticker, p in self.positions.items()
            },
        }

    def get_jump_history_summary(self) -> dict:
        """Get summary of jump history"""
        if not self.jump_history:
            return {"total_jumps": 0, "recent_jumps": []}

        recent = self.jump_history[-10:]  # Last 10 jumps

        return {
            "total_jumps": len(self.jump_history),
            "recent_jumps": [
                {
                    "id": j.id,
                    "from": j.from_ticker,
                    "to": j.to_ticker,
                    "date": j.jump_date,
                    "reason": j.reason[:100],
                }
                for j in recent
            ],
        }

    def calculate_vs_index(self, index_return: float) -> dict:
        """
        Calculate performance vs index

        Args:
            index_return: Index return percentage over same period

        Returns:
            Dict with alpha calculation
        """
        summary = self.get_portfolio_summary()
        portfolio_return = summary["total_pnl_pct"]

        alpha = portfolio_return - index_return
        beating_index = alpha > 0
        meeting_target = alpha >= BEAT_INDEX_TARGET

        return {
            "portfolio_return": portfolio_return,
            "index_return": index_return,
            "alpha": round(alpha, 2),
            "beating_index": beating_index,
            "meeting_target": meeting_target,
            "target": BEAT_INDEX_TARGET,
            "status": (
                "EXCEEDING TARGET" if meeting_target
                else "BEATING INDEX" if beating_index
                else "UNDERPERFORMING"
            ),
        }
