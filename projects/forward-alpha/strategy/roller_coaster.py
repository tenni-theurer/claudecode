"""
Roller Coaster Strategy
Detects stocks at peaks vs bottoms, especially around earnings
Implements "jumping roller coasters" - move from peaked stocks to recovering ones
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf

from config import ROLLER_COASTER_PEAK_THRESHOLD, ROLLER_COASTER_BOTTOM_THRESHOLD


@dataclass
class RollerCoasterStatus:
    """Status of a stock on the roller coaster"""
    ticker: str
    position: str  # "peak", "bottom", "climbing", "falling", "middle"
    percentile_52w: float  # 0-100
    momentum_30d: float  # percentage
    momentum_7d: float  # percentage
    days_to_earnings: Optional[int]
    in_earnings_window: bool  # Within 14 days of earnings
    post_earnings_days: Optional[int]  # Days since last earnings
    recommendation: str  # "hold", "sell_to_jump", "buy_dip"
    jump_candidates: list  # Tickers to potentially jump to
    reasoning: str


class RollerCoaster:
    """
    Implements the "jumping roller coasters" strategy

    Core idea: After a stock peaks (post-earnings rally), move capital
    to a stock at the bottom (pre-earnings dip) to catch the next wave.

    Example: TSMC peaks after earnings -> Jump to NVDA at bottom before their earnings
    """

    def __init__(self):
        self.watched_tickers = []
        self.status_cache = {}

    def analyze_ticker(self, ticker: str) -> RollerCoasterStatus:
        """Analyze a single ticker's roller coaster position"""
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")

        if hist.empty:
            return self._empty_status(ticker)

        # Calculate position metrics
        high_52w = info.get("fiftyTwoWeekHigh", hist["Close"].max())
        low_52w = info.get("fiftyTwoWeekLow", hist["Close"].min())
        current = info.get("currentPrice", hist["Close"].iloc[-1])

        if high_52w and low_52w and high_52w != low_52w:
            percentile = ((current - low_52w) / (high_52w - low_52w)) * 100
        else:
            percentile = 50.0

        # Calculate momentum
        momentum_30d = self._calculate_momentum(hist, 30)
        momentum_7d = self._calculate_momentum(hist, 7)

        # Get earnings info
        earnings_info = self._get_earnings_info(stock)

        # Determine position on roller coaster
        position = self._determine_position(percentile, momentum_30d, momentum_7d)

        # Generate recommendation
        recommendation, reasoning = self._generate_recommendation(
            position, percentile, earnings_info, momentum_30d
        )

        status = RollerCoasterStatus(
            ticker=ticker,
            position=position,
            percentile_52w=round(percentile, 1),
            momentum_30d=round(momentum_30d, 2),
            momentum_7d=round(momentum_7d, 2),
            days_to_earnings=earnings_info.get("days_to"),
            in_earnings_window=earnings_info.get("in_window", False),
            post_earnings_days=earnings_info.get("days_since"),
            recommendation=recommendation,
            jump_candidates=[],  # Filled by find_jump_candidates
            reasoning=reasoning,
        )

        self.status_cache[ticker] = status
        return status

    def _calculate_momentum(self, hist, days: int) -> float:
        """Calculate price momentum over N days"""
        if len(hist) < days:
            return 0.0

        current = hist["Close"].iloc[-1]
        past = hist["Close"].iloc[-days]

        if past and past != 0:
            return ((current - past) / past) * 100
        return 0.0

    def _get_earnings_info(self, stock) -> dict:
        """Get earnings calendar information"""
        try:
            calendar = stock.calendar
            earnings_date = None

            if calendar is not None:
                if isinstance(calendar, dict):
                    earnings_dates = calendar.get("Earnings Date", [])
                    earnings_date = earnings_dates[0] if earnings_dates else None
                elif hasattr(calendar, 'iloc') and len(calendar) > 0:
                    earnings_date = calendar.iloc[0].get("Earnings Date")

            days_to = None
            if earnings_date:
                try:
                    if isinstance(earnings_date, str):
                        ed = datetime.fromisoformat(earnings_date)
                    else:
                        ed = earnings_date
                    days_to = (ed - datetime.now()).days
                except:
                    pass

            # Estimate days since last earnings (roughly 90 days cycle)
            days_since = None
            if days_to is not None:
                # If earnings are in the future, estimate days since last
                if days_to > 0:
                    days_since = 90 - days_to if days_to < 90 else None

            return {
                "date": str(earnings_date) if earnings_date else None,
                "days_to": days_to,
                "days_since": days_since,
                "in_window": days_to is not None and 0 < days_to <= 14,
            }
        except Exception:
            return {"date": None, "days_to": None, "days_since": None, "in_window": False}

    def _determine_position(self, percentile: float, momentum_30d: float, momentum_7d: float) -> str:
        """Determine position on the roller coaster"""
        if percentile >= ROLLER_COASTER_PEAK_THRESHOLD:
            if momentum_7d < 0:
                return "falling"  # Just passed peak
            return "peak"
        elif percentile <= ROLLER_COASTER_BOTTOM_THRESHOLD:
            if momentum_7d > 0:
                return "climbing"  # Starting recovery
            return "bottom"
        elif momentum_30d > 5:
            return "climbing"
        elif momentum_30d < -5:
            return "falling"
        else:
            return "middle"

    def _generate_recommendation(
        self,
        position: str,
        percentile: float,
        earnings_info: dict,
        momentum_30d: float
    ) -> tuple[str, str]:
        """Generate trading recommendation"""
        days_to = earnings_info.get("days_to")
        days_since = earnings_info.get("days_since")

        # At peak, especially after recent earnings -> Consider jumping
        if position == "peak" or position == "falling":
            if days_since is not None and days_since < 30:
                return (
                    "sell_to_jump",
                    f"Stock at {percentile:.0f}th percentile, {days_since} days post-earnings. "
                    f"Consider jumping to a bottom-dweller before their earnings."
                )
            elif percentile > 90:
                return (
                    "sell_to_jump",
                    f"Stock at {percentile:.0f}th percentile (near 52w high). "
                    f"Limited upside, consider jumping to recovery candidates."
                )
            else:
                return (
                    "hold",
                    f"Stock elevated but not at extreme. Monitor for further gains."
                )

        # At bottom, especially before earnings -> Good buy candidate
        elif position == "bottom" or position == "climbing":
            if days_to is not None and 0 < days_to <= 30:
                return (
                    "buy_dip",
                    f"Stock at {percentile:.0f}th percentile with earnings in {days_to} days. "
                    f"Potential earnings catalyst ahead."
                )
            elif percentile < 20:
                return (
                    "buy_dip",
                    f"Stock at {percentile:.0f}th percentile (near 52w low). "
                    f"Deep value if fundamentals support."
                )
            else:
                return (
                    "hold",
                    f"Stock recovering from lows. Watch for continuation."
                )

        # Middle ground
        else:
            return (
                "hold",
                f"Stock in middle of range ({percentile:.0f}th percentile). "
                f"No strong roller coaster signal."
            )

    def find_jump_candidates(self, current_ticker: str, watchlist: list[str]) -> list[dict]:
        """
        Find candidates to jump to from current position

        Args:
            current_ticker: Ticker we're currently holding
            watchlist: List of tickers to consider jumping to

        Returns:
            List of candidates with their status and score
        """
        current_status = self.status_cache.get(current_ticker)
        if not current_status:
            current_status = self.analyze_ticker(current_ticker)

        candidates = []

        for ticker in watchlist:
            if ticker == current_ticker:
                continue

            status = self.status_cache.get(ticker)
            if not status:
                status = self.analyze_ticker(ticker)

            # Score the candidate (higher = better jump target)
            score = self._score_jump_candidate(current_status, status)

            if score > 0:
                candidates.append({
                    "ticker": ticker,
                    "status": status,
                    "score": score,
                    "reason": self._explain_jump(current_status, status),
                })

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:5]  # Top 5 candidates

    def _score_jump_candidate(self, current: RollerCoasterStatus, candidate: RollerCoasterStatus) -> float:
        """Score a potential jump candidate"""
        score = 0.0

        # Penalize if current is not at peak (why jump?)
        if current.position not in ["peak", "falling"]:
            score -= 50

        # Reward candidates at bottom
        if candidate.position == "bottom":
            score += 40
        elif candidate.position == "climbing":
            score += 30  # Early recovery

        # Reward lower percentile (more upside potential)
        score += (100 - candidate.percentile_52w) * 0.3

        # Reward upcoming earnings (catalyst)
        if candidate.days_to_earnings and 0 < candidate.days_to_earnings <= 30:
            score += 25

        # Penalize if candidate is also at peak
        if candidate.position in ["peak", "falling"]:
            score -= 40

        return score

    def _explain_jump(self, current: RollerCoasterStatus, candidate: RollerCoasterStatus) -> str:
        """Explain why this jump makes sense"""
        parts = []

        parts.append(f"Jump from {current.ticker} ({current.position}, {current.percentile_52w:.0f}%)")
        parts.append(f"to {candidate.ticker} ({candidate.position}, {candidate.percentile_52w:.0f}%)")

        if candidate.days_to_earnings and 0 < candidate.days_to_earnings <= 30:
            parts.append(f"- Earnings in {candidate.days_to_earnings} days")

        if candidate.momentum_7d > 0 and candidate.position == "climbing":
            parts.append(f"- Early recovery momentum (+{candidate.momentum_7d:.1f}% 7d)")

        return " | ".join(parts)

    def _empty_status(self, ticker: str) -> RollerCoasterStatus:
        """Return empty status when data unavailable"""
        return RollerCoasterStatus(
            ticker=ticker,
            position="unknown",
            percentile_52w=50.0,
            momentum_30d=0.0,
            momentum_7d=0.0,
            days_to_earnings=None,
            in_earnings_window=False,
            post_earnings_days=None,
            recommendation="hold",
            jump_candidates=[],
            reasoning="Insufficient data to analyze",
        )

    def get_watchlist_summary(self, tickers: list[str]) -> dict:
        """Get summary of all watched tickers"""
        peaks = []
        bottoms = []
        climbing = []
        falling = []

        for ticker in tickers:
            status = self.analyze_ticker(ticker)

            if status.position == "peak":
                peaks.append(status)
            elif status.position == "bottom":
                bottoms.append(status)
            elif status.position == "climbing":
                climbing.append(status)
            elif status.position == "falling":
                falling.append(status)

        return {
            "peaks": peaks,  # Consider selling
            "bottoms": bottoms,  # Consider buying
            "climbing": climbing,  # Watch for continuation
            "falling": falling,  # Watch for bottom
            "jump_opportunities": self._find_all_jumps(peaks, bottoms + climbing),
        }

    def _find_all_jumps(self, sources: list, targets: list) -> list:
        """Find all viable jump opportunities"""
        opportunities = []

        for source in sources:
            for target in targets:
                score = self._score_jump_candidate(source, target)
                if score > 30:  # Minimum threshold
                    opportunities.append({
                        "from": source.ticker,
                        "to": target.ticker,
                        "score": score,
                        "reason": self._explain_jump(source, target),
                    })

        opportunities.sort(key=lambda x: x["score"], reverse=True)
        return opportunities[:10]
