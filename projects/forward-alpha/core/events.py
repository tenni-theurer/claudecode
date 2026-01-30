"""
Event Tracker - Dynamic weighted events database
Tracks 1000+ events that can affect stocks with learned weights
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from config import EVENTS_FILE, MAX_EVENTS


@dataclass
class Event:
    """A single event that can affect stock prices"""
    id: str
    category: str  # macro, geopolitical, company, sector, technology
    subcategory: str
    description: str
    weight: float  # -1.0 to +1.0, negative = bearish, positive = bullish
    confidence: float  # 0.0 to 1.0
    tickers: list  # Affected tickers, empty = market-wide
    timestamp: str
    source: str
    tags: list

    # Learning fields
    predicted_impact: float  # What we predicted
    actual_impact: Optional[float] = None  # What actually happened (for learning)
    resolved: bool = False


class EventTracker:
    """
    Manages the dynamic events database

    Events are categorized as:
    - macro: Fed rates, inflation, GDP, unemployment
    - geopolitical: Wars, tariffs, sanctions, elections
    - company: Earnings, guidance, management changes, products
    - sector: Industry trends, regulations, supply chain
    - technology: Breakthroughs, adoption curves, disruptions
    """

    CATEGORIES = ["macro", "geopolitical", "company", "sector", "technology"]

    def __init__(self, events_file: Path = EVENTS_FILE):
        self.events_file = events_file
        self.events: dict[str, Event] = {}
        self._load()

    def _load(self):
        """Load events from file"""
        if self.events_file.exists():
            try:
                with open(self.events_file) as f:
                    data = json.load(f)
                    for event_data in data.get("events", []):
                        event = Event(**event_data)
                        self.events[event.id] = event
            except (json.JSONDecodeError, TypeError):
                self.events = {}

    def _save(self):
        """Save events to file"""
        data = {
            "updated_at": datetime.now().isoformat(),
            "count": len(self.events),
            "events": [asdict(e) for e in self.events.values()]
        }
        with open(self.events_file, "w") as f:
            json.dump(data, f, indent=2)

    def add_event(
        self,
        category: str,
        subcategory: str,
        description: str,
        weight: float,
        tickers: list = None,
        source: str = "manual",
        tags: list = None,
        confidence: float = 0.7,
    ) -> Event:
        """Add a new event to track"""
        event_id = f"E{len(self.events) + 1:05d}"

        event = Event(
            id=event_id,
            category=category,
            subcategory=subcategory,
            description=description,
            weight=max(-1.0, min(1.0, weight)),  # Clamp to [-1, 1]
            confidence=confidence,
            tickers=tickers or [],
            timestamp=datetime.now().isoformat(),
            source=source,
            tags=tags or [],
            predicted_impact=weight,
        )

        self.events[event_id] = event

        # Prune old events if over limit
        if len(self.events) > MAX_EVENTS:
            self._prune_old_events()

        self._save()
        return event

    def update_weight(self, event_id: str, new_weight: float, actual_impact: float = None):
        """Update an event's weight (for learning)"""
        if event_id in self.events:
            event = self.events[event_id]
            event.weight = max(-1.0, min(1.0, new_weight))
            if actual_impact is not None:
                event.actual_impact = actual_impact
                event.resolved = True
            self._save()

    def get_events_for_ticker(self, ticker: str) -> list[Event]:
        """Get all events affecting a specific ticker"""
        return [
            e for e in self.events.values()
            if ticker in e.tickers or len(e.tickers) == 0  # Market-wide events
        ]

    def get_events_by_category(self, category: str) -> list[Event]:
        """Get events by category"""
        return [e for e in self.events.values() if e.category == category]

    def get_active_events(self, days: int = 30) -> list[Event]:
        """Get recent unresolved events"""
        cutoff = datetime.now().timestamp() - (days * 86400)
        return [
            e for e in self.events.values()
            if not e.resolved and datetime.fromisoformat(e.timestamp).timestamp() > cutoff
        ]

    def calculate_aggregate_weight(self, ticker: str) -> dict:
        """
        Calculate aggregate weighted impact for a ticker

        Returns dict with:
        - total_weight: Sum of weighted events
        - bullish_count: Number of positive events
        - bearish_count: Number of negative events
        - top_bullish: Top 3 bullish events
        - top_bearish: Top 3 bearish events
        """
        events = self.get_events_for_ticker(ticker)
        active = [e for e in events if not e.resolved]

        total_weight = sum(e.weight * e.confidence for e in active)
        bullish = [e for e in active if e.weight > 0]
        bearish = [e for e in active if e.weight < 0]

        return {
            "total_weight": total_weight,
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "net_sentiment": "bullish" if total_weight > 0.1 else "bearish" if total_weight < -0.1 else "neutral",
            "top_bullish": sorted(bullish, key=lambda e: e.weight * e.confidence, reverse=True)[:3],
            "top_bearish": sorted(bearish, key=lambda e: e.weight * e.confidence)[:3],
        }

    def _prune_old_events(self):
        """Remove oldest resolved events when over limit"""
        resolved = [(eid, e) for eid, e in self.events.items() if e.resolved]
        resolved.sort(key=lambda x: x[1].timestamp)

        # Remove oldest 10%
        to_remove = len(self.events) - int(MAX_EVENTS * 0.9)
        for eid, _ in resolved[:to_remove]:
            del self.events[eid]

    def get_summary(self) -> dict:
        """Get summary statistics"""
        by_category = {}
        for cat in self.CATEGORIES:
            events = self.get_events_by_category(cat)
            by_category[cat] = {
                "count": len(events),
                "avg_weight": sum(e.weight for e in events) / len(events) if events else 0,
            }

        return {
            "total_events": len(self.events),
            "active_events": len([e for e in self.events.values() if not e.resolved]),
            "by_category": by_category,
        }


# Seed events for common macro factors
SEED_EVENTS = [
    ("macro", "fed_rates", "Fed rate hike cycle", -0.3, [], ["fed", "rates", "monetary"]),
    ("macro", "fed_rates", "Fed rate cut cycle", 0.4, [], ["fed", "rates", "monetary"]),
    ("macro", "inflation", "High inflation environment", -0.2, [], ["inflation", "cpi"]),
    ("macro", "gdp", "Strong GDP growth", 0.3, [], ["gdp", "growth"]),
    ("macro", "unemployment", "Rising unemployment", -0.3, [], ["jobs", "unemployment"]),
    ("geopolitical", "trade", "US-China trade tensions", -0.2, ["NVDA", "AAPL", "TSM"], ["china", "tariffs"]),
    ("geopolitical", "war", "Active military conflict", -0.2, [], ["war", "conflict"]),
    ("technology", "ai", "AI adoption acceleration", 0.5, ["NVDA", "MSFT", "GOOGL"], ["ai", "llm"]),
    ("sector", "semiconductors", "Chip shortage", 0.3, ["NVDA", "AMD", "TSM", "INTC"], ["chips", "supply"]),
]


def initialize_events():
    """Initialize event tracker with seed events"""
    tracker = EventTracker()

    if len(tracker.events) == 0:
        for cat, subcat, desc, weight, tickers, tags in SEED_EVENTS:
            tracker.add_event(
                category=cat,
                subcategory=subcat,
                description=desc,
                weight=weight,
                tickers=tickers,
                tags=tags,
                source="seed",
            )

    return tracker
