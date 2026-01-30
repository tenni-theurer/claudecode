"""
Forward Alpha v2 - Configuration
Multi-Events Collisions Engine
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Project paths
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

# Directories
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
EVENTS_DIR = OUTPUT_DIR / "events"
CACHE_DIR = PROJECT_ROOT / "cache"

# Ensure directories exist
for d in [DATA_DIR, REPORTS_DIR, EVENTS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")

# LLM Models
MODELS = {
    "claude": "claude-sonnet-4-20250514",
    "gpt": "gpt-4o",
    "gemini": "gemini-2.0-flash",
    "grok": "grok-3",
}

# Multi-LLM Debate Settings
DEBATE_ROUNDS = 3  # Max rounds to reach consensus
CONSENSUS_THRESHOLD = 0.15  # Max spread in probability estimates for consensus
QUERIES_PER_LLM = 5

# Probability Thresholds
PROB_THRESHOLDS = [0.05, 0.10]  # ±5%, ±10%

# Index Gate Settings
INDEX_TICKER = "SPY"  # S&P 500 ETF
MIN_INDEX_FORECAST = 0.10  # +10% to play
INDEX_GATE_THRESHOLD = 0.30  # P(+10%) must be >= 30% to open gate
FORECAST_HORIZON_MONTHS = 12

# Roller Coaster Settings
EARNINGS_WINDOW_DAYS = 14  # Days before earnings to consider
PEAK_THRESHOLD = 0.85  # Percentile for "at peak"
BOTTOM_THRESHOLD = 0.15  # Percentile for "at bottom"
ROLLER_COASTER_PEAK_THRESHOLD = 85  # Alias in percentage form
ROLLER_COASTER_BOTTOM_THRESHOLD = 15  # Alias in percentage form
LOOKBACK_DAYS = 90  # Days to look back for peak/bottom

# Position Management
POSITIONS_FILE = DATA_DIR / "positions.json"
BEAT_INDEX_TARGET = 10.0  # Goal: beat index by +10%

# Event Tracking
EVENTS_FILE = EVENTS_DIR / "events_database.json"
MAX_EVENTS = 2000

# Data Categories (11 total)
DATA_CATEGORIES = [
    "current_products",      # Current products sales forecast
    "new_products",          # New upcoming products sales forecast
    "customers",             # Types of customers and forecasts
    "competitors",           # Competitive landscape
    "key_people",            # Key people and their outlook
    "macro_events",          # Macroeconomic events
    "geopolitical",          # Geopolitical events
    "technology",            # Underlying technologies
    "future_products",       # Future products forecast
    "associated_fields",     # Associated fields forecast
    "market_players",        # Players at the gambling table
]

# Budget
MONTHLY_BUDGET = 100.0  # $100/month


class CostTracker:
    """Track API costs across all LLMs"""

    def __init__(self):
        self.costs = {"claude": 0.0, "gpt": 0.0, "gemini": 0.0, "grok": 0.0}
        self.calls = {"claude": 0, "gpt": 0, "gemini": 0, "grok": 0}

    def add(self, provider: str, cost: float):
        self.costs[provider] = self.costs.get(provider, 0) + cost
        self.calls[provider] = self.calls.get(provider, 0) + 1

    @property
    def total(self) -> float:
        return sum(self.costs.values())

    @property
    def total_calls(self) -> int:
        return sum(self.calls.values())


cost_tracker = CostTracker()
