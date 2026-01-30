"""
Collision Engine - Multi-Events Collision Theory Implementation

Models stocks as balls with velocity, mass, and shape:
- Velocity = momentum (speed + direction of price movement)
- Mass = market cap / importance
- Shape = how predictable/stable the stock behaves
- Nearby balls = correlated events that may collide

Allows dynamic injection of new events to recalculate probabilities.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json

from llm.multi_llm import MultiLLM
from config import REPORTS_DIR


@dataclass
class Ball:
    """A ball in the collision system - represents a factor/event"""
    id: str
    name: str
    category: str  # macro, company, geopolitical, technology, competitor

    # Physics properties
    velocity: float  # -1 to +1 (negative = bearish momentum, positive = bullish)
    mass: float  # 0 to 1 (importance/impact weight)
    roundness: float  # 0 to 1 (predictability - 1 = perfectly round/predictable)

    # Position relative to target
    distance: float  # 0 to 1 (0 = directly impacting, 1 = distant)

    description: str
    data_points: list = field(default_factory=list)


@dataclass
class ProbabilityState:
    """Probability estimates at a point in time"""
    timestamp: str
    p_up_5: float
    p_up_10: float
    p_down_5: float
    p_down_10: float
    reasoning: str
    events_considered: list
    confidence: str


class CollisionEngine:
    """
    Implements Multi-Events Collision Theory

    1. Model the target stock as a ball
    2. Identify all nearby balls (events/factors)
    3. Calculate initial probability based on trajectories
    4. Allow injection of new events
    5. Recalculate and show probability deltas
    """

    def __init__(self):
        self.llm = MultiLLM()
        self.target_ticker: str = None
        self.target_context: dict = None
        self.balls: list[Ball] = []
        self.probability_history: list[ProbabilityState] = []
        self.injected_events: list[dict] = []

    def initialize(self, ticker: str, context: dict, horizon: str = "earnings"):
        """Initialize the collision system for a target stock"""
        self.target_ticker = ticker
        self.target_context = context
        self.balls = []
        self.probability_history = []
        self.injected_events = []

        # Build initial ball set from context
        self._build_initial_balls(context)

        return self

    def _build_initial_balls(self, context: dict):
        """Build initial set of balls from context"""
        # These would typically come from the event tracker and data collector
        # For now, we'll let the LLM identify them
        pass

    def get_initial_estimate(self, horizon: str = "by earnings date") -> ProbabilityState:
        """Get initial probability estimate"""
        prompt = self._build_initial_prompt(horizon)

        responses = self.llm.query_all(prompt, system=self._get_system_prompt())

        # Parse and average responses
        state = self._parse_probability_response(responses, [])
        self.probability_history.append(state)

        return state

    def inject_event(self, event_description: str, category: str = "new_data") -> dict:
        """Inject a new event/data point into the system"""
        event = {
            "id": f"INJ_{len(self.injected_events) + 1:03d}",
            "description": event_description,
            "category": category,
            "injected_at": datetime.now().isoformat(),
        }
        self.injected_events.append(event)
        return event

    def recalculate(self) -> dict:
        """
        Recalculate probabilities with injected events

        Returns dict with:
        - new_state: Updated probability state
        - previous_state: Previous probability state
        - deltas: Changes in each probability
        - analysis: Explanation of what changed and why
        """
        if not self.probability_history:
            raise ValueError("No initial estimate. Call get_initial_estimate first.")

        previous_state = self.probability_history[-1]

        # Build challenge prompt with new events
        prompt = self._build_recalculation_prompt(previous_state)

        responses = self.llm.query_all(prompt, system=self._get_system_prompt())

        # Parse new state
        new_state = self._parse_probability_response(
            responses,
            [e["description"] for e in self.injected_events]
        )
        self.probability_history.append(new_state)

        # Calculate deltas
        deltas = {
            "p_up_5": (new_state.p_up_5 - previous_state.p_up_5) * 100,
            "p_up_10": (new_state.p_up_10 - previous_state.p_up_10) * 100,
            "p_down_5": (new_state.p_down_5 - previous_state.p_down_5) * 100,
            "p_down_10": (new_state.p_down_10 - previous_state.p_down_10) * 100,
        }

        # Generate analysis of what changed
        analysis = self._generate_delta_analysis(previous_state, new_state, deltas)

        return {
            "new_state": new_state,
            "previous_state": previous_state,
            "deltas": deltas,
            "analysis": analysis,
            "injected_events": self.injected_events.copy(),
        }

    def _get_system_prompt(self) -> str:
        return """You are a quantitative analyst using Multi-Events Collision Theory to forecast stock prices.

Imagine the stock as a ball moving through space. Other balls (events, factors, news) are moving nearby and may collide with it, changing its trajectory.

Your job:
1. Identify the key "balls" (factors) that could collide with this stock
2. Assess each ball's velocity (bullish/bearish momentum), mass (importance), and distance (how soon it impacts)
3. Calculate probability of price movements based on likely collisions

When new data is injected:
1. Reassess which balls are now closer or have changed velocity
2. Recalculate collision probabilities
3. Explain specifically HOW and WHY each new data point changes the forecast

Be specific with percentages. Show your reasoning about collision dynamics."""

    def _build_initial_prompt(self, horizon: str) -> str:
        ctx = self.target_context
        financials = ctx.get("financials", {})
        price_pos = ctx.get("price_position", {})
        earnings = ctx.get("earnings", {})

        earnings_info = ""
        if earnings.get("days_to_earnings"):
            earnings_info = f"Days to earnings: {earnings['days_to_earnings']}"

        return f"""Analyze {self.target_ticker} ({ctx.get('name', '')}) using Multi-Events Collision Theory.

CURRENT STATE:
- Price: ${financials.get('price', 'N/A')}
- Market Cap: ${(financials.get('market_cap') or 0)/1e9:.1f}B
- 52-Week Position: {price_pos.get('percentile_52w', 'N/A')}th percentile
- 30-Day Momentum: {price_pos.get('momentum_30d', 'N/A')}%
- {earnings_info}

TASK:
1. Identify the 5-10 most important "balls" (factors/events) that could collide with {self.target_ticker} {horizon}
2. For each ball, assess:
   - Velocity (bullish +1 to bearish -1)
   - Mass (importance 0-1)
   - Distance (how soon it impacts 0=now, 1=distant)
3. Calculate probability estimates

FORMAT YOUR RESPONSE:

## KEY COLLISION FACTORS
1. [Factor Name] - Velocity: X, Mass: X, Distance: X
   Description of this factor and its trajectory

## PROBABILITY ESTIMATES {horizon}
- P(+5% or more): XX%
- P(+10% or more): XX%
- P(-5% or more): XX%
- P(-10% or more): XX%

CONFIDENCE: low/medium/high

## REASONING
Explain the collision dynamics that lead to these probabilities."""

    def _build_recalculation_prompt(self, previous: ProbabilityState) -> str:
        events_text = "\n".join([
            f"- {e['description']}"
            for e in self.injected_events
        ])

        return f"""RECALCULATE probabilities for {self.target_ticker} given NEW DATA.

PREVIOUS ESTIMATES:
- P(+5% or more): {previous.p_up_5*100:.0f}%
- P(+10% or more): {previous.p_up_10*100:.0f}%
- P(-5% or more): {previous.p_down_5*100:.0f}%
- P(-10% or more): {previous.p_down_10*100:.0f}%

Previous reasoning: {previous.reasoning[:500]}

NEW DATA INJECTED:
{events_text}

TASK:
1. Analyze how each new data point changes the collision dynamics
2. Which balls moved closer? Which changed velocity? Which new balls appeared?
3. Recalculate probabilities

FORMAT YOUR RESPONSE:

## IMPACT ANALYSIS
For each new data point, explain:
- How it changes ball positions/velocities
- Net effect on collision probability (bullish/bearish)
- Magnitude of impact (minor/moderate/major)

## REVISED PROBABILITY ESTIMATES
- P(+5% or more): XX%
- P(+10% or more): XX%
- P(-5% or more): XX%
- P(-10% or more): XX%

## PROBABILITY DELTAS
- P(+5%): [+/-X percentage points] because...
- P(+10%): [+/-X percentage points] because...
- P(-5%): [+/-X percentage points] because...
- P(-10%): [+/-X percentage points] because...

CONFIDENCE: low/medium/high"""

    def _parse_probability_response(self, responses: dict, events: list) -> ProbabilityState:
        """Parse LLM responses into a ProbabilityState"""
        import re

        # Combine and parse responses
        all_estimates = []
        all_reasoning = []
        individual_estimates = {}  # Store per-provider estimates

        for provider, resp in responses.items():
            if resp.get("error"):
                # Log failed providers
                error_msg = str(resp.get("error", ""))[:100]
                if "429" in error_msg or "quota" in error_msg.lower():
                    individual_estimates[provider] = {"error": "RATE_LIMITED"}
                else:
                    individual_estimates[provider] = {"error": error_msg[:50]}
                continue

            if resp.get("response"):
                text = resp["response"]

                # Try multiple patterns to extract percentages
                # Remove markdown bold/italic formatting first
                clean_text = re.sub(r'\*+', '', text)

                p_up_5 = self._extract_pct(clean_text, r"P\(\+5%[^:]*\):\s*(\d+)")
                if p_up_5 is None:
                    p_up_5 = self._extract_pct(clean_text, r"\+5%[^:]*:\s*(\d+)")
                if p_up_5 is None:
                    p_up_5 = self._extract_pct(clean_text, r"5%\s*(?:or more|increase)[^:]*:\s*(\d+)")

                p_up_10 = self._extract_pct(clean_text, r"P\(\+10%[^:]*\):\s*(\d+)")
                if p_up_10 is None:
                    p_up_10 = self._extract_pct(clean_text, r"\+10%[^:]*:\s*(\d+)")

                p_down_5 = self._extract_pct(clean_text, r"P\(-5%[^:]*\):\s*(\d+)")
                if p_down_5 is None:
                    p_down_5 = self._extract_pct(clean_text, r"-5%[^:]*:\s*(\d+)")

                p_down_10 = self._extract_pct(clean_text, r"P\(-10%[^:]*\):\s*(\d+)")
                if p_down_10 is None:
                    p_down_10 = self._extract_pct(clean_text, r"-10%[^:]*:\s*(\d+)")

                if p_up_5 is not None:
                    estimate = {
                        "p_up_5": p_up_5 / 100,
                        "p_up_10": (p_up_10 or 20) / 100,
                        "p_down_5": (p_down_5 or 30) / 100,
                        "p_down_10": (p_down_10 or 15) / 100,
                    }
                    all_estimates.append(estimate)
                    individual_estimates[provider] = estimate
                else:
                    individual_estimates[provider] = {"error": "PARSE_FAILED"}

                # Extract reasoning - store full text
                reasoning_match = re.search(r"(?:REASONING|IMPACT ANALYSIS)[:\s]*(.+?)(?:##|$)", text, re.S | re.I)
                if reasoning_match:
                    all_reasoning.append(f"[{provider.upper()}]: {reasoning_match.group(1).strip()}")
                else:
                    # Store the full response if no specific reasoning section found
                    all_reasoning.append(f"[{provider.upper()}]: {text.strip()}")

        # Store individual estimates for display
        self.last_individual_estimates = individual_estimates

        # Average estimates
        if all_estimates:
            n = len(all_estimates)
            avg = {
                "p_up_5": sum(e["p_up_5"] for e in all_estimates) / n,
                "p_up_10": sum(e["p_up_10"] for e in all_estimates) / n,
                "p_down_5": sum(e["p_down_5"] for e in all_estimates) / n,
                "p_down_10": sum(e["p_down_10"] for e in all_estimates) / n,
            }
        else:
            avg = {"p_up_5": 0.5, "p_up_10": 0.3, "p_down_5": 0.3, "p_down_10": 0.15}

        return ProbabilityState(
            timestamp=datetime.now().isoformat(),
            p_up_5=avg["p_up_5"],
            p_up_10=avg["p_up_10"],
            p_down_5=avg["p_down_5"],
            p_down_10=avg["p_down_10"],
            reasoning="\n\n".join(all_reasoning) if all_reasoning else "No reasoning provided",
            events_considered=events,
            confidence="medium",
        )

    def get_individual_estimates(self) -> dict:
        """Get the last individual estimates from each LLM"""
        return getattr(self, 'last_individual_estimates', {})

    def _extract_pct(self, text: str, pattern: str) -> Optional[float]:
        import re
        match = re.search(pattern, text, re.I)
        if match:
            return float(match.group(1))
        return None

    def _generate_delta_analysis(self, prev: ProbabilityState, new: ProbabilityState, deltas: dict) -> str:
        """Generate analysis of what changed"""
        lines = ["## Probability Change Analysis", ""]

        # Overall direction
        net_bullish = deltas["p_up_10"] - deltas["p_down_10"]
        if net_bullish > 5:
            lines.append(f"**NET EFFECT: BULLISH** (+{net_bullish:.1f}% shift)")
        elif net_bullish < -5:
            lines.append(f"**NET EFFECT: BEARISH** ({net_bullish:.1f}% shift)")
        else:
            lines.append(f"**NET EFFECT: NEUTRAL** ({net_bullish:+.1f}% shift)")

        lines.append("")
        lines.append("### Probability Deltas:")
        lines.append(f"- P(+5%): {prev.p_up_5*100:.0f}% → {new.p_up_5*100:.0f}% ({deltas['p_up_5']:+.1f} pp)")
        lines.append(f"- P(+10%): {prev.p_up_10*100:.0f}% → {new.p_up_10*100:.0f}% ({deltas['p_up_10']:+.1f} pp)")
        lines.append(f"- P(-5%): {prev.p_down_5*100:.0f}% → {new.p_down_5*100:.0f}% ({deltas['p_down_5']:+.1f} pp)")
        lines.append(f"- P(-10%): {prev.p_down_10*100:.0f}% → {new.p_down_10*100:.0f}% ({deltas['p_down_10']:+.1f} pp)")

        lines.append("")
        lines.append("### New Data Impact:")
        for event in self.injected_events:
            lines.append(f"- {event['description']}")

        return "\n".join(lines)

    def get_history(self) -> list:
        """Get probability history"""
        return [
            {
                "timestamp": s.timestamp,
                "p_up_5": s.p_up_5,
                "p_up_10": s.p_up_10,
                "p_down_5": s.p_down_5,
                "p_down_10": s.p_down_10,
                "events": s.events_considered,
            }
            for s in self.probability_history
        ]

    def save_session(self, filename: str = None) -> str:
        """Save the collision session to a file"""
        if not filename:
            filename = f"{self.target_ticker}_collision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        path = REPORTS_DIR / filename

        data = {
            "ticker": self.target_ticker,
            "context": self.target_context,
            "injected_events": self.injected_events,
            "probability_history": [
                {
                    "timestamp": s.timestamp,
                    "p_up_5": s.p_up_5,
                    "p_up_10": s.p_up_10,
                    "p_down_5": s.p_down_5,
                    "p_down_10": s.p_down_10,
                    "reasoning": s.reasoning,
                    "events_considered": s.events_considered,
                    "confidence": s.confidence,
                }
                for s in self.probability_history
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return str(path)

    def generate_markdown_report(self) -> str:
        """Generate a detailed markdown report with full reasoning"""
        ctx = self.target_context
        financials = ctx.get("financials", {})
        price_pos = ctx.get("price_position", {})
        earnings = ctx.get("earnings", {})

        lines = [
            f"# {self.target_ticker} Collision Analysis Report",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
        ]

        # Summary table if we have history
        if len(self.probability_history) >= 2:
            initial = self.probability_history[0]
            final = self.probability_history[-1]
            lines.extend([
                "| Metric | Initial | After New Data | Delta |",
                "|--------|---------|----------------|-------|",
                f"| **P(+5%)** | {initial.p_up_5*100:.0f}% | {final.p_up_5*100:.0f}% | {(final.p_up_5-initial.p_up_5)*100:+.1f} pp |",
                f"| **P(+10%)** | {initial.p_up_10*100:.0f}% | {final.p_up_10*100:.0f}% | {(final.p_up_10-initial.p_up_10)*100:+.1f} pp |",
                f"| **P(-5%)** | {initial.p_down_5*100:.0f}% | {final.p_down_5*100:.0f}% | {(final.p_down_5-initial.p_down_5)*100:+.1f} pp |",
                f"| **P(-10%)** | {initial.p_down_10*100:.0f}% | {final.p_down_10*100:.0f}% | {(final.p_down_10-initial.p_down_10)*100:+.1f} pp |",
                "",
            ])
            net_shift = (final.p_up_10 - initial.p_up_10) - (final.p_down_10 - initial.p_down_10)
            if net_shift > 0.05:
                lines.append(f"**Net Effect: BULLISH (+{net_shift*100:.1f}% shift toward upside)**")
            elif net_shift < -0.05:
                lines.append(f"**Net Effect: BEARISH ({net_shift*100:.1f}% shift toward downside)**")
            else:
                lines.append(f"**Net Effect: NEUTRAL ({net_shift*100:+.1f}% shift)**")
        elif self.probability_history:
            initial = self.probability_history[0]
            lines.extend([
                "| Metric | Estimate |",
                "|--------|----------|",
                f"| **P(+5%)** | {initial.p_up_5*100:.0f}% |",
                f"| **P(+10%)** | {initial.p_up_10*100:.0f}% |",
                f"| **P(-5%)** | {initial.p_down_5*100:.0f}% |",
                f"| **P(-10%)** | {initial.p_down_10*100:.0f}% |",
            ])

        lines.extend([
            "",
            "---",
            "",
            "## Company Context",
            "",
            f"**{ctx.get('name', self.target_ticker)}**",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Price | ${financials.get('price', 'N/A')} |",
            f"| Market Cap | ${(financials.get('market_cap') or 0)/1e9:.0f}B |",
            f"| Revenue TTM | ${(financials.get('revenue_ttm') or 0)/1e9:.0f}B |",
            f"| Revenue Growth | {(financials.get('revenue_growth') or 0)*100:.1f}% |",
            f"| Gross Margin | {(financials.get('gross_margin') or 0)*100:.1f}% |",
            f"| Forward P/E | {financials.get('pe_forward') or 'N/A'} |",
            f"| 52-Week Range | ${financials.get('price_52w_low', 'N/A')} - ${financials.get('price_52w_high', 'N/A')} |",
            f"| Current Percentile | {price_pos.get('percentile_52w', 'N/A')}% |",
            f"| 30-Day Momentum | {price_pos.get('momentum_30d', 'N/A'):+.1f}% |" if price_pos.get('momentum_30d') else "| 30-Day Momentum | N/A |",
        ])

        if earnings.get("next_earnings_date"):
            lines.append(f"| **Earnings Date** | **{earnings['next_earnings_date']} ({earnings.get('days_to_earnings', '?')} days)** |")

        # Individual LLM estimates
        individual = self.get_individual_estimates()
        if individual:
            lines.extend([
                "",
                "---",
                "",
                "## Individual LLM Estimates",
                "",
                "| LLM | P(+5%) | P(+10%) | P(-5%) | P(-10%) |",
                "|-----|--------|---------|--------|---------|",
            ])
            for provider, est in individual.items():
                if "error" in est:
                    lines.append(f"| **{provider.upper()}** | [{est['error']}] | - | - | - |")
                else:
                    lines.append(f"| **{provider.upper()}** | {est['p_up_5']*100:.0f}% | {est['p_up_10']*100:.0f}% | {est['p_down_5']*100:.0f}% | {est['p_down_10']*100:.0f}% |")

        # Injected events
        if self.injected_events:
            lines.extend([
                "",
                "---",
                "",
                "## Injected Events",
                "",
            ])
            for event in self.injected_events:
                lines.append(f"- **{event['id']}**: {event['description']}")

        # Full reasoning from each state
        lines.extend([
            "",
            "---",
            "",
            "## Full LLM Reasoning",
            "",
        ])

        for i, state in enumerate(self.probability_history):
            if i == 0:
                lines.append("### Initial Analysis (Before New Data)")
            else:
                lines.append(f"### Revised Analysis (After Injecting Events)")

            lines.append("")

            # Split by provider
            if state.reasoning:
                reasoning_parts = state.reasoning.split("\n\n")
                for part in reasoning_parts:
                    if part.startswith("[GEMINI]:"):
                        lines.extend([
                            "#### Gemini's Analysis",
                            "",
                            part.replace("[GEMINI]: ", ""),
                            "",
                        ])
                    elif part.startswith("[GPT]:"):
                        lines.extend([
                            "#### GPT-4's Analysis",
                            "",
                            part.replace("[GPT]: ", ""),
                            "",
                        ])
                    elif part.startswith("[CLAUDE]:"):
                        lines.extend([
                            "#### Claude's Analysis",
                            "",
                            part.replace("[CLAUDE]: ", ""),
                            "",
                        ])
                    else:
                        lines.append(part)
                        lines.append("")

            lines.append("---")
            lines.append("")

        lines.extend([
            "",
            "*Report generated by Forward Alpha v2 - Multi-Events Collision Engine*",
        ])

        return "\n".join(lines)

    def save_markdown_report(self, filename: str = None) -> str:
        """Save a detailed markdown report"""
        if not filename:
            filename = f"{self.target_ticker}_collision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        path = REPORTS_DIR / filename
        content = self.generate_markdown_report()

        with open(path, "w") as f:
            f.write(content)

        return str(path)
