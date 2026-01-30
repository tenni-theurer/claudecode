"""
Index Gate
Only play when forecasting index +10% in next 12 months
Uses multi-LLM consensus to forecast S&P 500
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from core.data_collector import get_index_data
from core.events import EventTracker
from llm.multi_llm import MultiLLM
from llm.consensus import ConsensusBuilder, Consensus
from config import INDEX_GATE_THRESHOLD


@dataclass
class IndexForecast:
    """Forecast for an index"""
    ticker: str  # SPY, QQQ, etc.
    current_price: float
    forecast_date: str
    horizon_months: int

    # Probability estimates
    p_up_10: float  # P(+10% or more in horizon)
    p_up_5: float   # P(+5% or more)
    p_down_5: float # P(-5% or more)
    p_down_10: float# P(-10% or more)

    # Gate decision
    gate_open: bool  # True = safe to play
    confidence: str  # low/medium/high

    # Supporting data
    key_bullish_factors: list
    key_bearish_factors: list
    reasoning: str


class IndexGate:
    """
    Controls whether to enter positions based on index forecast

    Core rule: Only play when P(index +10% in 12 months) > threshold

    This prevents catching falling knives during bear markets.
    Even if individual stocks look attractive, if the overall
    market is forecast to decline, stay in cash.
    """

    def __init__(self):
        self.llm = MultiLLM()
        self.event_tracker = EventTracker()
        self.last_forecast: Optional[IndexForecast] = None
        self.forecast_history: list[IndexForecast] = []

    def check_gate(self, index_ticker: str = "SPY") -> IndexForecast:
        """
        Check if the index gate is open (safe to play)

        Args:
            index_ticker: Index to forecast (default SPY for S&P 500)

        Returns:
            IndexForecast with gate_open = True/False
        """
        # Get current index data
        index_data = get_index_data(index_ticker)

        # Get macro events
        macro_events = self.event_tracker.get_events_by_category("macro")
        geopolitical_events = self.event_tracker.get_events_by_category("geopolitical")

        # Build forecast prompt
        prompt = self._build_forecast_prompt(index_ticker, index_data, macro_events, geopolitical_events)

        # Query all LLMs
        responses = self.llm.query_all(prompt, system=self._get_system_prompt())

        # Parse forecasts
        forecasts = []
        for provider, resp in responses.items():
            if resp.get("response"):
                forecast = self._parse_forecast(provider, resp["response"])
                if forecast:
                    forecasts.append(forecast)

        # Calculate consensus
        if not forecasts:
            # Default to cautious if no forecasts
            return self._default_forecast(index_ticker, index_data)

        avg_p_up_10 = sum(f["p_up_10"] for f in forecasts) / len(forecasts)
        avg_p_up_5 = sum(f["p_up_5"] for f in forecasts) / len(forecasts)
        avg_p_down_5 = sum(f["p_down_5"] for f in forecasts) / len(forecasts)
        avg_p_down_10 = sum(f["p_down_10"] for f in forecasts) / len(forecasts)

        # Collect factors from all responses
        bullish = []
        bearish = []
        for f in forecasts:
            bullish.extend(f.get("bullish", [])[:2])
            bearish.extend(f.get("bearish", [])[:2])

        # Determine if gate is open
        gate_open = avg_p_up_10 >= INDEX_GATE_THRESHOLD

        # Determine confidence based on spread
        spread = max(f["p_up_10"] for f in forecasts) - min(f["p_up_10"] for f in forecasts)
        if spread < 0.1:
            confidence = "high"
        elif spread < 0.2:
            confidence = "medium"
        else:
            confidence = "low"

        forecast = IndexForecast(
            ticker=index_ticker,
            current_price=index_data.get("price", 0),
            forecast_date=datetime.now().isoformat(),
            horizon_months=12,
            p_up_10=round(avg_p_up_10, 3),
            p_up_5=round(avg_p_up_5, 3),
            p_down_5=round(avg_p_down_5, 3),
            p_down_10=round(avg_p_down_10, 3),
            gate_open=gate_open,
            confidence=confidence,
            key_bullish_factors=list(set(bullish))[:5],
            key_bearish_factors=list(set(bearish))[:5],
            reasoning=self._generate_reasoning(gate_open, avg_p_up_10, bullish, bearish),
        )

        self.last_forecast = forecast
        self.forecast_history.append(forecast)
        return forecast

    def _get_system_prompt(self) -> str:
        return """You are a macroeconomic analyst at a major hedge fund. Your job is to forecast index returns.

Rules:
1. Provide specific probability estimates, not ranges
2. Consider current market conditions, valuations, and macro factors
3. Account for both bullish and bearish scenarios
4. Be calibrated - don't be overly optimistic or pessimistic
5. Format your response exactly as requested"""

    def _build_forecast_prompt(
        self,
        ticker: str,
        index_data: dict,
        macro_events: list,
        geo_events: list
    ) -> str:
        """Build prompt for index forecast"""
        price_pos = index_data.get("price_position", {})

        macro_summary = "\n".join([
            f"- {e.description} (weight: {e.weight:+.2f})"
            for e in macro_events[:10]
        ]) if macro_events else "No macro events tracked"

        geo_summary = "\n".join([
            f"- {e.description} (weight: {e.weight:+.2f})"
            for e in geo_events[:10]
        ]) if geo_events else "No geopolitical events tracked"

        return f"""Forecast {ticker} (S&P 500 ETF) returns over the NEXT 12 MONTHS.

Current Conditions:
- Price: ${index_data.get('price', 'N/A')}
- YTD Return: {index_data.get('ytd_return', 'N/A')}%
- P/E Ratio: {index_data.get('pe', 'N/A')}
- 52-Week Position: {price_pos.get('percentile_52w', 'N/A')}th percentile
- Near 52w High: {price_pos.get('near_high', False)}
- 30-Day Momentum: {price_pos.get('momentum_30d', 'N/A')}%

Macro Events:
{macro_summary}

Geopolitical Events:
{geo_summary}

Provide your forecast in this exact format:
- P(+10% or more): XX%
- P(+5% or more): XX%
- P(-5% or more): XX%
- P(-10% or more): XX%

KEY BULLISH FACTORS (top 3):
1. ...
2. ...
3. ...

KEY BEARISH FACTORS (top 3):
1. ...
2. ...
3. ...

Be specific with percentages. Commit to your best estimate."""

    def _parse_forecast(self, provider: str, response: str) -> Optional[dict]:
        """Parse forecast from LLM response"""
        import re

        try:
            # Extract percentages
            p_up_10 = self._extract_pct(response, r"P\(\+10%[^:]*\):\s*(\d+)")
            p_up_5 = self._extract_pct(response, r"P\(\+5%[^:]*\):\s*(\d+)")
            p_down_5 = self._extract_pct(response, r"P\(-5%[^:]*\):\s*(\d+)")
            p_down_10 = self._extract_pct(response, r"P\(-10%[^:]*\):\s*(\d+)")

            # Extract factors
            bullish = re.findall(r"BULLISH[^:]*:?\s*\n(?:\d+\.\s*(.+))+", response, re.I)
            bearish = re.findall(r"BEARISH[^:]*:?\s*\n(?:\d+\.\s*(.+))+", response, re.I)

            # Simpler factor extraction if regex fails
            if not bullish:
                bullish = re.findall(r"(?:bullish|positive)[^:]*:\s*(.+)", response, re.I)
            if not bearish:
                bearish = re.findall(r"(?:bearish|negative|risk)[^:]*:\s*(.+)", response, re.I)

            return {
                "provider": provider,
                "p_up_10": p_up_10 / 100 if p_up_10 else 0.3,
                "p_up_5": p_up_5 / 100 if p_up_5 else 0.5,
                "p_down_5": p_down_5 / 100 if p_down_5 else 0.3,
                "p_down_10": p_down_10 / 100 if p_down_10 else 0.15,
                "bullish": bullish if isinstance(bullish, list) else [bullish] if bullish else [],
                "bearish": bearish if isinstance(bearish, list) else [bearish] if bearish else [],
            }
        except Exception as e:
            print(f"Error parsing forecast from {provider}: {e}")
            return None

    def _extract_pct(self, text: str, pattern: str) -> Optional[float]:
        """Extract percentage from text"""
        import re
        match = re.search(pattern, text, re.I)
        if match:
            return float(match.group(1))
        return None

    def _generate_reasoning(
        self,
        gate_open: bool,
        p_up_10: float,
        bullish: list,
        bearish: list
    ) -> str:
        """Generate reasoning for gate decision"""
        if gate_open:
            status = f"GATE OPEN - P(+10%) = {p_up_10*100:.0f}% >= {INDEX_GATE_THRESHOLD*100:.0f}% threshold"
            action = "Safe to enter positions. Index expected to provide tailwind."
        else:
            status = f"GATE CLOSED - P(+10%) = {p_up_10*100:.0f}% < {INDEX_GATE_THRESHOLD*100:.0f}% threshold"
            action = "Stay in cash or reduce exposure. Index headwind expected."

        return f"""{status}

{action}

Top Bullish Factors:
{chr(10).join(f'- {f}' for f in bullish[:3]) if bullish else '- None identified'}

Top Bearish Factors:
{chr(10).join(f'- {f}' for f in bearish[:3]) if bearish else '- None identified'}"""

    def _default_forecast(self, ticker: str, index_data: dict) -> IndexForecast:
        """Return cautious default forecast when LLMs fail"""
        return IndexForecast(
            ticker=ticker,
            current_price=index_data.get("price", 0),
            forecast_date=datetime.now().isoformat(),
            horizon_months=12,
            p_up_10=0.3,  # Cautious default
            p_up_5=0.5,
            p_down_5=0.3,
            p_down_10=0.15,
            gate_open=False,  # Default to closed (cautious)
            confidence="low",
            key_bullish_factors=[],
            key_bearish_factors=[],
            reasoning="Unable to generate forecast. Defaulting to cautious stance.",
        )

    def get_gate_status(self) -> dict:
        """Get current gate status"""
        if not self.last_forecast:
            return {
                "status": "UNKNOWN",
                "message": "No forecast available. Run check_gate() first.",
                "gate_open": None,
            }

        return {
            "status": "OPEN" if self.last_forecast.gate_open else "CLOSED",
            "p_up_10": self.last_forecast.p_up_10,
            "threshold": INDEX_GATE_THRESHOLD,
            "gate_open": self.last_forecast.gate_open,
            "forecast_date": self.last_forecast.forecast_date,
            "confidence": self.last_forecast.confidence,
            "reasoning": self.last_forecast.reasoning,
        }

    def should_play(self) -> tuple[bool, str]:
        """Simple check: should we enter positions?"""
        if not self.last_forecast:
            return False, "No forecast available"

        if self.last_forecast.gate_open:
            return True, f"Gate OPEN: P(+10%) = {self.last_forecast.p_up_10*100:.0f}%"
        else:
            return False, f"Gate CLOSED: P(+10%) = {self.last_forecast.p_up_10*100:.0f}% (need {INDEX_GATE_THRESHOLD*100:.0f}%)"
