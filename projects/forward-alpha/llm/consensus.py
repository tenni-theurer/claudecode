"""
Consensus Builder
Drives multi-LLM debate to reach consensus on probability estimates
"""
import re
from typing import Optional
from dataclasses import dataclass

from config import DEBATE_ROUNDS, CONSENSUS_THRESHOLD
from llm.multi_llm import MultiLLM, build_probability_prompt, build_challenge_prompt


@dataclass
class ProbabilityEstimate:
    """Probability estimates from a single LLM"""
    provider: str
    p_up_5: float  # P(+5% or more)
    p_up_10: float  # P(+10% or more)
    p_down_5: float  # P(-5% or more)
    p_down_10: float  # P(-10% or more)
    confidence: str
    key_factors: list
    risks: list
    raw_response: str


@dataclass
class Consensus:
    """Consensus result from multi-LLM debate"""
    reached: bool
    rounds: int
    estimates: dict  # Provider -> ProbabilityEstimate
    final_estimate: dict  # Averaged/consensus probabilities
    spread: float  # Max spread between estimates
    reasoning: str


class ConsensusBuilder:
    """
    Orchestrates multi-LLM debate to reach consensus

    Process:
    1. Query all LLMs for initial estimates
    2. Compare estimates - if spread > threshold, continue
    3. Challenge with missed data points
    4. Repeat until consensus or max rounds
    """

    def __init__(self):
        self.llm = MultiLLM()

    def build_consensus(
        self,
        ticker: str,
        data: dict,
        category_analyses: dict,
        missed_data_callback=None,
    ) -> Consensus:
        """
        Drive debate until consensus is reached

        Args:
            ticker: Stock ticker
            data: Stock data
            category_analyses: Analyses from all 11 categories
            missed_data_callback: Function to get missed data points to challenge with

        Returns:
            Consensus object with final estimates
        """
        # Build context from all category analyses
        context = self._build_context(ticker, data, category_analyses)

        estimates_history = []
        current_round = 0

        while current_round < DEBATE_ROUNDS:
            current_round += 1

            if current_round == 1:
                # Initial query
                prompt = build_probability_prompt(ticker, data, context)
            else:
                # Challenge with missed data
                missed_data = missed_data_callback(ticker, estimates_history[-1]) if missed_data_callback else []
                if not missed_data:
                    missed_data = self._generate_challenge_points(ticker, estimates_history[-1])

                prompt = build_challenge_prompt(ticker, estimates_history[-1], missed_data)

            # Query all LLMs
            responses = self.llm.query_all(prompt, system=self._get_system_prompt())

            # Parse estimates
            estimates = {}
            for provider, resp in responses.items():
                if resp.get("response"):
                    estimate = self._parse_estimate(provider, resp["response"])
                    if estimate:
                        estimates[provider] = estimate

            estimates_history.append(responses)

            # Check for consensus
            spread = self._calculate_spread(estimates)

            if spread <= CONSENSUS_THRESHOLD or current_round >= DEBATE_ROUNDS:
                final_estimate = self._calculate_final_estimate(estimates)

                return Consensus(
                    reached=spread <= CONSENSUS_THRESHOLD,
                    rounds=current_round,
                    estimates=estimates,
                    final_estimate=final_estimate,
                    spread=spread,
                    reasoning=self._generate_reasoning(estimates),
                )

        # Should not reach here, but just in case
        return Consensus(
            reached=False,
            rounds=current_round,
            estimates={},
            final_estimate={},
            spread=1.0,
            reasoning="Failed to reach consensus",
        )

    def _get_system_prompt(self) -> str:
        return """You are a quantitative analyst at a hedge fund. Your job is to provide precise probability estimates for stock price movements.

Rules:
1. Be specific - commit to actual percentages, don't hedge
2. Base estimates on concrete data and reasoning
3. Consider both bull and bear cases
4. Acknowledge uncertainty but still provide your best estimate
5. Format your response exactly as requested"""

    def _build_context(self, ticker: str, data: dict, category_analyses: dict) -> str:
        """Build context string from all analyses"""
        market_cap = data['financials'].get('market_cap') or 0
        revenue_growth = data['financials'].get('revenue_growth') or 0

        lines = [
            f"# Analysis of {ticker} ({data.get('name', '')})",
            "",
            f"## Company Overview",
            f"- Sector: {data.get('sector') or 'N/A'}",
            f"- Industry: {data.get('industry') or 'N/A'}",
            f"- Market Cap: ${market_cap/1e9:.1f}B",
            f"- Revenue Growth: {revenue_growth*100:.1f}%",
            "",
        ]

        for category, analysis in category_analyses.items():
            if analysis:
                lines.append(f"## {category.replace('_', ' ').title()}")
                lines.append(str(analysis)[:1500])  # Truncate long analyses
                lines.append("")

        return "\n".join(lines)

    def _parse_estimate(self, provider: str, response: str) -> Optional[ProbabilityEstimate]:
        """Parse probability estimates from LLM response"""
        try:
            # Extract percentages using regex
            p_up_5 = self._extract_percentage(response, r"P\(\+5%[^:]*\):\s*(\d+)")
            p_up_10 = self._extract_percentage(response, r"P\(\+10%[^:]*\):\s*(\d+)")
            p_down_5 = self._extract_percentage(response, r"P\(-5%[^:]*\):\s*(\d+)")
            p_down_10 = self._extract_percentage(response, r"P\(-10%[^:]*\):\s*(\d+)")

            # Extract confidence
            confidence_match = re.search(r"CONFIDENCE[:\s]*(low|medium|high)", response, re.I)
            confidence = confidence_match.group(1).lower() if confidence_match else "medium"

            # Extract key factors (simple extraction)
            factors = re.findall(r"[-•]\s*(.{10,100})", response)[:5]

            return ProbabilityEstimate(
                provider=provider,
                p_up_5=p_up_5 / 100 if p_up_5 else 0.5,
                p_up_10=p_up_10 / 100 if p_up_10 else 0.3,
                p_down_5=p_down_5 / 100 if p_down_5 else 0.3,
                p_down_10=p_down_10 / 100 if p_down_10 else 0.15,
                confidence=confidence,
                key_factors=factors,
                risks=[],
                raw_response=response,
            )

        except Exception as e:
            print(f"Error parsing estimate from {provider}: {e}")
            return None

    def _extract_percentage(self, text: str, pattern: str) -> Optional[float]:
        """Extract percentage from text using regex"""
        match = re.search(pattern, text, re.I)
        if match:
            return float(match.group(1))
        return None

    def _calculate_spread(self, estimates: dict) -> float:
        """Calculate max spread between LLM estimates"""
        if len(estimates) < 2:
            return 0.0

        spreads = []
        for metric in ["p_up_5", "p_up_10", "p_down_5", "p_down_10"]:
            values = [getattr(e, metric) for e in estimates.values()]
            if values:
                spreads.append(max(values) - min(values))

        return max(spreads) if spreads else 0.0

    def _calculate_final_estimate(self, estimates: dict) -> dict:
        """Calculate final consensus estimate (average)"""
        if not estimates:
            return {}

        n = len(estimates)
        return {
            "p_up_5": sum(e.p_up_5 for e in estimates.values()) / n,
            "p_up_10": sum(e.p_up_10 for e in estimates.values()) / n,
            "p_down_5": sum(e.p_down_5 for e in estimates.values()) / n,
            "p_down_10": sum(e.p_down_10 for e in estimates.values()) / n,
        }

    def _generate_reasoning(self, estimates: dict) -> str:
        """Generate summary reasoning from all estimates"""
        if not estimates:
            return "No estimates available."

        lines = ["## Consensus Reasoning", ""]

        for provider, est in estimates.items():
            lines.append(f"### {provider.upper()}")
            lines.append(f"Key factors identified:")
            for factor in est.key_factors[:3]:
                lines.append(f"- {factor}")
            lines.append("")

        return "\n".join(lines)

    def _generate_challenge_points(self, ticker: str, previous_responses: dict) -> list:
        """Generate challenge points if no callback provided"""
        # Default challenge points based on common blind spots
        return [
            f"Recent insider trading activity for {ticker}",
            f"Options market sentiment (put/call ratio) for {ticker}",
            f"Supply chain disruption risks not mentioned",
            f"Currency exposure and hedging strategy",
            f"Recent changes in institutional ownership",
        ]
