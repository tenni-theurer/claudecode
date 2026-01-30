"""
Multi-LLM Query Engine
Queries Claude, GPT-4, and Gemini in parallel
"""
import anthropic
from openai import OpenAI
from google import genai
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY,
    MODELS, cost_tracker
)


class MultiLLM:
    """
    Query multiple LLMs and aggregate responses
    """

    def __init__(self):
        self.clients = {}

        if ANTHROPIC_API_KEY:
            self.clients["claude"] = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        if OPENAI_API_KEY:
            self.clients["gpt"] = OpenAI(api_key=OPENAI_API_KEY)

        if GOOGLE_API_KEY:
            self.clients["gemini"] = genai.Client(api_key=GOOGLE_API_KEY)

    def query_all(self, prompt: str, system: str = None) -> dict:
        """
        Query all available LLMs with the same prompt

        Returns dict with responses from each LLM
        """
        results = {}

        # Use ThreadPoolExecutor for parallel queries
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}

            if "claude" in self.clients:
                futures[executor.submit(self._query_claude, prompt, system)] = "claude"

            if "gpt" in self.clients:
                futures[executor.submit(self._query_gpt, prompt, system)] = "gpt"

            if "gemini" in self.clients:
                futures[executor.submit(self._query_gemini, prompt, system)] = "gemini"

            for future in as_completed(futures):
                provider = futures[future]
                try:
                    results[provider] = future.result()
                except Exception as e:
                    results[provider] = {"error": str(e), "response": None}

        return results

    def query_single(self, provider: str, prompt: str, system: str = None) -> dict:
        """Query a single LLM"""
        if provider == "claude":
            return self._query_claude(prompt, system)
        elif provider == "gpt":
            return self._query_gpt(prompt, system)
        elif provider == "gemini":
            return self._query_gemini(prompt, system)
        else:
            return {"error": f"Unknown provider: {provider}", "response": None}

    def _query_claude(self, prompt: str, system: str = None) -> dict:
        """Query Claude"""
        try:
            messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": MODELS["claude"],
                "max_tokens": 4000,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system

            response = self.clients["claude"].messages.create(**kwargs)

            # Track cost (approximate)
            cost = (response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000
            cost_tracker.add("claude", cost)

            return {
                "response": response.content[0].text,
                "tokens": {
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                },
                "cost": cost,
            }

        except Exception as e:
            return {"error": str(e), "response": None}

    def _query_gpt(self, prompt: str, system: str = None) -> dict:
        """Query GPT-4"""
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = self.clients["gpt"].chat.completions.create(
                model=MODELS["gpt"],
                messages=messages,
                max_tokens=4000,
            )

            # Track cost (approximate for GPT-4o)
            cost = (response.usage.prompt_tokens * 2.5 + response.usage.completion_tokens * 10) / 1_000_000
            cost_tracker.add("gpt", cost)

            return {
                "response": response.choices[0].message.content,
                "tokens": {
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                },
                "cost": cost,
            }

        except Exception as e:
            return {"error": str(e), "response": None}

    def _query_gemini(self, prompt: str, system: str = None) -> dict:
        """Query Gemini"""
        try:
            full_prompt = f"{system}\n\n{prompt}" if system else prompt

            response = self.clients["gemini"].models.generate_content(
                model=MODELS["gemini"],
                contents=full_prompt,
            )

            # Track cost (approximate for Gemini 1.5 Pro)
            cost = 0.01  # Rough estimate
            cost_tracker.add("gemini", cost)

            return {
                "response": response.text,
                "tokens": {"input": 0, "output": 0},
                "cost": cost,
            }

        except Exception as e:
            return {"error": str(e), "response": None}

    def get_available_providers(self) -> list:
        """Get list of available LLM providers"""
        return list(self.clients.keys())


def build_probability_prompt(ticker: str, data: dict, context: str) -> str:
    """
    Build a prompt asking for probability estimates

    Asks for P(+5%), P(+10%), P(-5%), P(-10%) over next 3 months
    """
    return f"""Based on the following analysis of {ticker}, provide probability estimates for stock price movement over the NEXT 3 MONTHS.

{context}

Provide your estimates in this exact format:
- P(+5% or more): XX%
- P(+10% or more): XX%
- P(-5% or more): XX%
- P(-10% or more): XX%

Also provide:
- CONFIDENCE in your estimates (low/medium/high)
- KEY FACTORS driving your estimate (top 3)
- BIGGEST RISKS that could invalidate this

Be specific with percentages. Do not hedge - commit to your best estimate."""


def build_challenge_prompt(
    ticker: str,
    responses: dict,
    missed_data: list[str]
) -> str:
    """
    Build a prompt to challenge LLM responses with missed data

    Args:
        ticker: Stock ticker
        responses: Previous responses from each LLM
        missed_data: List of data points the LLMs missed
    """
    response_summary = ""
    for provider, resp in responses.items():
        if resp.get("response"):
            response_summary += f"\n### {provider.upper()}'s Analysis:\n{resp['response'][:1000]}...\n"

    missed_points = "\n".join(f"- {point}" for point in missed_data)

    return f"""Previous analysis of {ticker} missed these important data points:

{missed_points}

Previous responses:
{response_summary}

Given this NEW information, revise your probability estimates:
- P(+5% or more): XX%
- P(+10% or more): XX%
- P(-5% or more): XX%
- P(-10% or more): XX%

Explain how each missed data point affects your estimate. Be specific."""
