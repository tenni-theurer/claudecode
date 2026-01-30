#!/usr/bin/env python3
"""
Forward Alpha v2 - Multi-Events Collisions Engine
Autonomous equity analysis using multi-LLM consensus (Claude, GPT-4, Gemini, Grok)

Usage:
    # Collision Analysis (primary feature)
    python main.py collision NVDA                    # Initial probability estimate
    python main.py collision NVDA "event 1" "event 2"  # Inject events, see deltas
    python main.py collision-news AAPL               # Auto-inject recent news
    python main.py news AAPL                         # View news as injection candidates

    # Other commands
    python main.py analyze NVDA          # Full analysis of a ticker
    python main.py gate                  # Check index gate (market conditions)
    python main.py roller NVDA TSM AAPL  # Roller coaster - find peaks/bottoms
    python main.py events                # Show active events
    python main.py portfolio             # Show portfolio status
    python main.py jump                  # Get jump suggestions
"""
import sys
import json
from datetime import datetime

from config import (
    REPORTS_DIR, cost_tracker, DATA_CATEGORIES,
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
)
from core.data_collector import collect_stock_data, build_analysis_prompt
from core.events import EventTracker, initialize_events
from llm.multi_llm import MultiLLM
from llm.consensus import ConsensusBuilder
from strategy.roller_coaster import RollerCoaster
from strategy.position import PositionManager
from strategy.index_gate import IndexGate
from strategy.collision import CollisionEngine


def generate_markdown_report(report: dict) -> str:
    """Generate a detailed Markdown report"""
    ticker = report["ticker"]
    company = report.get("company", {})
    index_gate = report.get("index_gate", {})
    rc = report.get("roller_coaster", {})
    events = report.get("events", {})
    consensus = report.get("consensus", {})
    categories = report.get("category_analyses", {})
    rec = report.get("recommendation", {})

    lines = [
        f"# Forward Alpha Analysis: {ticker}",
        f"*Generated: {report['generated_at']}*",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"**Recommendation: {rec.get('action', 'N/A')}**",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Expected Value | {rec.get('expected_value', 0):+.1f}% |",
        f"| P(+10%) | {consensus.get('estimates', {}).get('p_up_10', 0)*100:.0f}% |",
        f"| P(-10%) | {consensus.get('estimates', {}).get('p_down_10', 0)*100:.0f}% |",
        f"| Index Gate | {'OPEN' if index_gate.get('open') else 'CLOSED'} |",
        f"| Position | {rc.get('position', 'N/A').upper()} ({rc.get('percentile', 0):.0f}th %ile) |",
        "",
        "---",
        "",
        "## Company Overview",
        "",
        f"**{company.get('name', ticker)}**",
        "",
        f"- Sector: {company.get('sector') or 'N/A'}",
        f"- Industry: {company.get('industry') or 'N/A'}",
    ]

    financials = company.get("financials", {})
    if financials.get("price"):
        lines.append(f"- Price: ${financials['price']:.2f}")
    if financials.get("market_cap"):
        lines.append(f"- Market Cap: ${financials['market_cap']/1e9:.1f}B")
    if financials.get("pe_forward"):
        lines.append(f"- Forward P/E: {financials['pe_forward']:.1f}")

    lines.extend([
        "",
        "---",
        "",
        "## Index Gate Analysis",
        "",
        f"**Status: {'OPEN' if index_gate.get('open') else 'CLOSED'}**",
        "",
        "| Probability | Estimate |",
        "|-------------|----------|",
        f"| P(+10% in 12mo) | {index_gate.get('p_up_10', 0)*100:.0f}% |",
        f"| P(+5% in 12mo) | {index_gate.get('p_up_5', 0)*100:.0f}% |",
        f"| P(-5% in 12mo) | {index_gate.get('p_down_5', 0)*100:.0f}% |",
        f"| P(-10% in 12mo) | {index_gate.get('p_down_10', 0)*100:.0f}% |",
        "",
    ])

    if index_gate.get("bullish_factors"):
        lines.append("**Bullish Factors:**")
        for f in index_gate["bullish_factors"]:
            lines.append(f"- {f}")
        lines.append("")

    if index_gate.get("bearish_factors"):
        lines.append("**Bearish Factors:**")
        for f in index_gate["bearish_factors"]:
            lines.append(f"- {f}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Roller Coaster Position",
        "",
        f"**Position: {rc.get('position', 'N/A').upper()}**",
        "",
        f"- 52-Week Percentile: {rc.get('percentile', 0):.1f}%",
        f"- 30-Day Momentum: {rc.get('momentum_30d', 0):+.1f}%",
        f"- 7-Day Momentum: {rc.get('momentum_7d', 0):+.1f}%",
        f"- Days to Earnings: {rc.get('days_to_earnings') or 'N/A'}",
        "",
        f"**Recommendation:** {rc.get('recommendation', 'N/A')}",
        "",
        f"*{rc.get('reasoning', '')}*",
        "",
        "---",
        "",
        "## Event Analysis",
        "",
        f"**Net Sentiment: {events.get('sentiment', 'N/A').upper()}**",
        "",
        f"- Total Weight: {events.get('total_weight', 0):+.2f}",
        f"- Bullish Events: {events.get('bullish_count', 0)}",
        f"- Bearish Events: {events.get('bearish_count', 0)}",
        "",
    ])

    if events.get("top_bullish"):
        lines.append("**Top Bullish Events:**")
        for e in events["top_bullish"]:
            lines.append(f"- [{e['category']}] {e['description']} ({e['weight']:+.2f})")
        lines.append("")

    if events.get("top_bearish"):
        lines.append("**Top Bearish Events:**")
        for e in events["top_bearish"]:
            lines.append(f"- [{e['category']}] {e['description']} ({e['weight']:+.2f})")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Probability Estimates (3-Month Horizon)",
        "",
        f"**Consensus: {'Reached' if consensus.get('reached') else 'Not Reached'}** (Rounds: {consensus.get('rounds', 0)}, Spread: {consensus.get('spread', 0)*100:.1f}%)",
        "",
        "| Probability | Estimate |",
        "|-------------|----------|",
        f"| P(+5% or more) | {consensus.get('estimates', {}).get('p_up_5', 0)*100:.0f}% |",
        f"| P(+10% or more) | {consensus.get('estimates', {}).get('p_up_10', 0)*100:.0f}% |",
        f"| P(-5% or more) | {consensus.get('estimates', {}).get('p_down_5', 0)*100:.0f}% |",
        f"| P(-10% or more) | {consensus.get('estimates', {}).get('p_down_10', 0)*100:.0f}% |",
        "",
    ])

    if consensus.get("individual_estimates"):
        lines.append("### Individual LLM Estimates")
        lines.append("")
        for provider, est in consensus["individual_estimates"].items():
            lines.append(f"**{provider.upper()}:**")
            lines.append(f"- P(+5%): {est['p_up_5']*100:.0f}% | P(+10%): {est['p_up_10']*100:.0f}%")
            lines.append(f"- P(-5%): {est['p_down_5']*100:.0f}% | P(-10%): {est['p_down_10']*100:.0f}%")
            lines.append(f"- Confidence: {est['confidence']}")
            if est.get("key_factors"):
                lines.append("- Key Factors:")
                for factor in est["key_factors"][:3]:
                    lines.append(f"  - {factor}")
            lines.append("")

    if consensus.get("reasoning"):
        lines.extend([
            "### Consensus Reasoning",
            "",
            consensus["reasoning"],
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Category Analyses",
        "",
    ])

    for category, analysis in categories.items():
        if analysis:
            lines.append(f"### {category.replace('_', ' ').title()}")
            lines.append("")
            lines.append(analysis[:2000] + "..." if len(analysis) > 2000 else analysis)
            lines.append("")

    lines.extend([
        "---",
        "",
        "## Audit Trail",
        "",
        f"- Total API Calls: {report.get('costs', {}).get('calls', 0)}",
        f"- Total Cost: ${report.get('costs', {}).get('total', 0):.4f}",
    ])

    for provider, cost in report.get("costs", {}).get("by_provider", {}).items():
        if cost > 0:
            lines.append(f"- {provider.title()}: ${cost:.4f}")

    return "\n".join(lines)


def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_section(text: str):
    """Print a section header"""
    print(f"\n--- {text} ---")


def check_api_keys():
    """Check which API keys are available"""
    available = []
    if ANTHROPIC_API_KEY:
        available.append("Claude")
    if OPENAI_API_KEY:
        available.append("GPT-4")
    if GOOGLE_API_KEY:
        available.append("Gemini")

    if not available:
        print("ERROR: No API keys configured!")
        print("Please set at least one of:")
        print("  - ANTHROPIC_API_KEY")
        print("  - OPENAI_API_KEY")
        print("  - GOOGLE_API_KEY")
        sys.exit(1)

    print(f"Available LLMs: {', '.join(available)}")
    return available


def cmd_analyze(ticker: str):
    """
    Full analysis of a ticker using multi-LLM consensus

    1. Check index gate first
    2. Collect data across 11 categories
    3. Query LLMs for category analyses
    4. Build consensus on probability estimates
    5. Generate report
    """
    print_header(f"FORWARD ALPHA ANALYSIS: {ticker.upper()}")
    check_api_keys()

    # Initialize components
    event_tracker = initialize_events()
    llm = MultiLLM()
    consensus_builder = ConsensusBuilder()
    roller_coaster = RollerCoaster()

    # Step 1: Check Index Gate
    print_section("Index Gate Check")
    gate = IndexGate()
    forecast = gate.check_gate()
    print(f"Index ({forecast.ticker}): P(+10% in 12mo) = {forecast.p_up_10*100:.0f}%")
    print(f"Gate Status: {'OPEN' if forecast.gate_open else 'CLOSED'}")
    if not forecast.gate_open:
        print("WARNING: Index gate is closed. Proceeding with analysis but recommend caution.")

    # Step 2: Collect stock data
    print_section("Collecting Stock Data")
    data = collect_stock_data(ticker)
    print(f"Company: {data['name']}")
    print(f"Sector: {data.get('sector') or 'N/A'} | Industry: {data.get('industry') or 'N/A'}")
    price = data['financials'].get('price')
    print(f"Price: ${price:.2f}" if price else "Price: N/A")
    market_cap = data['financials'].get('market_cap')
    print(f"Market Cap: ${market_cap/1e9:.1f}B" if market_cap else "Market Cap: N/A")

    # Step 3: Roller coaster position
    print_section("Roller Coaster Analysis")
    rc_status = roller_coaster.analyze_ticker(ticker)
    print(f"Position: {rc_status.position.upper()}")
    print(f"52-Week Percentile: {rc_status.percentile_52w}%")
    print(f"30-Day Momentum: {rc_status.momentum_30d:+.1f}%")
    print(f"Recommendation: {rc_status.recommendation}")
    print(f"Reasoning: {rc_status.reasoning}")

    # Step 4: Get relevant events
    print_section("Event Analysis")
    events = event_tracker.calculate_aggregate_weight(ticker)
    print(f"Net Event Sentiment: {events['net_sentiment'].upper()}")
    print(f"Bullish Events: {events['bullish_count']} | Bearish: {events['bearish_count']}")
    if events['top_bullish']:
        print("Top Bullish:")
        for e in events['top_bullish'][:2]:
            print(f"  + {e.description} ({e.weight:+.2f})")
    if events['top_bearish']:
        print("Top Bearish:")
        for e in events['top_bearish'][:2]:
            print(f"  - {e.description} ({e.weight:+.2f})")

    # Step 5: Category analysis (query LLMs for each category)
    print_section("Category Analysis (11 Categories)")
    category_analyses = {}

    for i, category in enumerate(DATA_CATEGORIES, 1):
        print(f"  [{i}/11] Analyzing {category}...", end=" ", flush=True)

        prompt = build_analysis_prompt(ticker, data, category)
        responses = llm.query_all(prompt)

        # Combine responses
        combined = []
        for provider, resp in responses.items():
            if resp.get("response"):
                combined.append(f"[{provider.upper()}]: {resp['response'][:500]}")

        category_analyses[category] = "\n\n".join(combined) if combined else None
        print("Done" if combined else "Failed")

    # Step 6: Build consensus on probability estimates
    print_section("Building Multi-LLM Consensus")

    def missed_data_callback(ticker, previous):
        # Generate challenge points based on what was missed
        return [
            f"Recent insider trading activity for {ticker}",
            f"Put/call ratio and options sentiment",
            f"Short interest trends",
            f"Recent analyst rating changes",
            f"Supply chain risk factors",
        ]

    consensus = consensus_builder.build_consensus(
        ticker=ticker,
        data=data,
        category_analyses=category_analyses,
        missed_data_callback=missed_data_callback,
    )

    print(f"Consensus Reached: {'Yes' if consensus.reached else 'No'}")
    print(f"Rounds: {consensus.rounds}")
    print(f"Spread: {consensus.spread*100:.1f}%")

    # Step 7: Display results
    print_section("PROBABILITY ESTIMATES (3-month horizon)")
    fe = consensus.final_estimate
    if fe:
        print(f"  P(+5% or more):  {fe.get('p_up_5', 0)*100:.0f}%")
        print(f"  P(+10% or more): {fe.get('p_up_10', 0)*100:.0f}%")
        print(f"  P(-5% or more):  {fe.get('p_down_5', 0)*100:.0f}%")
        print(f"  P(-10% or more): {fe.get('p_down_10', 0)*100:.0f}%")
    else:
        print("  Unable to generate estimates")

    # Step 8: Generate recommendation
    print_section("RECOMMENDATION")
    if fe:
        upside_downside = (fe.get('p_up_10', 0) * 10 - fe.get('p_down_10', 0) * 10)
        expected_value = upside_downside

        if not forecast.gate_open:
            print("STAY OUT - Index gate closed")
            print("Wait for better macro conditions before entering positions.")
        elif rc_status.recommendation == "sell_to_jump":
            print("CONSIDER SELLING - Stock at peak")
            print("Look for recovery candidates to jump to.")
        elif expected_value > 2:
            print("BUY - Positive expected value")
            print(f"Expected value: {expected_value:+.1f}%")
        elif expected_value < -2:
            print("AVOID - Negative expected value")
            print(f"Expected value: {expected_value:+.1f}%")
        else:
            print("NEUTRAL - Limited edge")
            print("No strong signal, consider alternatives.")

    # Step 9: Save detailed report
    report = {
        "ticker": ticker,
        "generated_at": datetime.now().isoformat(),
        "company": {
            "name": data["name"],
            "sector": data.get("sector"),
            "industry": data.get("industry"),
            "financials": data["financials"],
            "price_position": data.get("price_position"),
            "earnings": data.get("earnings"),
        },
        "index_gate": {
            "open": forecast.gate_open,
            "p_up_10": forecast.p_up_10,
            "p_up_5": forecast.p_up_5,
            "p_down_5": forecast.p_down_5,
            "p_down_10": forecast.p_down_10,
            "confidence": forecast.confidence,
            "bullish_factors": forecast.key_bullish_factors,
            "bearish_factors": forecast.key_bearish_factors,
            "reasoning": forecast.reasoning,
        },
        "roller_coaster": {
            "position": rc_status.position,
            "percentile": rc_status.percentile_52w,
            "momentum_30d": rc_status.momentum_30d,
            "momentum_7d": rc_status.momentum_7d,
            "days_to_earnings": rc_status.days_to_earnings,
            "recommendation": rc_status.recommendation,
            "reasoning": rc_status.reasoning,
        },
        "events": {
            "sentiment": events["net_sentiment"],
            "total_weight": events["total_weight"],
            "bullish_count": events["bullish_count"],
            "bearish_count": events["bearish_count"],
            "top_bullish": [
                {"description": e.description, "weight": e.weight, "category": e.category}
                for e in events["top_bullish"]
            ],
            "top_bearish": [
                {"description": e.description, "weight": e.weight, "category": e.category}
                for e in events["top_bearish"]
            ],
        },
        "category_analyses": category_analyses,
        "consensus": {
            "reached": consensus.reached,
            "rounds": consensus.rounds,
            "spread": consensus.spread,
            "estimates": consensus.final_estimate,
            "reasoning": consensus.reasoning,
            "individual_estimates": {
                provider: {
                    "p_up_5": est.p_up_5,
                    "p_up_10": est.p_up_10,
                    "p_down_5": est.p_down_5,
                    "p_down_10": est.p_down_10,
                    "confidence": est.confidence,
                    "key_factors": est.key_factors,
                }
                for provider, est in consensus.estimates.items()
            } if consensus.estimates else {},
        },
        "recommendation": {
            "action": "STAY_OUT" if not forecast.gate_open else
                     "SELL_TO_JUMP" if rc_status.recommendation == "sell_to_jump" else
                     "BUY" if fe and (fe.get('p_up_10', 0) * 10 - fe.get('p_down_10', 0) * 10) > 2 else
                     "AVOID" if fe and (fe.get('p_up_10', 0) * 10 - fe.get('p_down_10', 0) * 10) < -2 else
                     "NEUTRAL",
            "expected_value": (fe.get('p_up_10', 0) * 10 - fe.get('p_down_10', 0) * 10) if fe else 0,
        },
        "costs": {
            "total": cost_tracker.total,
            "calls": cost_tracker.total_calls,
            "by_provider": cost_tracker.costs,
        },
    }

    # Save JSON report
    report_path = REPORTS_DIR / f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save Markdown report
    md_path = REPORTS_DIR / f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(md_path, "w") as f:
        f.write(generate_markdown_report(report))

    print_section("REPORTS SAVED")
    print(f"JSON: {report_path}")
    print(f"Markdown: {md_path}")
    print(f"API Costs: ${cost_tracker.total:.2f} ({cost_tracker.total_calls} calls)")


def cmd_gate():
    """Check index gate status"""
    print_header("INDEX GATE CHECK")
    check_api_keys()

    gate = IndexGate()
    forecast = gate.check_gate()

    print(f"\nIndex: {forecast.ticker}")
    print(f"Current Price: ${forecast.current_price:.2f}")
    print(f"Forecast Horizon: {forecast.horizon_months} months")
    print(f"\nProbability Estimates:")
    print(f"  P(+10% or more): {forecast.p_up_10*100:.0f}%")
    print(f"  P(+5% or more):  {forecast.p_up_5*100:.0f}%")
    print(f"  P(-5% or more):  {forecast.p_down_5*100:.0f}%")
    print(f"  P(-10% or more): {forecast.p_down_10*100:.0f}%")
    print(f"\nGate Status: {'OPEN' if forecast.gate_open else 'CLOSED'}")
    print(f"Confidence: {forecast.confidence.upper()}")
    print(f"\n{forecast.reasoning}")


def cmd_roller(*tickers):
    """Roller coaster analysis for multiple tickers"""
    if not tickers:
        print("Usage: python main.py roller NVDA TSM AAPL")
        return

    print_header("ROLLER COASTER ANALYSIS")

    roller = RollerCoaster()
    summary = roller.get_watchlist_summary(list(tickers))

    print("\n[AT PEAK] (consider selling):")
    for s in summary["peaks"]:
        print(f"  {s.ticker}: {s.percentile_52w}% | {s.recommendation}")

    print("\n[AT BOTTOM] (consider buying):")
    for s in summary["bottoms"]:
        print(f"  {s.ticker}: {s.percentile_52w}% | {s.recommendation}")

    print("\n[CLIMBING]:")
    for s in summary["climbing"]:
        print(f"  {s.ticker}: {s.percentile_52w}% | +{s.momentum_30d:.1f}% (30d)")

    print("\n[FALLING]:")
    for s in summary["falling"]:
        print(f"  {s.ticker}: {s.percentile_52w}% | {s.momentum_30d:.1f}% (30d)")

    if summary["jump_opportunities"]:
        print("\n[JUMP OPPORTUNITIES]:")
        for jump in summary["jump_opportunities"][:5]:
            print(f"  {jump['from']} -> {jump['to']} (score: {jump['score']:.0f})")
            print(f"    {jump['reason']}")


def cmd_events():
    """Show active events"""
    print_header("ACTIVE EVENTS")

    tracker = initialize_events()
    summary = tracker.get_summary()

    print(f"\nTotal Events: {summary['total_events']}")
    print(f"Active Events: {summary['active_events']}")

    print("\nBy Category:")
    for cat, data in summary["by_category"].items():
        if data["count"] > 0:
            print(f"  {cat}: {data['count']} events (avg weight: {data['avg_weight']:+.2f})")

    active = tracker.get_active_events(30)
    if active:
        print("\nRecent Active Events:")
        for e in active[:10]:
            print(f"  [{e.category}] {e.description} ({e.weight:+.2f})")


def cmd_portfolio():
    """Show portfolio status"""
    print_header("PORTFOLIO STATUS")

    manager = PositionManager()
    summary = manager.get_portfolio_summary()

    if summary["positions_count"] == 0:
        print("\nNo positions. Use 'add_position' to add holdings.")
        return

    print(f"\nTotal Value: ${summary['total_value']:,.2f}")
    print(f"Total Cost: ${summary['total_cost']:,.2f}")
    print(f"Unrealized P&L: ${summary['total_pnl']:,.2f} ({summary['total_pnl_pct']:+.1f}%)")
    print(f"Target: Beat index by {summary['beat_target']}%")

    print("\nPositions:")
    for ticker, pos in summary.get("positions", {}).items():
        print(f"  {ticker}: {pos['shares']:.2f} shares @ ${pos['entry']:.2f}")
        print(f"    Current: ${pos['current'] or 'N/A'} | P&L: {pos['pnl_pct']:+.1f}%")
        print(f"    Status: {pos['status'] or 'unknown'}")


def cmd_jump():
    """Get jump suggestions for current positions"""
    print_header("JUMP SUGGESTIONS")

    manager = PositionManager()

    if not manager.positions:
        print("\nNo positions to jump from.")
        return

    # Default watchlist - could be configurable
    watchlist = ["NVDA", "TSM", "AAPL", "MSFT", "GOOGL", "META", "AMD", "AVGO", "ASML"]

    suggestions = manager.get_jump_suggestions(watchlist)

    if not suggestions:
        print("\nNo jump opportunities found.")
        return

    print("\nRecommended Jumps:")
    for i, s in enumerate(suggestions[:5], 1):
        print(f"\n{i}. {s['from']} ({s['from_status']}) -> {s['to']} ({s['to_status']})")
        print(f"   Score: {s['score']:.0f}")
        print(f"   Reason: {s['reason']}")


def cmd_collision(ticker: str, *new_events):
    """
    Interactive collision analysis - inject events and see probability changes

    Usage:
        python main.py collision NVDA
        python main.py collision NVDA "China approves only 33% of H200 sales" "TSMC Arizona producing Blackwell"
    """
    print_header(f"COLLISION ANALYSIS: {ticker}")
    check_api_keys()

    # Collect stock data
    print_section("Collecting Stock Data")
    data = collect_stock_data(ticker)
    print(f"Company: {data['name']}")
    print(f"Price: ${data['financials'].get('price') or 'N/A'}")

    earnings = data.get("earnings", {})
    if earnings.get("days_to_earnings"):
        print(f"Days to Earnings: {earnings['days_to_earnings']}")
        horizon = "by earnings date"
    else:
        horizon = "over next 3 months"

    # Initialize collision engine
    engine = CollisionEngine()
    engine.initialize(ticker, data, horizon)

    # Get initial estimate
    print_section("Initial Probability Estimate")
    print(f"(Modeling {ticker} as a ball, identifying collision factors...)")

    initial = engine.get_initial_estimate(horizon)

    # Show individual LLM estimates
    individual = engine.get_individual_estimates()
    if individual:
        print(f"\nINDIVIDUAL LLM ESTIMATES ({horizon}):")
        print("-" * 60)
        print(f"{'LLM':<10} {'P(+5%)':<10} {'P(+10%)':<10} {'P(-5%)':<10} {'P(-10%)':<10}")
        print("-" * 60)
        for provider, est in individual.items():
            if "error" in est:
                print(f"{provider.upper():<10} {'[' + est['error'] + ']':<40}")
            else:
                print(f"{provider.upper():<10} {est['p_up_5']*100:>6.0f}%   {est['p_up_10']*100:>6.0f}%    {est['p_down_5']*100:>6.0f}%    {est['p_down_10']*100:>6.0f}%")
        print("-" * 60)

    print(f"\nCONSENSUS ESTIMATE ({horizon}):")
    print(f"  P(+5% or more):  {initial.p_up_5*100:.0f}%")
    print(f"  P(+10% or more): {initial.p_up_10*100:.0f}%")
    print(f"  P(-5% or more):  {initial.p_down_5*100:.0f}%")
    print(f"  P(-10% or more): {initial.p_down_10*100:.0f}%")
    print(f"\nConfidence: {initial.confidence.upper()}")

    if initial.reasoning:
        print(f"\nReasoning:\n{initial.reasoning[:800]}...")

    # If new events provided, inject and recalculate
    if new_events:
        print_section("Injecting New Data")
        for event in new_events:
            injected = engine.inject_event(event)
            print(f"  + [{injected['id']}] {event}")

        print_section("Recalculating Probabilities")
        print("(Analyzing how new data changes collision dynamics...)")

        result = engine.recalculate()
        new_state = result["new_state"]
        deltas = result["deltas"]

        # Show individual LLM estimates after recalculation
        individual = engine.get_individual_estimates()
        if individual:
            print(f"\nINDIVIDUAL LLM REVISED ESTIMATES:")
            print("-" * 60)
            print(f"{'LLM':<10} {'P(+5%)':<10} {'P(+10%)':<10} {'P(-5%)':<10} {'P(-10%)':<10}")
            print("-" * 60)
            for provider, est in individual.items():
                if "error" in est:
                    print(f"{provider.upper():<10} {'[' + est['error'] + ']':<40}")
                else:
                    print(f"{provider.upper():<10} {est['p_up_5']*100:>6.0f}%   {est['p_up_10']*100:>6.0f}%    {est['p_down_5']*100:>6.0f}%    {est['p_down_10']*100:>6.0f}%")
            print("-" * 60)

        print(f"\nCONSENSUS REVISED ESTIMATES:")
        print(f"  P(+5% or more):  {new_state.p_up_5*100:.0f}% ({deltas['p_up_5']:+.1f} pp)")
        print(f"  P(+10% or more): {new_state.p_up_10*100:.0f}% ({deltas['p_up_10']:+.1f} pp)")
        print(f"  P(-5% or more):  {new_state.p_down_5*100:.0f}% ({deltas['p_down_5']:+.1f} pp)")
        print(f"  P(-10% or more): {new_state.p_down_10*100:.0f}% ({deltas['p_down_10']:+.1f} pp)")

        # Net effect
        net_shift = deltas["p_up_10"] - deltas["p_down_10"]
        if net_shift > 5:
            print(f"\n>>> NET EFFECT: BULLISH (+{net_shift:.1f}% shift toward upside)")
        elif net_shift < -5:
            print(f"\n>>> NET EFFECT: BEARISH ({net_shift:.1f}% shift toward downside)")
        else:
            print(f"\n>>> NET EFFECT: NEUTRAL ({net_shift:+.1f}% shift)")

        print(f"\n{result['analysis']}")

        # Save session and markdown report
        session_path = engine.save_session()
        md_path = engine.save_markdown_report()
        print(f"\nSession saved: {session_path}")
        print(f"Full report: {md_path}")

    else:
        # No events - save initial analysis
        session_path = engine.save_session()
        md_path = engine.save_markdown_report()
        print(f"\nSession saved: {session_path}")
        print(f"Full report: {md_path}")

        print("\n" + "="*50)
        print("To inject new data and recalculate, run:")
        print(f'  python3 main.py collision {ticker} "your new data point"')
        print(f'  python3 main.py news {ticker}  # Get recent news as injection candidates')
        print("="*50)


def cmd_news(ticker: str):
    """
    Collect recent news and show as injection candidates

    Usage:
        python main.py news AAPL
    """
    from data.news_collector import collect_injection_candidates
    from core.data_collector import collect_stock_data

    print_header(f"NEWS INJECTION CANDIDATES: {ticker}")

    # Get company name
    data = collect_stock_data(ticker)
    company_name = data.get("name", ticker)

    print(f"Collecting recent news for {company_name}...\n")

    candidates = collect_injection_candidates(ticker, company_name)

    if not candidates:
        print("No recent news found.")
        print("\nTip: Set NEWS_API_KEY in .env for more news sources.")
        return

    print(f"Found {len(candidates)} recent news items:\n")
    for i, c in enumerate(candidates, 1):
        print(f"  {i}. {c}")

    print("\n" + "="*60)
    print("To inject these into collision analysis, copy the headlines:")
    print(f'  python3 main.py collision {ticker} "headline 1" "headline 2"')
    print("="*60)

    return candidates


def cmd_collision_with_news(ticker: str):
    """
    Run collision analysis then offer to inject recent news

    Usage:
        python main.py collision-news AAPL
    """
    from data.news_collector import collect_injection_candidates
    from core.data_collector import collect_stock_data

    print_header(f"COLLISION ANALYSIS WITH NEWS: {ticker}")
    check_api_keys()

    # Collect stock data
    print_section("Collecting Stock Data")
    data = collect_stock_data(ticker)
    company_name = data.get("name", ticker)
    print(f"Company: {company_name}")
    print(f"Price: ${data['financials'].get('price') or 'N/A'}")

    earnings = data.get("earnings", {})
    if earnings.get("days_to_earnings"):
        print(f"Days to Earnings: {earnings['days_to_earnings']}")
        horizon = "by earnings date"
    else:
        horizon = "over next 3 months"

    # Initialize collision engine
    engine = CollisionEngine()
    engine.initialize(ticker, data, horizon)

    # Get initial estimate
    print_section("Initial Probability Estimate")
    print(f"(Modeling {ticker} as a ball, identifying collision factors...)")

    initial = engine.get_initial_estimate(horizon)

    # Show individual LLM estimates
    individual = engine.get_individual_estimates()
    if individual:
        print(f"\nINDIVIDUAL LLM ESTIMATES ({horizon}):")
        print("-" * 60)
        print(f"{'LLM':<10} {'P(+5%)':<10} {'P(+10%)':<10} {'P(-5%)':<10} {'P(-10%)':<10}")
        print("-" * 60)
        for provider, est in individual.items():
            if "error" in est:
                print(f"{provider.upper():<10} {'[' + est['error'][:30] + ']':<40}")
            else:
                print(f"{provider.upper():<10} {est['p_up_5']*100:>6.0f}%   {est['p_up_10']*100:>6.0f}%    {est['p_down_5']*100:>6.0f}%    {est['p_down_10']*100:>6.0f}%")
        print("-" * 60)

    print(f"\nCONSENSUS ESTIMATE ({horizon}):")
    print(f"  P(+5% or more):  {initial.p_up_5*100:.0f}%")
    print(f"  P(+10% or more): {initial.p_up_10*100:.0f}%")
    print(f"  P(-5% or more):  {initial.p_down_5*100:.0f}%")
    print(f"  P(-10% or more): {initial.p_down_10*100:.0f}%")

    # Collect news
    print_section("Collecting Recent News")
    candidates = collect_injection_candidates(ticker, company_name)

    if candidates:
        print(f"Found {len(candidates)} recent news items:\n")
        for i, c in enumerate(candidates[:10], 1):  # Show top 10
            print(f"  {i}. {c}")

        if len(candidates) > 10:
            print(f"  ... and {len(candidates) - 10} more")

        # Inject all news and recalculate
        print_section("Injecting All News")
        for c in candidates[:10]:  # Inject top 10
            engine.inject_event(c)
            print(f"  + {c[:60]}...")

        print_section("Recalculating Probabilities")
        print("(Analyzing how news changes collision dynamics...)")

        result = engine.recalculate()
        new_state = result["new_state"]
        deltas = result["deltas"]

        # Show revised estimates
        individual = engine.get_individual_estimates()
        if individual:
            print(f"\nINDIVIDUAL LLM REVISED ESTIMATES:")
            print("-" * 60)
            print(f"{'LLM':<10} {'P(+5%)':<10} {'P(+10%)':<10} {'P(-5%)':<10} {'P(-10%)':<10}")
            print("-" * 60)
            for provider, est in individual.items():
                if "error" in est:
                    print(f"{provider.upper():<10} {'[' + est['error'][:30] + ']':<40}")
                else:
                    print(f"{provider.upper():<10} {est['p_up_5']*100:>6.0f}%   {est['p_up_10']*100:>6.0f}%    {est['p_down_5']*100:>6.0f}%    {est['p_down_10']*100:>6.0f}%")
            print("-" * 60)

        print(f"\nCONSENSUS REVISED ESTIMATES:")
        print(f"  P(+5% or more):  {new_state.p_up_5*100:.0f}% ({deltas['p_up_5']:+.1f} pp)")
        print(f"  P(+10% or more): {new_state.p_up_10*100:.0f}% ({deltas['p_up_10']:+.1f} pp)")
        print(f"  P(-5% or more):  {new_state.p_down_5*100:.0f}% ({deltas['p_down_5']:+.1f} pp)")
        print(f"  P(-10% or more): {new_state.p_down_10*100:.0f}% ({deltas['p_down_10']:+.1f} pp)")

        # Net effect
        net_shift = deltas["p_up_10"] - deltas["p_down_10"]
        if net_shift > 5:
            print(f"\n>>> NET EFFECT: BULLISH (+{net_shift:.1f}% shift toward upside)")
        elif net_shift < -5:
            print(f"\n>>> NET EFFECT: BEARISH ({net_shift:.1f}% shift toward downside)")
        else:
            print(f"\n>>> NET EFFECT: NEUTRAL ({net_shift:+.1f}% shift)")

        print(f"\n{result['analysis']}")

    else:
        print("No recent news found.")
        print("Tip: Set NEWS_API_KEY in .env for more news sources.")

    # Save session
    session_path = engine.save_session()
    md_path = engine.save_markdown_report()
    print(f"\nSession saved: {session_path}")
    print(f"Full report: {md_path}")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()

    if command == "analyze" and len(sys.argv) >= 3:
        cmd_analyze(sys.argv[2].upper())
    elif command == "gate":
        cmd_gate()
    elif command == "roller" and len(sys.argv) >= 3:
        cmd_roller(*[t.upper() for t in sys.argv[2:]])
    elif command == "events":
        cmd_events()
    elif command == "portfolio":
        cmd_portfolio()
    elif command == "jump":
        cmd_jump()
    elif command == "collision" and len(sys.argv) >= 3:
        cmd_collision(sys.argv[2].upper(), *sys.argv[3:])
    elif command == "news" and len(sys.argv) >= 3:
        cmd_news(sys.argv[2].upper())
    elif command == "collision-news" and len(sys.argv) >= 3:
        cmd_collision_with_news(sys.argv[2].upper())
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
