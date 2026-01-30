"""
Data Collector - Gathers data across 11 categories for analysis
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional
import requests

from config import DATA_CATEGORIES


def collect_stock_data(ticker: str) -> dict:
    """
    Collect comprehensive data for a stock across all 11 categories

    Categories:
    1. current_products - Current products sales forecast
    2. new_products - New upcoming products sales forecast
    3. customers - Types of customers and forecasts
    4. competitors - Competitive landscape
    5. key_people - Key people and their outlook
    6. macro_events - Macroeconomic events
    7. geopolitical - Geopolitical events
    8. technology - Underlying technologies
    9. future_products - Future products forecast
    10. associated_fields - Associated fields forecast
    11. market_players - Players at the gambling table
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    # Get price history for context
    hist = stock.history(period="1y")

    data = {
        "ticker": ticker,
        "name": info.get("longName", ticker),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "collected_at": datetime.now().isoformat(),

        # Current financials for context
        "financials": {
            "price": info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "revenue_ttm": info.get("totalRevenue"),
            "revenue_growth": info.get("revenueGrowth"),
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "pe_forward": info.get("forwardPE"),
            "price_52w_high": info.get("fiftyTwoWeekHigh"),
            "price_52w_low": info.get("fiftyTwoWeekLow"),
        },

        # Price position (for roller coaster)
        "price_position": _calculate_price_position(info, hist),

        # Earnings info
        "earnings": _get_earnings_info(stock),

        # Category-specific data (to be filled by LLMs)
        "categories": {cat: None for cat in DATA_CATEGORIES},
    }

    return data


def _calculate_price_position(info: dict, hist) -> dict:
    """Calculate where price is relative to 52-week range"""
    high = info.get("fiftyTwoWeekHigh", 0)
    low = info.get("fiftyTwoWeekLow", 0)
    current = info.get("currentPrice", 0)

    if high and low and high != low:
        position = (current - low) / (high - low)
    else:
        position = 0.5

    # Recent momentum (30-day)
    if len(hist) >= 30:
        price_30d_ago = hist["Close"].iloc[-30]
        momentum_30d = (current - price_30d_ago) / price_30d_ago if price_30d_ago else 0
    else:
        momentum_30d = 0

    return {
        "percentile_52w": round(position * 100, 1),
        "near_high": position > 0.85,
        "near_low": position < 0.15,
        "momentum_30d": round(momentum_30d * 100, 2),
        "status": "peak" if position > 0.85 else "bottom" if position < 0.15 else "middle",
    }


def _get_earnings_info(stock) -> dict:
    """Get earnings calendar info"""
    try:
        calendar = stock.calendar
        earnings_date = None

        if calendar is not None:
            # Handle dict format (new yfinance)
            if isinstance(calendar, dict):
                earnings_dates = calendar.get("Earnings Date", [])
                if earnings_dates and len(earnings_dates) > 0:
                    earnings_date = earnings_dates[0]
            # Handle DataFrame format (old yfinance)
            elif hasattr(calendar, 'empty') and not calendar.empty:
                earnings_date = calendar.iloc[0].get("Earnings Date") if len(calendar) > 0 else None
    except Exception as e:
        print(f"Warning: Could not get earnings calendar: {e}")
        earnings_date = None

    # Calculate days to earnings
    days_to_earnings = None
    if earnings_date:
        try:
            from datetime import date
            if isinstance(earnings_date, date):
                days_to_earnings = (earnings_date - datetime.now().date()).days
            elif isinstance(earnings_date, datetime):
                days_to_earnings = (earnings_date - datetime.now()).days
            elif isinstance(earnings_date, str):
                ed = datetime.fromisoformat(earnings_date)
                days_to_earnings = (ed - datetime.now()).days
        except Exception as e:
            print(f"Warning: Could not calculate days to earnings: {e}")
            pass

    return {
        "next_earnings_date": str(earnings_date) if earnings_date else None,
        "days_to_earnings": days_to_earnings,
        "in_earnings_window": days_to_earnings is not None and 0 < days_to_earnings <= 14,
    }


def get_index_data(ticker: str = "SPY") -> dict:
    """Get S&P 500 index data for the gate check"""
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="1y")

    # Calculate YTD return
    if len(hist) > 0:
        start_price = hist["Close"].iloc[0]
        current_price = hist["Close"].iloc[-1]
        ytd_return = (current_price - start_price) / start_price
    else:
        ytd_return = 0

    return {
        "ticker": ticker,
        "price": info.get("currentPrice"),
        "ytd_return": round(ytd_return * 100, 2),
        "pe": info.get("trailingPE"),
        "price_position": _calculate_price_position(info, hist),
    }


def build_analysis_prompt(ticker: str, data: dict, category: str) -> str:
    """
    Build a prompt for LLM to analyze a specific category

    Returns a structured prompt for the given category
    """
    prompts = {
        "current_products": f"""Analyze {data['name']} ({ticker})'s current product portfolio:
1. List their main products/services and estimated revenue contribution
2. Sales trends for each major product line
3. Market share in each segment
4. Growth or decline trajectory for each product
5. Which products are cash cows vs growth drivers?""",

        "new_products": f"""Analyze {data['name']} ({ticker})'s upcoming products:
1. What new products are expected in the next 12-24 months?
2. What is the evidence for each (announcements, patents, job postings)?
3. Estimated launch timeline for each
4. Potential revenue impact of each new product
5. Probability of successful launch?""",

        "customers": f"""Analyze {data['name']} ({ticker})'s customer base:
1. Who are the main customer segments? (enterprise, consumer, government, etc.)
2. Customer concentration - any major customers >10% of revenue?
3. Customer retention/churn trends
4. New customer acquisition trends
5. Which customer segments are growing/declining?""",

        "competitors": f"""Analyze {data['name']} ({ticker})'s competitive landscape:
1. Who are the top 5 competitors?
2. Market share comparison
3. Competitive advantages and moats
4. Competitive threats and vulnerabilities
5. Who is gaining/losing share?""",

        "key_people": f"""Analyze key people at {data['name']} ({ticker}):
1. CEO background and track record
2. Recent executive changes
3. Board composition and activism risk
4. Key technical leaders
5. Any concerning departures or additions?""",

        "macro_events": f"""Analyze macroeconomic factors affecting {data['name']} ({ticker}):
1. Interest rate sensitivity
2. Inflation impact on costs and pricing
3. Currency exposure
4. Economic cycle sensitivity
5. Current macro headwinds and tailwinds""",

        "geopolitical": f"""Analyze geopolitical factors affecting {data['name']} ({ticker}):
1. Geographic revenue exposure
2. China/Taiwan risk (supply chain, revenue)
3. Trade policy and tariff exposure
4. Regulatory risks by region
5. Current geopolitical headwinds and tailwinds""",

        "technology": f"""Analyze underlying technology trends for {data['name']} ({ticker}):
1. Core technologies the company depends on
2. Technology adoption curves relevant to their business
3. Disruption risks from new technologies
4. R&D investment and innovation pipeline
5. Technology moats and vulnerabilities""",

        "future_products": f"""Forecast future products for {data['name']} ({ticker}) (2-5 year horizon):
1. What products could they build based on their capabilities?
2. What market gaps could they address?
3. What acquisitions might they make?
4. What new markets might they enter?
5. Probability and timeline for each""",

        "associated_fields": f"""Analyze fields associated with {data['name']} ({ticker}):
1. What adjacent markets are relevant?
2. How are those markets trending?
3. Potential expansion opportunities
4. Risks from adjacent market disruption
5. Synergies with current business""",

        "market_players": f"""Identify key players affecting {data['name']} ({ticker}) stock:
1. Major institutional holders and their recent moves
2. Activist investors to watch
3. Short interest and short seller thesis
4. Analyst consensus and notable bulls/bears
5. Retail sentiment indicators""",
    }

    market_cap = data['financials'].get('market_cap') or 0
    revenue = data['financials'].get('revenue_ttm') or 0
    growth = data['financials'].get('revenue_growth') or 0

    base_context = f"""Company: {data['name']} ({ticker})
Sector: {data.get('sector') or 'N/A'} | Industry: {data.get('industry') or 'N/A'}
Market Cap: ${market_cap/1e9:.1f}B
Revenue: ${revenue/1e9:.1f}B
Revenue Growth: {growth*100:.1f}%
Forward P/E: {data['financials'].get('pe_forward', 'N/A')}

"""

    return base_context + prompts.get(category, f"Analyze {category} for {ticker}")
