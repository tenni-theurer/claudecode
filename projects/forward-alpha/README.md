# Forward Alpha v2

Autonomous equity analysis system using **Multi-Events Collision Theory** with multi-LLM consensus (Claude, GPT-4, Gemini, Grok).

## What is Multi-Events Collision Theory?

Imagine a stock as a ball moving through space. Other balls (events, news, macro factors) are moving nearby and may **collide** with it, changing its trajectory.

Each "ball" has physics properties:

| Property | Description | Range |
|----------|-------------|-------|
| **Velocity** | Bullish/bearish momentum | -1 (bearish) to +1 (bullish) |
| **Mass** | Importance/impact weight | 0 to 1 |
| **Distance** | How soon it impacts | 0 (imminent) to 1 (distant) |

### Example: NVDA Analysis

```
Initial State: NVDA at $191, 83rd percentile, earnings in 26 days

Collision Factors Identified:
1. AI Chip Demand      → Velocity: +0.9, Mass: 0.9, Distance: 0.3
2. Earnings Report     → Velocity: +0.6, Mass: 0.9, Distance: 0.2
3. China Export Risk   → Velocity: -0.7, Mass: 0.8, Distance: 0.4
4. Blackwell Ramp      → Velocity: +0.7, Mass: 0.8, Distance: 0.5

Initial Probability: P(+5%) = 50%, P(-5%) = 30%
```

Then you **inject new data** and watch probabilities shift:

```
Injected: "China may approve only 33%-50% of H200 sales"

Result: P(+5%) drops 50% → 35%, P(-5%) rises 30% → 45%
Net Effect: BEARISH (-15% shift)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FORWARD ALPHA v2                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ DATA COLLECT │───▶│  COLLISION   │───▶│    REPORT    │  │
│  │              │    │   ENGINE     │    │  GENERATOR   │  │
│  │ • yfinance   │    │              │    │              │  │
│  │ • earnings   │    │ • Model ball │    │ • JSON       │  │
│  │ • 52w range  │    │ • Inject     │    │ • Markdown   │  │
│  │ • momentum   │    │ • Recalc     │    │ • Per-LLM    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                             │                               │
│                             ▼                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    MULTI-LLM LAYER                    │  │
│  │                                                       │  │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐         │  │
│  │   │ CLAUDE  │    │  GPT-4  │    │ GEMINI  │         │  │
│  │   └────┬────┘    └────┬────┘    └────┬────┘         │  │
│  │        │              │              │               │  │
│  │        └──────────────┼──────────────┘               │  │
│  │                       ▼                              │  │
│  │              ┌─────────────────┐                     │  │
│  │              │    CONSENSUS    │                     │  │
│  │              │   (averaging)   │                     │  │
│  │              └─────────────────┘                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Commands

### Collision Analysis (Primary Feature)

```bash
# Basic analysis - get initial probabilities
python main.py collision NVDA

# Inject events and see probability shifts
python main.py collision NVDA "China approves only 33% of H200 sales"

# Multiple events
python main.py collision NVDA \
  "China approves only 33% of H200 sales" \
  "TSMC Arizona now producing Blackwell chips"
```

### Other Commands

```bash
# Full company analysis
python main.py analyze AAPL

# Roller coaster scan - find stocks at peaks/bottoms
python main.py roller NVDA AAPL MSFT GOOGL

# Index gate - check if market conditions favor buying
python main.py gate
```

## Output

### Console Output

```
INDIVIDUAL LLM ESTIMATES (by earnings date):
------------------------------------------------------------
LLM        P(+5%)     P(+10%)    P(-5%)     P(-10%)
------------------------------------------------------------
GEMINI         45%       20%        25%        10%
GPT            65%       45%        30%        15%
CLAUDE         35%       18%        42%        23%
------------------------------------------------------------

CONSENSUS ESTIMATE: P(+5%) = 48%, P(-5%) = 32%
```

### Reports Generated

- `output/reports/NVDA_collision_YYYYMMDD_HHMMSS.json` - Raw data
- `output/reports/NVDA_collision_YYYYMMDD_HHMMSS.md` - Full analysis with reasoning

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...    # Claude
OPENAI_API_KEY=sk-proj-...      # GPT-4
GOOGLE_API_KEY=AIzaSy...        # Gemini
XAI_API_KEY=xai-...             # Grok
```

You need at least one LLM key. For best results, use all four for consensus.

### 3. Run

```bash
python main.py collision NVDA
```

## File Structure

```
forward-alpha/
├── main.py                 # CLI entry point
├── config.py               # API keys, settings
├── requirements.txt        # Dependencies
│
├── core/
│   ├── data_collector.py   # Stock data via yfinance
│   └── events.py           # Event tracking
│
├── llm/
│   ├── multi_llm.py        # Claude, GPT-4, Gemini clients
│   └── consensus.py        # Consensus building
│
├── strategy/
│   ├── collision.py        # Multi-Events Collision Engine
│   ├── roller_coaster.py   # Peak/bottom detection
│   ├── index_gate.py       # Market condition gate
│   └── position.py         # Position management
│
└── output/
    └── reports/            # Generated reports (gitignored)
```

## Key Concepts

### Why Multiple LLMs?

Each LLM has different "personalities":

| LLM | Tendency | Style |
|-----|----------|-------|
| **Gemini** | Most reactive to news | Quantitative, uses specific velocity/mass values |
| **GPT-4** | Most bullish baseline | Narrative, balanced |
| **Claude** | Most conservative | Emphasizes downside risk, detailed collision math |
| **Grok** | Contrarian, direct | Unfiltered analysis, challenges consensus |

Consensus averaging reduces individual model bias.

### Probability Estimates

The system estimates four probabilities by the earnings date (or 3-month horizon):

- **P(+5%)** - Probability stock rises 5% or more
- **P(+10%)** - Probability stock rises 10% or more
- **P(-5%)** - Probability stock falls 5% or more
- **P(-10%)** - Probability stock falls 10% or more

### Event Injection

The power of the system is **dynamic recalculation**. When you inject new data:

1. LLMs reassess which "balls" moved closer or changed velocity
2. New probabilities are calculated
3. Deltas show exactly how much each event shifted the outlook

This lets you quickly test "what if" scenarios.

## Limitations

- LLM estimates are probabilistic, not predictions
- No backtesting or historical validation yet
- Relies on LLM interpretation of collision physics metaphor
- API costs scale with usage (~$0.10-0.50 per full analysis)

## License

Personal use. Not financial advice.
