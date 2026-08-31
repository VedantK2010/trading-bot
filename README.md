# Multi-Strategy Paper Trading Bot

A multi-strategy quant trading system that allocates capital across three independent strategies and executes on Alpaca's **paper trading** API. Runs automatically once per trading day via GitHub Actions — no server needed.

> ⚠️ This trades on Alpaca's paper (simulated) API, not real money.

## Strategies

| Strategy | Idea | Capital Weight |
|---|---|---|
| Trend Following | 45/90-day MA crossover + RSI filter on a diversified Macro ETF basket (SPY, QQQ, GLD, TLT) | 30% |
| Pairs Trading | Sector-clustered cointegration scan (Semiconductors, Big Tech, Financials), mean-reversion on the spread z-score | 25% |
| Cross-Sectional Momentum | Dynamically scrapes the Nasdaq-100 universe, ranks by 63-day return, holds the top 5 fractionally | 30% |

15% of equity is held back as a cash buffer.

## Backtest results

*(fill in with your actual numbers — annualized return, Sharpe ratio, max drawdown, win rate, and the date range tested. Numbers here, not adjectives.)*

## Architecture

- **Stateless Portfolio Engine**: Calculates a target net-exposure dictionary for the entire portfolio, compares it to actual holdings, and automatically nets out overlaps to submit precise `BUY/SELL` delta orders.
- **Fractional Shares**: Every dollar of budget is efficiently deployed using fractional share execution precision.
- `master_pipeline.py` — Entrypoint; pulls account equity, allocates budget, evaluates all 3 strategies, and executes the delta via Alpaca using `DAY` orders.
- `.github/workflows/trade.yml` — GitHub Actions cron job that runs the pipeline every weekday at 10:00 AM EDT.
- Market data is pulled directly from Alpaca's API (IEX data feed).
- Secrets (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`) are stored as encrypted GitHub Actions secrets, never committed to the repo.

## Known limitations (being upfront about these)

- The pairs-trading cointegration scan tests pairs at p < 0.05. While "sector-clustering" (grouping by industry) heavily reduces spurious correlation, false positives are still possible without strict multiple-comparisons correction (e.g., Benjamini-Hochberg).
- No per-position stop-loss beyond the pairs strategy's z-score exit and the moving average signal flips.
- Relying on Wikipedia HTML for the Nasdaq-100 constituent list could break if the Wikipedia page structure changes heavily, though a robust fallback list is implemented.

## Running locally

```bash
pip install -r requirements.txt
# Create a keys.env file and fill in your Alpaca paper keys
python master_pipeline.py
```
