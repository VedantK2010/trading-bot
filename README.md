# Multi-Strategy Paper Trading Bot

A multi-strategy quant trading system that allocates capital across three
independent strategies and executes on Alpaca's **paper trading** API.
Runs automatically once per trading day via GitHub Actions — no server
needed.

> ⚠️ This trades on Alpaca's paper (simulated) API, not real money.

## Strategies

| Strategy | Idea | Capital Weight |
|---|---|---|
| Trend Following | 45/90-day MA crossover + RSI filter on AAPL | 30% |
| Pairs Trading | Cointegration scan across an 8-stock basket, mean-reversion on the spread z-score | 25% |
| Cross-Sectional Momentum | Ranks a 10-stock basket by 63-day return, holds the top 3 equal-weight | 30% |

15% of equity is held back as a cash buffer.

## Backtest results

*(fill in with your actual numbers — annualized return, Sharpe ratio, max
drawdown, win rate, and the date range tested. Numbers here, not adjectives.)*

## Architecture

- `master_pipeline.py` — entrypoint; pulls account equity, allocates budget
  per strategy, runs each strategy, submits orders via Alpaca.
- `.github/workflows/trade.yml` — GitHub Actions cron job that runs the
  pipeline every weekday at market close, in place of an always-on server.
- Secrets (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`) are stored as encrypted
  GitHub Actions secrets, never committed to the repo.

## Known limitations (being upfront about these)

- The pairs-trading cointegration scan tests 28 pairs at p < 0.05 with no
  multiple-comparisons correction (e.g. Benjamini-Hochberg), so a handful of
  false positives are expected by chance.
- Market data comes from `yfinance`, which can lag or rate-limit — fine for
  backtesting, a known weak point for live execution.
- No per-position stop-loss beyond the pairs strategy's z-score exit.

## Running locally

```bash
pip install -r requirements.txt
cp keys.env.example keys.env   # fill in your Alpaca paper keys
python master_pipeline.py
```
