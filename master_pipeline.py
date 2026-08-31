import os
import datetime
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import pandas_ta_classic as ta
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from dotenv import load_dotenv

load_dotenv('keys.env')

API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
BASE_URL = 'https://paper-api.alpaca.markets'

if not API_KEY or not SECRET_KEY:
    raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY")

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
DATA_FEED = 'iex'

ALLOCATION_WEIGHTS = {
    'trend_following': 0.30,
    'pairs_trading': 0.25,
    'momentum_basket': 0.30
}

def get_strategy_budget(strategy_name):
    account = api.get_account()
    total_equity = float(account.equity)
    weight = ALLOCATION_WEIGHTS.get(strategy_name, 0)
    budget = total_equity * weight
    print(f"[{strategy_name.upper()}] Allocated Budget: ${budget:,.2f} ({weight*100}%)")
    return budget


def fetch_daily_bars(tickers, lookback_days=400):
    single = isinstance(tickers, str)
    symbols = [tickers] if single else list(tickers)

    end = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=16)
    start = end - datetime.timedelta(days=lookback_days)

    bars = api.get_bars(
        symbols,
        TimeFrame.Day,
        start.strftime('%Y-%m-%dT%H:%M:%SZ'),
        end.strftime('%Y-%m-%dT%H:%M:%SZ'),
        feed=DATA_FEED,
        adjustment='raw',
    ).df

    if bars.empty:
        raise RuntimeError(f"Alpaca returned no bar data for {symbols}")

    if single:
        return bars[['open', 'high', 'low', 'close', 'volume']]

    bars = bars.reset_index()
    if 'symbol' not in bars.columns:
        bars['symbol'] = symbols[0]

    close_wide = bars.pivot(index='timestamp', columns='symbol', values='close')
    close_wide = close_wide.dropna(axis=1, how='any')

    return close_wide


# --- 1. Trend Following Strategy ---
def run_moving_average_strategy():
    strategy_id = 'trend_following'
    budget = get_strategy_budget(strategy_id)
    
    etf_basket = ['SPY', 'QQQ', 'GLD', 'TLT']
    target_portfolio = {ticker: 0.0 for ticker in etf_basket}
    
    print(f"\n--- Running MA Strategy for Macro ETFs ---")
    try:
        data = fetch_daily_bars(etf_basket, lookback_days=400)
        per_etf_budget = budget / len(etf_basket)
        
        for ticker in etf_basket:
            if ticker not in data.columns: 
                continue
            close = data[ticker]
            
            fast_ma = close.rolling(window=45).mean()
            slow_ma = close.rolling(window=90).mean()
            rsi = ta.rsi(close, length=14)
            
            raw_signal = np.where((fast_ma > slow_ma) & (rsi < 70), 1, 0)
            current_signal = raw_signal[-1]
            print(f"MA Signal for {ticker}: {current_signal} (1=Buy, 0=Sell)")
            
            if current_signal == 1:
                latest_price = close.iloc[-1]
                target_portfolio[ticker] = round(per_etf_budget / latest_price, 4)

    except Exception as e:
        print(f"Error in MA Strategy: {e}")
        
    print(f"Trend Following Target: {target_portfolio}")
    return target_portfolio


# --- 2. Pairs Trading Strategy ---
def run_pairs_trading_strategy():
    strategy_id = 'pairs_trading'
    budget = get_strategy_budget(strategy_id)
    
    SECTORS = {
        'Semiconductors': ['NVDA', 'AMD', 'INTC', 'TSM', 'QCOM', 'AVGO', 'TXN'],
        'Big_Tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NFLX'],
        'Financials': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA']
    }
    
    all_tickers = [ticker for sector in SECTORS.values() for ticker in sector]
    target_portfolio = {ticker: 0.0 for ticker in all_tickers}
    
    print("\n--- Running Sector-Clustered Pairs Scanner ---")
    try:
        prices_df = fetch_daily_bars(all_tickers, lookback_days=400)
        
        valid_pairs = []
        for sector_name, tickers in SECTORS.items():
            valid_tickers = [t for t in tickers if t in prices_df.columns]
            n = len(valid_tickers)
            for i in range(n):
                for j in range(i + 1, n):
                    result = coint(prices_df[valid_tickers[i]], prices_df[valid_tickers[j]])
                    if result[1] < 0.05:
                        valid_pairs.append((valid_tickers[i], valid_tickers[j], result[1], sector_name))

        if not valid_pairs:
            print("Pairs Bot: No cointegrated pairs found today.")
            return target_portfolio

        valid_pairs.sort(key=lambda x: x[2])
        stock1, stock2, pval, sector = valid_pairs[0]
        print(f"Pairs Bot: Selected top pair in {sector} -> {stock2} vs {stock1} (p-value: {pval:.4f})")

        X = sm.add_constant(prices_df[stock1])
        model = sm.OLS(prices_df[stock2], X).fit()
        beta = model.params[stock1]

        spread = prices_df[stock2] - (beta * prices_df[stock1])
        z_score = (spread - spread.rolling(window=60).mean()) / spread.rolling(window=60).std()
        latest_z = z_score.iloc[-1]
        print(f"Pairs Bot: Latest Z-Score for pair: {latest_z:.4f}")

        price1 = prices_df[stock1].iloc[-1]
        price2 = prices_df[stock2].iloc[-1]

        qty2 = round(budget / (price2 + abs(beta) * price1), 4)
        qty1 = round(abs(beta) * qty2, 4)

        if 1.5 < latest_z < 3.0:
            print(f"Pairs Bot: Z-score high. Shorting spread (Short {stock2}, Long {stock1}).")
            target_portfolio[stock1] = qty1
            target_portfolio[stock2] = -qty2
        elif -3.0 < latest_z < -1.5:
            print(f"Pairs Bot: Z-score low. Longing spread (Long {stock2}, Short {stock1}).")
            target_portfolio[stock2] = qty2
            target_portfolio[stock1] = -qty1
        else:
            print("Pairs Bot: Z-score in neutral band or stop-loss. Target is 0.")

    except Exception as e:
        print(f"Error in Pairs Strategy: {e}")
        
    print(f"Pairs Trading Target: {target_portfolio}")
    return target_portfolio


# --- 3. Momentum Strategy ---
def run_momentum_strategy():
    strategy_id = 'momentum_basket'
    budget = get_strategy_budget(strategy_id)
    
    print("\n--- Running Dynamic Cross Sectional Momentum ---")
    try:
        print("Fetching Nasdaq-100 Universe...")
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100", match="Ticker")
        
        if len(tables) > 0:
            nasdaq_df = tables[0]
            basket = nasdaq_df['Ticker'].tolist()
            print(f"Successfully scraped {len(basket)} tickers.")
        else:
            print("Failed to scrape Wikipedia. Using fallback basket.")
            basket = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN', 'TSLA', 'AMD', 'INTC', 'QCOM', 'ADBE', 'NFLX', 'CSCO', 'PEP', 'AVGO']
            
        target_portfolio = {ticker: 0.0 for ticker in basket}
        lookback_days = 63
        top_n = 5

        prices_df = fetch_daily_bars(basket, lookback_days=400)

        if len(prices_df) < lookback_days:
            print("Error: Insufficient historical data.")
            return target_portfolio

        momentum_returns = (prices_df.iloc[-1] - prices_df.iloc[-lookback_days]) / prices_df.iloc[-lookback_days]
        ranked_momentum = momentum_returns.sort_values(ascending=False)
        
        target_tickers = []
        for rank, (ticker, ret) in enumerate(ranked_momentum.items(), start=1):
            if rank <= top_n:
                target_tickers.append(ticker)

        per_stock_budget = budget / top_n if top_n > 0 else 0
        latest_prices = prices_df.iloc[-1]

        for ticker in target_tickers:
            price = latest_prices[ticker]
            target_portfolio[ticker] = round(per_stock_budget / price, 4)

    except Exception as e:
        print(f"Error in Momentum Strategy: {e}")
        target_portfolio = {}
        
    print(f"Momentum Target: {target_portfolio}")
    return target_portfolio


def get_actual_positions():
    try:
        positions = api.list_positions()
        actuals = {}
        for p in positions:
            qty = float(p.qty)
            if getattr(p, 'side', '') == 'short':
                qty = -abs(qty)
            actuals[p.symbol] = qty
        return actuals
    except Exception as e:
        print(f"Could not fetch positions: {e}")
        return {}


def execute_target_portfolio(target_portfolio):
    print("\n--- Execution Engine ---")
    
    try:
        api.cancel_all_orders()
        print("Cancelled all pending open orders to clean the slate.")
    except Exception as e:
        print(f"Failed to cancel open orders: {e}")

    actual_positions = get_actual_positions()
    
    all_symbols = set(target_portfolio.keys()).union(set(actual_positions.keys()))
    
    for symbol in all_symbols:
        target_qty = round(target_portfolio.get(symbol, 0.0), 4)
        actual_qty = round(actual_positions.get(symbol, 0.0), 4)
        delta = round(target_qty - actual_qty, 4)
        
        if abs(delta) < 0.0001:
            continue
            
        print(f"Rebalancing {symbol}: Actual={actual_qty}, Target={target_qty}, Delta={delta}")
        
        try:
            side = 'buy' if delta > 0 else 'sell'
            api.submit_order(
                symbol=symbol,
                qty=abs(delta),
                side=side,
                type='market',
                time_in_force='day'
            )
            print(f"-> Submitted {side.upper()} order for {abs(delta)} shares of {symbol}")
        except Exception as e:
            print(f"-> Error placing order for {symbol}: {e}")


def master_trading_pipeline():
    print(f"\n==========================================")
    print(f"Multi-Strategy Fund Waking Up: {datetime.datetime.now()}")
    print(f"==========================================")

    ma_targets = run_moving_average_strategy()
    pairs_targets = run_pairs_trading_strategy()
    mom_targets = run_momentum_strategy()

    aggregated_targets = {}
    all_targets = [ma_targets, pairs_targets, mom_targets]
    
    for strategy_portfolio in all_targets:
        for ticker, qty in strategy_portfolio.items():
            aggregated_targets[ticker] = aggregated_targets.get(ticker, 0.0) + qty
            
    print("\n--- Final Aggregated Target Portfolio ---")
    for ticker, qty in aggregated_targets.items():
        if abs(qty) > 0.0001:
            print(f"{ticker}: {qty:.4f}")

    execute_target_portfolio(aggregated_targets)

    print("\nMaster Pipeline Complete. Going back to sleep.\n")


if __name__ == "__main__":
    master_trading_pipeline()
