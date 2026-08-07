import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
import pandas_ta_classic as ta
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from dotenv import load_dotenv
load_dotenv('keys.env')

# --- 1. API Configuration ---
API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
BASE_URL = 'https://paper-api.alpaca.markets'

if not API_KEY or not SECRET_KEY:
    raise RuntimeError(
        "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables. "
        "Set them before running this script."
    )

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# --- 2. Fund Capital Allocation Weights ---
# Total equals 0.85 (leaving 15% in cash buffer)
ALLOCATION_WEIGHTS = {
    'trend_following': 0.30,
    'pairs_trading': 0.25,
    'momentum_basket': 0.30
}


# --- 3. Dynamic Budget Calculator ---
def get_strategy_budget(strategy_name):
    """Fetches total account equity and calculates the dollar budget for a strategy."""
    account = api.get_account()
    total_equity = float(account.equity)
    weight = ALLOCATION_WEIGHTS.get(strategy_name, 0)
    budget = total_equity * weight
    print(f"[{strategy_name.upper()}] Allocated Budget: ${budget:,.2f} ({weight*100}%)")
    return budget


def get_current_qty(ticker):
    """Returns current held quantity for a ticker, or 0 if no position exists."""
    try:
        position = api.get_position(ticker)
        return int(position.qty)
    except Exception:
        return 0


# --- 4. Strategy Modules ---

def run_moving_average_strategy():
    strategy_id = 'trend_following'
    budget = get_strategy_budget(strategy_id)
    ma_ticker = 'AAPL'

    print(f"\n--- Running MA Strategy for {ma_ticker} ---")
    try:
        data = yf.download(ma_ticker, period='1y', progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        close = data['Close'].squeeze()
        data['Fast_MA'] = close.rolling(window=45).mean()
        data['Slow_MA'] = close.rolling(window=90).mean()
        data['RSI'] = ta.rsi(close, length=14)

        data['Raw_Signal'] = np.where((data['Fast_MA'] > data['Slow_MA']) & (data['RSI'] < 70), 1, 0)
        current_signal = data['Raw_Signal'].iloc[-1]
        print(f"MA Signal for {ma_ticker}: {current_signal} (1=Buy, 0=Sell)")

        current_qty = get_current_qty(ma_ticker)
        latest_price = close.iloc[-1]

        # Size the position from the strategy's dollar budget rather than a fixed qty
        target_qty = int(budget // latest_price) if current_signal == 1 else 0

        if current_signal == 1 and current_qty == 0 and target_qty > 0:
            api.submit_order(symbol=ma_ticker, qty=target_qty, side='buy', type='market', time_in_force='gtc')
            print(f"MA Bot: BUY order submitted for {target_qty} shares of {ma_ticker}.")
        elif current_signal == 0 and current_qty > 0:
            api.submit_order(symbol=ma_ticker, qty=current_qty, side='sell', type='market', time_in_force='gtc')
            print(f"MA Bot: SELL order submitted. Closed position for {ma_ticker}.")
        else:
            print("MA Bot: Position matches signal. Holding.")
    except Exception as e:
        print(f"Error in MA Strategy: {e}")
    print("Trend Following executed.\n")


def run_pairs_trading_strategy():
    strategy_id = 'pairs_trading'
    budget = get_strategy_budget(strategy_id)
    pairs_basket = ['AAPL', 'MSFT', 'GOOGL', 'META', 'PEP', 'KO', 'V', 'MA']

    print("\n--- Running Pairs Trading Strategy Scanner ---")
    try:
        data = yf.download(pairs_basket, period='1y', progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            prices_df = data['Close']
        else:
            prices_df = data[['Close']].droplevel(0, axis=1)
        prices_df = prices_df.dropna(axis=1)

        # Scan for cointegrated pairs
        n = prices_df.shape[1]
        keys = prices_df.columns
        valid_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                result = coint(prices_df[keys[i]], prices_df[keys[j]])
                if result[1] < 0.05:
                    valid_pairs.append((keys[i], keys[j], result[1]))

        if not valid_pairs:
            print("Pairs Bot: No cointegrated pairs found today.")
            return

        # Take the most cointegrated pair (lowest p-value)
        valid_pairs.sort(key=lambda x: x[2])
        stock1, stock2, pval = valid_pairs[0]
        print(f"Pairs Bot: Selected top pair -> {stock2} vs {stock1} (p-value: {pval:.4f})")

        # Calculate hedge ratio & z-score
        X = sm.add_constant(prices_df[stock1])
        model = sm.OLS(prices_df[stock2], X).fit()
        beta = model.params[stock1]

        spread = prices_df[stock2] - (beta * prices_df[stock1])
        z_score = (spread - spread.rolling(window=60).mean()) / spread.rolling(window=60).std()
        latest_z = z_score.iloc[-1]
        print(f"Pairs Bot: Latest Z-Score for pair: {latest_z:.4f}")

        price1 = prices_df[stock1].iloc[-1]
        price2 = prices_df[stock2].iloc[-1]

        # Split the budget between the two legs, weighted by the hedge ratio
        leg2_dollars = budget / (1 + beta)
        leg1_dollars = budget - leg2_dollars
        qty1 = int(leg1_dollars // price1)
        qty2 = int(leg2_dollars // price2)

        held1 = get_current_qty(stock1)
        held2 = get_current_qty(stock2)

        if latest_z > 1.5 and latest_z < 3.0:
            print(f"Pairs Bot: Z-score high. Shorting spread (Short {stock2}, Long {stock1}).")
            if held2 == 0 and held1 == 0 and qty1 > 0 and qty2 > 0:
                api.submit_order(symbol=stock1, qty=qty1, side='buy', type='market', time_in_force='gtc')
                api.submit_order(symbol=stock2, qty=qty2, side='sell', type='market', time_in_force='gtc')
                print(f"Pairs Bot: Entered spread short — BUY {qty1} {stock1}, SELL {qty2} {stock2}.")
            else:
                print("Pairs Bot: Position already open or size too small; no new entry.")

        elif latest_z < -1.5 and latest_z > -3.0:
            print(f"Pairs Bot: Z-score low. Longing spread (Long {stock2}, Short {stock1}).")
            if held2 == 0 and held1 == 0 and qty1 > 0 and qty2 > 0:
                api.submit_order(symbol=stock2, qty=qty2, side='buy', type='market', time_in_force='gtc')
                api.submit_order(symbol=stock1, qty=qty1, side='sell', type='market', time_in_force='gtc')
                print(f"Pairs Bot: Entered spread long — BUY {qty2} {stock2}, SELL {qty1} {stock1}.")
            else:
                print("Pairs Bot: Position already open or size too small; no new entry.")

        elif abs(latest_z) >= 3.0:
            print("Pairs Bot: Stop-loss threshold hit! Flattening pair positions.")
            if held1 != 0:
                side = 'sell' if held1 > 0 else 'buy'
                api.submit_order(symbol=stock1, qty=abs(held1), side=side, type='market', time_in_force='gtc')
            if held2 != 0:
                side = 'sell' if held2 > 0 else 'buy'
                api.submit_order(symbol=stock2, qty=abs(held2), side=side, type='market', time_in_force='gtc')
            print("Pairs Bot: Flattened both legs.")

        else:
            print("Pairs Bot: Z-score inside neutral band. No action required.")

    except Exception as e:
        print(f"Error in Pairs Strategy: {e}")
    print("Pairs Trading executed.\n")


def run_momentum_strategy():
    strategy_id = 'momentum_basket'
    budget = get_strategy_budget(strategy_id)
    basket = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN', 'TSLA', 'AMD', 'INTC', 'QCOM']

    lookback_days = 63   # ~3 trading months
    top_n = 3            # how many names to hold at once

    print("\n--- Running Cross Sectional Momentum Strategy ---")
    try:
        raw_data = yf.download(basket, period='1y', progress=False)

        if isinstance(raw_data.columns, pd.MultiIndex):
            prices_df = raw_data['Close']
        else:
            prices_df = raw_data[['Close']]

        prices_df = prices_df.dropna(axis=1)

        if len(prices_df) < lookback_days:
            print("Error: Insufficient historical data to compute lookback returns.")
            return None

        # Rank by lookback return
        momentum_returns = (prices_df.iloc[-1] - prices_df.iloc[-lookback_days]) / prices_df.iloc[-lookback_days]
        ranked_momentum = momentum_returns.sort_values(ascending=False)

        results = []
        target_tickers = []
        for rank, (ticker, ret) in enumerate(ranked_momentum.items(), start=1):
            signal = "TARGET (LONG)" if rank <= top_n else "IGNORE (LAGGARD)"
            if rank <= top_n:
                target_tickers.append(ticker)
            results.append({
                'Rank': rank,
                'Ticker': ticker,
                f'{lookback_days}D Return (%)': round(ret * 100, 2),
                'Action': signal
            })

        results_df = pd.DataFrame(results)
        print(f"\nLookback Window: {lookback_days} Trading Days")
        print(f"Targeting Top {top_n} Strongest Assets\n")
        print(results_df.to_string(index=False))

        # Execution: equal-dollar allocation across the top_n names
        per_stock_budget = budget / top_n if top_n > 0 else 0
        latest_prices = prices_df.iloc[-1]

        try:
            current_positions = {p.symbol: int(p.qty) for p in api.list_positions()}
        except Exception as e:
            print(f"Could not fetch positions: {e}")
            current_positions = {}

        # Exit names that fell out of the top_n
        for ticker in basket:
            if ticker not in target_tickers and current_positions.get(ticker, 0) > 0:
                qty = current_positions[ticker]
                api.submit_order(symbol=ticker, qty=qty, side='sell', type='market', time_in_force='gtc')
                print(f"Momentum Bot: SELL {qty} {ticker} (dropped out of top {top_n}).")

        # Buy / trim toward target size for the current top_n
        for ticker in target_tickers:
            price = latest_prices[ticker]
            target_qty = int(per_stock_budget // price)
            held_qty = current_positions.get(ticker, 0)
            delta = target_qty - held_qty

            if delta > 0:
                api.submit_order(symbol=ticker, qty=delta, side='buy', type='market', time_in_force='gtc')
                print(f"Momentum Bot: BUY {delta} {ticker} (target {target_qty}, held {held_qty}).")
            elif delta < 0:
                api.submit_order(symbol=ticker, qty=abs(delta), side='sell', type='market', time_in_force='gtc')
                print(f"Momentum Bot: TRIM {abs(delta)} {ticker} (target {target_qty}, held {held_qty}).")
            else:
                print(f"Momentum Bot: {ticker} already at target size ({held_qty}).")

        return results_df

    except Exception as e:
        print(f"Error in Momentum Strategy: {e}")
        return None
    finally:
        print("Momentum Basket executed.\n")


# --- 5. Master Pipeline ---
def master_trading_pipeline():
    print(f"\n==========================================")
    print(f"Multi-Strategy Fund Waking Up: {datetime.datetime.now()}")
    print(f"==========================================")

    run_moving_average_strategy()
    run_pairs_trading_strategy()
    run_momentum_strategy()

    print("Master Pipeline Complete. Going back to sleep.\n")


# --- 6. Scheduler Setup ---
if __name__ == "__main__":
    master_trading_pipeline()
