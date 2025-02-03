import numpy as np
import pandas as pd


def load_data(file_path, interval="1min"):
    """
    Load and return the CSV data with resampling
    Args:
        file_path: str, path to CSV file
        interval: str, resampling interval ('1min', '1H', '1D', etc.)
    """
    # Load original data
    df = pd.read_csv(file_path)
    df.index = pd.to_datetime(df["time_utc"])  # Use time_utc column as index

    # If interval is not '1min', resample the data
    if interval != "1min":
        resampled_dfs = []
        for ticker in ["KRW-BTC", "KRW-ETH", "KRW-XRP"]:
            # Create ticker specific dataframe
            ticker_df = pd.DataFrame()

            # Map original columns to OHLCV format
            ticker_df["open"] = df[f"{ticker}_open"]
            ticker_df["high"] = df[f"{ticker}_high"]
            ticker_df["low"] = df[f"{ticker}_low"]
            ticker_df["close"] = df[f"{ticker}_close"]
            ticker_df["volume"] = df[f"{ticker}_volume_krw"]
            ticker_df.index = df.index  # Set the datetime index

            # Resample using the provided logic
            resampled = ticker_df.resample(interval, origin="start").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )

            # Rename columns back to original format
            resampled.columns = [
                f"{ticker}_open",
                f"{ticker}_high",
                f"{ticker}_low",
                f"{ticker}_close",
                f"{ticker}_volume_krw",
            ]

            resampled_dfs.append(resampled)

        # Combine all resampled dataframes
        df = pd.concat(resampled_dfs, axis=1)
        df = df.dropna()  # Remove any NaN rows after resampling

        # Save resampled data
        resampled_file = file_path.replace(".csv", f"_{interval}.csv")
        df.to_csv(resampled_file)

    return df


def find_short_term_points(df):
    """Find short-term highs and lows (comparing with adjacent candles)"""
    highs = df["high"].values
    lows = df["low"].values

    short_term_highs = np.zeros(len(df))
    short_term_lows = np.zeros(len(df))

    for i in range(1, len(df) - 1):
        # Short term high
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            short_term_highs[i] = highs[i]
        # Short term low
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            short_term_lows[i] = lows[i]

    return pd.Series(short_term_highs, index=df.index), pd.Series(
        short_term_lows, index=df.index
    )


def find_medium_term_points(short_term_highs, short_term_lows):
    """Find medium-term highs and lows based on short-term points"""
    medium_term_highs = pd.Series(
        np.zeros(len(short_term_highs)), index=short_term_highs.index
    )
    medium_term_lows = pd.Series(
        np.zeros(len(short_term_lows)), index=short_term_lows.index
    )

    # Filter out zeros for easier comparison
    st_highs = short_term_highs[short_term_highs != 0]
    st_lows = short_term_lows[short_term_lows != 0]

    for i in range(1, len(st_highs) - 1):
        idx = st_highs.index[i]
        if (
            st_highs.iloc[i] > st_highs.iloc[i - 1]
            and st_highs.iloc[i] > st_highs.iloc[i + 1]
        ):
            medium_term_highs[idx] = st_highs.iloc[i]

    for i in range(1, len(st_lows) - 1):
        idx = st_lows.index[i]
        if (
            st_lows.iloc[i] < st_lows.iloc[i - 1]
            and st_lows.iloc[i] < st_lows.iloc[i + 1]
        ):
            medium_term_lows[idx] = st_lows.iloc[i]

    return medium_term_highs, medium_term_lows


def generate_signals(
    df, short_term_highs, short_term_lows, medium_term_highs, medium_term_lows
):
    """Generate buy/sell signals based on Larry Williams' strategy"""
    signals = pd.Series(np.zeros(len(df)), index=df.index)
    position = 0  # Track current position to avoid duplicate signals

    # Filter out zeros
    mt_highs = medium_term_highs[medium_term_highs != 0]
    mt_lows = medium_term_lows[medium_term_lows != 0]

    for i in range(2, len(mt_lows)):
        # Only generate buy signal if not already long
        if position <= 0 and mt_lows.iloc[i] > mt_lows.iloc[i - 1]:
            # Find the next day after medium-term low confirmation
            idx = mt_lows.index[i]
            next_day_idx = df.index[df.index.get_loc(idx) + 1]

            # Check if price breaks above the high of medium-term low
            if df.loc[next_day_idx, "high"] > df.loc[idx, "high"]:
                signals[next_day_idx] = 1
                position = 1

    for i in range(2, len(mt_highs)):
        # Only generate sell signal if not already short
        if position >= 0 and mt_highs.iloc[i] < mt_highs.iloc[i - 1]:
            # Find the next day after medium-term high confirmation
            idx = mt_highs.index[i]
            next_day_idx = df.index[df.index.get_loc(idx) + 1]

            # Check if price breaks below the low of medium-term high
            if df.loc[next_day_idx, "low"] < df.loc[idx, "low"]:
                signals[next_day_idx] = -1
                position = -1

    return signals


def calculate_target_price(df, mt_highs, mt_lows, signal_idx, signals):
    """Calculate target price based on the previous medium-term swing"""
    if signals[signal_idx] == 0:  # No signal
        return 0

    # Find the previous medium-term points
    prev_mt_highs = mt_highs[mt_highs != 0]
    prev_mt_lows = mt_lows[mt_lows != 0]

    if signals[signal_idx] == 1:  # Buy signal
        signal_loc = prev_mt_lows.index.get_loc(signal_idx)
        if signal_loc < 2:  # Need at least 2 previous points
            return 0

        # Get the previous medium-term high and low
        prev_high = prev_mt_highs.iloc[signal_loc - 1]
        prev_low = prev_mt_lows.iloc[signal_loc - 1]

        # Calculate the target based on the previous swing's height
        swing_height = prev_high - prev_low
        current_low = prev_mt_lows.iloc[signal_loc]

        return current_low + swing_height

    else:  # Sell signal
        signal_loc = prev_mt_highs.index.get_loc(signal_idx)
        if signal_loc < 2:
            return 0

        # Get the previous medium-term high and low
        prev_high = prev_mt_highs.iloc[signal_loc - 1]
        prev_low = prev_mt_lows.iloc[signal_loc - 1]

        # Calculate the target based on the previous swing's height
        swing_height = prev_high - prev_low
        current_high = prev_mt_highs.iloc[signal_loc]

        return current_high - swing_height


def calculate_metrics(results_df, df):
    """Calculate detailed performance metrics"""
    if len(results_df) == 0:
        return {}

    # Calculate basic metrics
    total_trades = len(results_df[results_df["action"] == "ENTRY"])
    winning_trades = len(
        results_df[
            (results_df["action"].isin(["TARGET_HIT", "STOP_LOSS"]))
            & (results_df["pnl"] > 0)
        ]
    )

    # Calculate profit metrics
    gross_profits = results_df[results_df["pnl"] > 0]["pnl"].sum()
    gross_losses = abs(results_df[results_df["pnl"] < 0]["pnl"].sum())

    # Calculate equity curve
    equity_curve = []
    current_equity = 100000  # Starting with 100,000
    max_equity = current_equity
    max_drawdown = 0

    for _, row in results_df[results_df["pnl"].notna()].iterrows():
        pnl = row["pnl"]
        # Apply trading fee (0.05% per trade)
        fee = abs(row["price"]) * 0.0005
        pnl -= fee

        current_equity += pnl
        max_equity = max(max_equity, current_equity)
        drawdown = (max_equity - current_equity) / max_equity * 100
        max_drawdown = max(max_drawdown, drawdown)
        equity_curve.append(current_equity)

    # Calculate returns
    total_return = (current_equity - 100000) / 100000 * 100
    trading_days = (results_df["date"].max() - results_df["date"].min()).days

    # Calculate compound annual growth rate (CAGR) instead of simple annual return
    years = trading_days / 365
    if years > 0:
        cagr = (((current_equity / 100000) ** (1 / years)) - 1) * 100
    else:
        cagr = 0

    # Calculate risk metrics
    if gross_losses != 0:
        profit_factor = gross_profits / gross_losses
    else:
        profit_factor = float("inf")

    sharpe_ratio = 0
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.sqrt(252) * (np.mean(returns) / np.std(returns))

    return {
        "Total Trades": total_trades,
        "Win Rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
        "Profit Factor": profit_factor,
        "Maximum Drawdown (%)": max_drawdown,
        "Total Return (%)": total_return,
        "CAGR (%)": cagr,
        "Sharpe Ratio": sharpe_ratio,
        "Average PnL": results_df[results_df["pnl"].notna()]["pnl"].mean(),
        "Final Equity": current_equity,
    }


def backtest_strategy(
    df, signals, medium_term_highs, medium_term_lows, short_term_highs, short_term_lows
):
    """Backtest the strategy with improved position management"""
    position = 0
    entry_price = 0
    target_price = 0
    results = []

    for i in range(1, len(df)):
        current_idx = df.index[i]
        prev_idx = df.index[i - 1]

        # Check for new signals
        if signals[current_idx] != 0 and position == 0:
            position = signals[current_idx]
            # Add slippage (0.1%)
            slippage = df.loc[current_idx, "close"] * 0.001
            entry_price = df.loc[current_idx, "close"] + (slippage * position)

            if position == 1:  # If buy signal
                target_price = calculate_target_price(
                    df, medium_term_highs, medium_term_lows, current_idx, signals
                )
                stop_loss = df.loc[current_idx, "low"]
            else:  # If sell signal
                target_price = entry_price - (entry_price * 0.1)
                stop_loss = df.loc[current_idx, "high"]

            results.append(
                {
                    "date": current_idx,
                    "action": "ENTRY",
                    "position": position,
                    "price": entry_price,
                    "target": target_price,
                    "stop": stop_loss,
                }
            )

        # Position management
        elif position != 0:
            if position == 1:  # Long position management
                # Check if current low breaks the previous short-term low
                recent_st_lows = short_term_lows[short_term_lows != 0]
                if len(recent_st_lows) > 0:
                    last_st_low = recent_st_lows.iloc[-1]
                    if df.loc[current_idx, "low"] < last_st_low:
                        results.append(
                            {
                                "date": current_idx,
                                "action": "ST_LOW_BREAK",
                                "position": 0,
                                "price": df.loc[current_idx, "close"],
                                "pnl": df.loc[current_idx, "close"] - entry_price,
                            }
                        )
                        position = 0
                        continue

                # Rest of the long position management...
                if df.loc[current_idx, "high"] >= target_price:
                    results.append(
                        {
                            "date": current_idx,
                            "action": "TARGET_HIT",
                            "position": 0,
                            "price": target_price,
                            "pnl": target_price - entry_price,
                        }
                    )
                    position = 0
                elif df.loc[current_idx, "low"] < stop_loss:
                    results.append(
                        {
                            "date": current_idx,
                            "action": "STOP_LOSS",
                            "position": 0,
                            "price": stop_loss,
                            "pnl": stop_loss - entry_price,
                        }
                    )
                    position = 0
                else:
                    stop_loss = min(df.loc[prev_idx, "low"], stop_loss)

            else:  # Short position management
                # Check if current high breaks the previous short-term high
                recent_st_highs = short_term_highs[short_term_highs != 0]
                if len(recent_st_highs) > 0:
                    last_st_high = recent_st_highs.iloc[-1]
                    if df.loc[current_idx, "high"] > last_st_high:
                        results.append(
                            {
                                "date": current_idx,
                                "action": "ST_HIGH_BREAK",
                                "position": 0,
                                "price": df.loc[current_idx, "close"],
                                "pnl": entry_price - df.loc[current_idx, "close"],
                            }
                        )
                        position = 0
                        continue

                # Rest of the short position management
                if df.loc[current_idx, "low"] <= target_price:
                    results.append(
                        {
                            "date": current_idx,
                            "action": "TARGET_HIT",
                            "position": 0,
                            "price": target_price,
                            "pnl": entry_price - target_price,
                        }
                    )
                    position = 0
                elif df.loc[current_idx, "high"] > stop_loss:
                    results.append(
                        {
                            "date": current_idx,
                            "action": "STOP_LOSS",
                            "position": 0,
                            "price": stop_loss,
                            "pnl": entry_price - stop_loss,
                        }
                    )
                    position = 0
                else:
                    stop_loss = max(df.loc[prev_idx, "high"], stop_loss)

    return pd.DataFrame(results)


def get_investing_simulation(df, tickers, start_cash):
    """Get investing simulation for multiple coins"""
    # Initialize with explicit float64 dtype
    df["cash"] = pd.Series(start_cash, index=df.index, dtype="float64")
    df["coins_value"] = pd.Series(0.0, index=df.index, dtype="float64")
    df["balance"] = pd.Series(start_cash, index=df.index, dtype="float64")
    df["cash_available_portion"] = pd.Series(
        len(tickers), index=df.index, dtype="float64"
    )

    for ticker in tickers:
        df[f"{ticker}_amount"] = pd.Series(0.0, index=df.index, dtype="float64")

    for idx in range(1, len(df)):
        cash_temp = df.iloc[idx - 1]["cash"]

        for ticker in tickers:
            signal = df.iloc[idx][f"{ticker}_signal"]
            prev_amount = df.iloc[idx - 1][f"{ticker}_amount"]

            if signal == 1 and prev_amount == 0:  # Buy signal and no position
                buying_amount = cash_temp * (1 / df.iloc[idx]["cash_available_portion"])
                df.loc[df.index[idx], f"{ticker}_amount"] = (
                    buying_amount / df.iloc[idx][f"{ticker}_close"]
                )
                cash_temp -= buying_amount
                df.loc[df.index[idx], "cash_available_portion"] = float(
                    df.iloc[idx]["cash_available_portion"] - 1
                )

            elif signal == -1 and prev_amount > 0:  # Sell signal and has position
                selling_amount = prev_amount
                cash_temp += selling_amount * df.iloc[idx][f"{ticker}_close"]
                df.loc[df.index[idx], f"{ticker}_amount"] = 0.0
                df.loc[df.index[idx], "cash_available_portion"] = float(
                    df.iloc[idx]["cash_available_portion"] + 1
                )

            else:  # Hold position
                df.loc[df.index[idx], f"{ticker}_amount"] = float(prev_amount)

        df.loc[df.index[idx], "cash"] = float(cash_temp)
        coins_value = sum(
            df.iloc[idx][f"{ticker}_amount"] * df.iloc[idx][f"{ticker}_close"]
            for ticker in tickers
        )
        df.loc[df.index[idx], "coins_value"] = float(coins_value)
        df.loc[df.index[idx], "balance"] = float(cash_temp + coins_value)

    return df


def main():
    # Test different intervals
    intervals = ["1h", "4h", "1d"]

    for interval in intervals:
        print(f"\nTesting {interval} interval:")

        # Load data with specified interval
        df = load_data("backtests/02_larry01/data/merged_df.csv", interval=interval)
        df = df.dropna(subset=["KRW-BTC_close", "KRW-ETH_close", "KRW-XRP_close"])

        tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
        all_signals = pd.DataFrame(index=df.index)

        # Process each ticker
        for ticker in tickers:
            # Prepare price data for this ticker
            ticker_df = pd.DataFrame(
                {
                    "high": df[f"{ticker}_high"],
                    "low": df[f"{ticker}_low"],
                    "close": df[f"{ticker}_close"],
                },
                index=df.index,
            )

            # Find points and generate signals
            short_term_highs, short_term_lows = find_short_term_points(ticker_df)
            medium_term_highs, medium_term_lows = find_medium_term_points(
                short_term_highs, short_term_lows
            )
            signals = generate_signals(
                ticker_df,
                short_term_highs,
                short_term_lows,
                medium_term_highs,
                medium_term_lows,
            )

            # Store signals
            all_signals[f"{ticker}_signal"] = signals

        # Run investment simulation
        df = pd.concat([df, all_signals], axis=1)
        df = get_investing_simulation(df, tickers, start_cash=100000)

        # Calculate final metrics
        final_balance = df.iloc[-1]["balance"]
        running_max = df["balance"].expanding().max()
        drawdowns = (running_max - df["balance"]) / running_max
        max_drawdown = drawdowns.max() * 100

        # Calculate CAGR
        total_days = (df.index[-1] - df.index[0]).days
        years = total_days / 365
        if years > 0:
            cagr = (((final_balance / 100000) ** (1 / years)) - 1) * 100
        else:
            cagr = ((final_balance / 100000) - 1) * 100

        print(f"Final Balance: {final_balance:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Return: {cagr:.2f}%")

        # Save results with interval in filename
        result_file = f"backtests/02_larry01/results/portfolio_results_{interval}.csv"
        df.to_csv(result_file)
        print(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
