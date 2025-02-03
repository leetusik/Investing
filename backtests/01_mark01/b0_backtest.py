import numpy as np
import pandas as pd


def load_data(file_path):
    """Load and return the CSV data"""
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)  # Convert index to datetime
    return df


def get_vol_indices(df, ticker, min_per_day):
    """Get indices of volumes"""
    df[f"{ticker}_past_24h_vol"] = (
        df[f"{ticker}_volume_krw"].rolling(window=24 * 60, min_periods=24 * 60).mean()
    )
    df[f"{ticker}_ma_vol_10"] = (
        df[f"{ticker}_volume_krw"]
        .rolling(window=10 * min_per_day, min_periods=10 * min_per_day)
        .mean()
    )
    return df


def calculate_ma_signals(df, ticker, min_per_day, periods=[365, 270, 90]):
    """Calculate moving averages and MA signals for a given ticker
    Args:
        df: DataFrame
        ticker: str
        min_per_day: int
        periods: list of int, periods in days for moving averages (default: [365, 270, 90])
    """
    # Sort periods in descending order for proper signal calculation
    periods = sorted(periods, reverse=True)

    # Calculate MAs for all periods
    for period in periods:
        window = period * min_per_day
        df[f"{ticker}_{period}ma"] = (
            df[f"{ticker}_close"].rolling(window=window, min_periods=window).mean()
        )

    # Calculate MA signal based on ascending order condition
    df[f"{ticker}_ma_signal"] = 0

    # Build conditions dynamically based on periods
    conditions = (
        df[f"{ticker}_close"] > df[f"{ticker}_{periods[-1]}ma"]
    )  # Price > shortest MA
    for i in range(len(periods) - 1):
        shorter_ma = df[f"{ticker}_{periods[i+1]}ma"]
        longer_ma = df[f"{ticker}_{periods[i]}ma"]
        conditions &= shorter_ma > longer_ma

    df.loc[conditions, f"{ticker}_ma_signal"] = 1

    return df


def calculate_vcp_metrics(df, ticker, min_base_length=30240):
    """Calculate VCP metrics for a given ticker"""
    length = len(df)
    arrays = {
        "base_length": np.zeros(length, dtype=np.int64),
        "base_start_price": np.zeros(length, dtype=np.float64),
        "vcp": np.zeros(length, dtype=np.bool_),
        "base_max": np.zeros(length, dtype=np.float64),
    }

    # Initialize first row
    arrays["base_start_price"][0] = df.iloc[0][f"{ticker}_close"]
    arrays["base_max"][0] = df.iloc[0][f"{ticker}_high"]

    for idx in range(1, length):
        prev_base_price = arrays["base_start_price"][idx - 1]
        cur_price = df.iloc[idx][f"{ticker}_close"]
        prev_base_max = arrays["base_max"][idx - 1]
        cur_max = df.iloc[idx][f"{ticker}_high"]

        # Base price reset logic
        if (prev_base_price > cur_price * 1.3) or (prev_base_price < cur_price * 0.7):
            arrays["base_start_price"][idx] = cur_price
            arrays["base_length"][idx] = 0
            arrays["base_max"][idx] = cur_max
        else:
            arrays["base_max"][idx] = max(cur_max, prev_base_max)
            arrays["base_start_price"][idx] = prev_base_price
            arrays["base_length"][idx] = arrays["base_length"][idx - 1] + 1

            if arrays["base_length"][idx] > min_base_length:
                start_idx = int(idx - arrays["base_length"][idx])
                mid_idx = int(start_idx + arrays["base_length"][idx] // 2)

                # Calculate first and second half metrics
                first_half_slice = slice(start_idx, mid_idx)
                second_half_slice = slice(mid_idx + 1, idx)

                first_half_returns = df.iloc[first_half_slice][f"{ticker}_close"].std()
                second_half_returns = df.iloc[second_half_slice][
                    f"{ticker}_close"
                ].std()
                first_half_vol = df.iloc[first_half_slice][
                    f"{ticker}_volume_krw"
                ].mean()
                second_half_vol = df.iloc[second_half_slice][
                    f"{ticker}_volume_krw"
                ].mean()

                # VCP signal
                if (
                    df.iloc[idx][f"{ticker}_ma_signal"] == 1
                    and first_half_returns > second_half_returns
                    and first_half_vol > second_half_vol
                ):
                    arrays["vcp"][idx] = True

    # Assign arrays back to dataframe
    column_mapping = {
        "base_length": f"{ticker}_base_length",
        "base_start_price": f"{ticker}_base_start_price",
        "vcp": f"{ticker}_vcp",
        "base_max": f"{ticker}_base_max",
    }

    for arr_name, col_name in column_mapping.items():
        df[col_name] = arrays[arr_name]

    return df


def get_buy_signals(df, ticker, sleepage, loss_rate, profit_rate=1.1):
    """Get buy signals with profit taking
    Args:
        df: DataFrame
        ticker: str
        sleepage: float
        loss_rate: float (e.g., 0.9 for 90% of max price)
        profit_rate: float (e.g., 1.1 for 10% profit target)
    """
    length = len(df)
    arrays = {
        "signal": np.zeros(length, dtype=np.int8),
        "holding_position": np.zeros(length, dtype=np.bool_),
        "buy_price": np.zeros(length, dtype=np.float64),
        "sell_price": np.zeros(length, dtype=np.float64),
        "max_while_holding": np.zeros(length, dtype=np.float64),
    }

    for idx in range(1, length):
        if arrays["holding_position"][idx - 1]:
            arrays["max_while_holding"][idx] = max(
                arrays["max_while_holding"][idx - 1], df.iloc[idx][f"{ticker}_high"]
            )

            # Check profit target first
            if (
                df.iloc[idx][f"{ticker}_high"]
                >= arrays["buy_price"][idx - 1] * profit_rate
            ):
                arrays["signal"][idx] = -1
                arrays["sell_price"][idx] = arrays["buy_price"][idx - 1] * profit_rate
                arrays["holding_position"][idx] = False
            # Then check stop loss
            elif (
                df.iloc[idx][f"{ticker}_low"]
                < arrays["max_while_holding"][idx] * loss_rate
            ):
                arrays["signal"][idx] = -1
                arrays["sell_price"][idx] = df.iloc[idx][f"{ticker}_low"]
                arrays["holding_position"][idx] = False
            else:
                arrays["holding_position"][idx] = True
        else:
            conditions = [
                df.iloc[idx][f"{ticker}_vcp"],
                df.iloc[idx][f"{ticker}_base_max"]
                > df.iloc[idx - 1][f"{ticker}_base_max"],
                # df.iloc[idx][f"{ticker}_past_24h_vol"] > 2 * df.iloc[idx][f"{ticker}_ma_vol_10"],  # Commented out volume check
            ]

            if all(conditions):
                arrays["signal"][idx] = 1
                arrays["holding_position"][idx] = True
                arrays["buy_price"][idx] = df.iloc[idx - 1][f"{ticker}_base_max"] * (
                    1 + sleepage
                )
                arrays["max_while_holding"][idx] = df.iloc[idx][f"{ticker}_high"]

    # Assign arrays back to dataframe
    column_mapping = {
        "signal": f"{ticker}_signal",
        "holding_position": f"{ticker}_holding_position",
        "buy_price": f"{ticker}_buy_price",
        "sell_price": f"{ticker}_sell_price",
        "max_while_holding": f"{ticker}_max_while_holding",
    }

    for arr_name, col_name in column_mapping.items():
        df[col_name] = arrays[arr_name]

    return df


def get_investing_simulation(df, tickers, start_cash):
    """Get investing simulation"""
    df["cash"] = start_cash
    df["coins_value"] = 0.0
    df["balance"] = start_cash
    df["cash_available_portion"] = len(tickers)

    for ticker in tickers:
        df[f"{ticker}_amount"] = 0.0

    for idx in range(1, len(df)):
        cash_temp = df.iloc[idx - 1]["cash"]

        for ticker in tickers:
            if df.iloc[idx][f"{ticker}_signal"] == 1:
                buying_amount = cash_temp * (1 / df.iloc[idx]["cash_available_portion"])
                df.loc[df.index[idx], f"{ticker}_amount"] = (
                    buying_amount / df.iloc[idx][f"{ticker}_buy_price"]
                )
                cash_temp -= buying_amount
                df.loc[df.index[idx], "cash_available_portion"] -= 1

            elif df.iloc[idx][f"{ticker}_signal"] == -1:
                selling_amount = df.iloc[idx - 1][f"{ticker}_amount"]
                cash_temp += selling_amount * df.iloc[idx][f"{ticker}_sell_price"]
                df.loc[df.index[idx], "cash_available_portion"] += 1
                df.loc[df.index[idx], f"{ticker}_amount"] = 0
            else:
                df.loc[df.index[idx], f"{ticker}_amount"] = df.iloc[idx - 1][
                    f"{ticker}_amount"
                ]

        df.loc[df.index[idx], "cash"] = cash_temp
        coins_value = sum(
            df.iloc[idx][f"{ticker}_amount"] * df.iloc[idx][f"{ticker}_close"]
            for ticker in tickers
        )
        df.loc[df.index[idx], "coins_value"] = coins_value
        df.loc[df.index[idx], "balance"] = cash_temp + coins_value

    return df


def process_ticker_data(
    df,
    ticker,
    min_per_day=1440,
    min_base_length=30240,
    sleepage=0.002,
    loss_rate=0.9,
    profit_rate=1.1,
    base_reset_upper=1.3,
    base_reset_lower=0.7,
    ma_periods=[365, 270, 90],
):
    """Process all calculations for a ticker in a single pass"""
    length = len(df)

    # Initialize all arrays at once
    arrays = {
        "base_length": np.zeros(length, dtype=np.int64),
        "base_start_price": np.zeros(length, dtype=np.float64),
        "base_max": np.zeros(length, dtype=np.float64),
        "signal": np.zeros(length, dtype=np.int8),
        "holding_position": np.zeros(length, dtype=np.bool_),
        "buy_price": np.zeros(length, dtype=np.float64),
        "sell_price": np.zeros(length, dtype=np.float64),
        "max_while_holding": np.zeros(length, dtype=np.float64),
        "vcp": np.zeros(length, dtype=np.bool_),
    }

    # Calculate all moving averages and volumes first (these can be vectorized)
    df = get_vol_indices(df, ticker, min_per_day)
    df = calculate_ma_signals(df, ticker, min_per_day, ma_periods)

    # Initialize first row
    arrays["base_start_price"][0] = df.iloc[0][f"{ticker}_close"]
    arrays["base_max"][0] = df.iloc[0][f"{ticker}_high"]

    # Single pass through the data
    for idx in range(1, length):
        cur_price = df.iloc[idx][f"{ticker}_close"]
        cur_high = df.iloc[idx][f"{ticker}_high"]
        cur_low = df.iloc[idx][f"{ticker}_low"]

        # Base price and VCP calculations
        prev_base_price = arrays["base_start_price"][idx - 1]
        prev_base_max = arrays["base_max"][idx - 1]

        if (prev_base_price > cur_price * base_reset_upper) or (
            prev_base_price < cur_price * base_reset_lower
        ):
            arrays["base_start_price"][idx] = cur_price
            arrays["base_length"][idx] = 0
            arrays["base_max"][idx] = cur_high
        else:
            arrays["base_max"][idx] = max(cur_high, prev_base_max)
            arrays["base_start_price"][idx] = prev_base_price
            arrays["base_length"][idx] = arrays["base_length"][idx - 1] + 1

            if arrays["base_length"][idx] > min_base_length:
                start_idx = int(idx - arrays["base_length"][idx])
                mid_idx = int(start_idx + arrays["base_length"][idx] // 2)

                # Use numpy operations for faster calculations
                first_half_prices = df.iloc[start_idx:mid_idx][f"{ticker}_close"].values
                second_half_prices = df.iloc[mid_idx + 1 : idx][
                    f"{ticker}_close"
                ].values
                first_half_vol = df.iloc[start_idx:mid_idx][
                    f"{ticker}_volume_krw"
                ].values
                second_half_vol = df.iloc[mid_idx + 1 : idx][
                    f"{ticker}_volume_krw"
                ].values

                if (
                    df.iloc[idx][f"{ticker}_ma_signal"] == 1
                    and np.std(first_half_prices) > np.std(second_half_prices)
                    and np.mean(first_half_vol) > np.mean(second_half_vol)
                ):
                    arrays["vcp"][idx] = True

        # Position management (combined with VCP check)
        if arrays["holding_position"][idx - 1]:
            arrays["max_while_holding"][idx] = max(
                arrays["max_while_holding"][idx - 1], cur_high
            )

            if cur_low < arrays["max_while_holding"][idx] * loss_rate:
                arrays["signal"][idx] = -1
                arrays["sell_price"][idx] = arrays["max_while_holding"][idx] * loss_rate
                arrays["holding_position"][idx] = False
            else:
                arrays["holding_position"][idx] = True
        else:
            if (
                arrays["vcp"][idx]
                and arrays["base_max"][idx] > arrays["base_max"][idx - 1]
                # df.iloc[idx][f"{ticker}_past_24h_vol"] > volume_threshold * df.iloc[idx][f"{ticker}_ma_vol_10"],  # Commented out volume check
            ):

                arrays["signal"][idx] = 1
                arrays["holding_position"][idx] = True
                arrays["buy_price"][idx] = arrays["base_max"][idx - 1] * (1 + sleepage)
                arrays["max_while_holding"][idx] = cur_high

    # Assign all arrays to dataframe at once
    column_mapping = {
        "base_length": f"{ticker}_base_length",
        "base_start_price": f"{ticker}_base_start_price",
        "base_max": f"{ticker}_base_max",
        "signal": f"{ticker}_signal",
        "holding_position": f"{ticker}_holding_position",
        "buy_price": f"{ticker}_buy_price",
        "sell_price": f"{ticker}_sell_price",
        "max_while_holding": f"{ticker}_max_while_holding",
        "vcp": f"{ticker}_vcp",
    }

    for arr_name, col_name in column_mapping.items():
        df[col_name] = arrays[arr_name]

    df = get_buy_signals(df, ticker, sleepage, loss_rate, profit_rate)
    return df


def run_backtest(
    df,
    tickers,
    min_per_day=1440,
    min_base_length=30240,
    sleepage=0.002,
    loss_rate=0.9,
    profit_rate=1.1,
    start_cash=1.0,
    base_reset_upper=1.3,
    base_reset_lower=0.7,
    ma_periods=[365, 270, 90],
):
    """Run complete backtest with given parameters"""
    # Process each ticker
    for ticker in tickers:
        df = process_ticker_data(
            df,
            ticker,
            min_per_day,
            min_base_length,
            sleepage,
            loss_rate,
            profit_rate,
            base_reset_upper,
            base_reset_lower,
            ma_periods,
        )

    # Filter data to start from 2021-01-01
    df = df[df["time_utc"] >= "2021-01-01"]
    df = df.reset_index(drop=True)  # Reset index after filtering

    # Run investment simulation on filtered data
    df = get_investing_simulation(df, tickers, start_cash)
    final_balance = df.iloc[-1]["balance"]

    # Calculate proper max drawdown
    running_max = df["balance"].expanding().max()
    drawdowns = (running_max - df["balance"]) / running_max
    max_drawdown = drawdowns.max()

    return df, final_balance, max_drawdown


def main():
    # Load data
    df = load_data("backtests/01_mark01/data/merged_df.csv")
    # df = df.tail(10000)
    df = df.dropna(subset=["KRW-BTC_close", "KRW-ETH_close", "KRW-XRP_close"])
    df = df.reset_index(drop=True)  # Reset index after loading

    # Test different parameters
    tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
    min_per_day = 1440  # minutes per day

    parameters = [
        {
            "loss_rate": 0.9,
            "profit_rate": 1.3,
            "base_reset_upper": 1.2,
            "base_reset_lower": 0.8,
            "min_base_length": 21 * min_per_day,
            "ma_periods": [365, 270, 90],
        },
        {
            "loss_rate": 0.95,
            "profit_rate": 1.15,
            "base_reset_upper": 1.15,
            "base_reset_lower": 0.85,
            "min_base_length": 7 * min_per_day,
            "ma_periods": [200, 150, 50],
        },
        {
            "loss_rate": 0.97,
            "profit_rate": 1.1,
            "base_reset_upper": 1.1,
            "base_reset_lower": 0.9,
            "min_base_length": 1 * min_per_day,
            "ma_periods": [7, 5, 3],
        },
    ]

    results = []
    for params in parameters:
        result_df, final_balance, max_drawdown = run_backtest(
            df.copy(), tickers, **params
        )

        # Create parameter string for filename
        param_str = "_".join(
            [
                f"loss{params['loss_rate']}",
                f"profit{params['profit_rate']}",
                f"base_upper{params['base_reset_upper']}",
                f"base_lower{params['base_reset_lower']}",
                f"min_base_len{params['min_base_length']}",
                f"ma_periods{params['ma_periods']}",
            ]
        )

        # Save with comprehensive filename
        result_df.to_csv(f"backtests/01_mark01/results/backtest_{param_str}.csv")

        results.append(
            {
                "parameters": params,
                "final_balance": final_balance,
                "max_drawdown": max_drawdown,
            }
        )

    # Print results summary
    for result in results:
        print("\nParameters:", result["parameters"])
        print(f"Final Balance: {result['final_balance']:.2f}")
        print(f"Max Drawdown: {result['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()
