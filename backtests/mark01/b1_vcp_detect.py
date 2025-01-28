import numpy as np
import pandas as pd


def load_data(file_path):
    """Load and return the CSV data"""
    return pd.read_csv(file_path, index_col=0)


def calculate_moving_averages(df, ticker, min_per_day):
    """Calculate various moving averages for a given ticker"""
    periods = [365, 270, 90, 35]

    for period in periods:
        window = period * min_per_day
        if period != 365:
            df[f"{ticker}_{period}ma"] = (
                df[f"{ticker}_close"]
                .rolling(window=window, min_periods=window)
                .mean()
                .fillna(0)
            )
        else:
            df[f"{ticker}_{period}ma"] = (
                df[f"{ticker}_close"].rolling(window=window, min_periods=window).mean()
            )

    # Calculate 365-day min/max
    window_365 = 365 * min_per_day
    df[f"{ticker}_365d_min"] = (
        df[f"{ticker}_close"]
        .rolling(window=window_365, min_periods=window_365)
        .min()
        .fillna(0)
    )
    df[f"{ticker}_365d_max"] = (
        df[f"{ticker}_close"]
        .rolling(window=window_365, min_periods=window_365)
        .max()
        .fillna(0)
    )

    return df


def calculate_ma_signal(df, ticker):
    """Calculate MA signal based on conditions"""
    df[f"{ticker}_ma_signal"] = 0
    conditions = (
        (df[f"{ticker}_close"] > df[f"{ticker}_90ma"])
        & (df[f"{ticker}_90ma"] > df[f"{ticker}_270ma"])
        & (df[f"{ticker}_270ma"] > df[f"{ticker}_365ma"])
    )
    df.loc[conditions, f"{ticker}_ma_signal"] = 1
    return df


def initialize_vcp_columns(df, ticker):
    """Initialize columns needed for VCP calculation"""
    columns = {
        f"{ticker}_base_length": 0,
        f"{ticker}_base_created": 0,
        f"{ticker}_ready_to_buy": 0,
        f"{ticker}_return": 0.0,
        f"{ticker}_first_half_std": 0.0,
        f"{ticker}_second_half_std": 0.0,
        f"{ticker}_std_diff": 0.0,
        f"{ticker}_volume_diff": 0.0,
        f"{ticker}_base_start_price": 0,
        f"{ticker}_first_half_volume": 0.0,
        f"{ticker}_second_half_volume": 0.0,
        f"{ticker}_vcp": False,
        f"{ticker}_breakout_count": 0,
    }

    for col, default_value in columns.items():
        df[col] = default_value
    return df


def calculate_vcp_metrics(df, ticker):
    """Calculate VCP metrics for a given ticker"""
    # Create local arrays
    length = len(df)
    arrays = {
        "base_length": np.zeros(length, dtype=np.int64),
        "base_created": np.zeros(length, dtype=np.int64),
        "return": np.zeros(length, dtype=np.float64),
        "first_half_std": np.zeros(length, dtype=np.float64),
        "second_half_std": np.zeros(length, dtype=np.float64),
        "std_diff": np.zeros(length, dtype=np.float64),
        "volume_diff": np.zeros(length, dtype=np.float64),
        "base_start_price": np.zeros(length, dtype=np.float64),
        "first_half_vol": np.zeros(length, dtype=np.float64),
        "second_half_vol": np.zeros(length, dtype=np.float64),
        "vcp": np.zeros(length, dtype=np.bool_),
        "breakout_count": np.zeros(length, dtype=np.int64),
    }

    # Initialize first row
    arrays["base_start_price"][0] = df.iloc[0][f"{ticker}_close"]

    for idx in range(1, length):
        prev_base_price = arrays["base_start_price"][idx - 1]
        cur_price = df.iloc[idx][f"{ticker}_close"]

        # Maintain previous breakout count when conditions aren't met
        arrays["breakout_count"][idx] = arrays["breakout_count"][idx - 1]

        # Base price reset logic
        if (prev_base_price > cur_price * 1.3) or (prev_base_price < cur_price * 0.7):
            arrays["base_start_price"][idx] = cur_price
            arrays["base_length"][idx] = 0
        else:
            arrays["base_start_price"][idx] = prev_base_price
            arrays["base_length"][idx] = arrays["base_length"][idx - 1] + 1

            if arrays["base_length"][idx] > 30240:  # 21 days in minutes
                arrays["base_created"][idx] = 1

                # Calculate various metrics
                start_idx = int(idx - arrays["base_length"][idx])
                mid_idx = int(start_idx + arrays["base_length"][idx] // 2)

                # Calculate returns and standard deviations
                base_price = arrays["base_start_price"][idx]
                arrays["return"][idx] = (cur_price - base_price) / base_price

                # Calculate first and second half metrics
                first_half_slice = slice(start_idx, mid_idx)
                second_half_slice = slice(mid_idx + 1, idx)

                first_half_returns = (
                    df.iloc[first_half_slice][f"{ticker}_close"] - base_price
                ) / base_price
                second_half_returns = (
                    df.iloc[second_half_slice][f"{ticker}_close"] - base_price
                ) / base_price

                arrays["first_half_std"][idx] = first_half_returns.std()
                arrays["second_half_std"][idx] = second_half_returns.std()
                arrays["std_diff"][idx] = (
                    arrays["first_half_std"][idx] - arrays["second_half_std"][idx]
                )

                # Volume calculations
                arrays["first_half_vol"][idx] = df.iloc[first_half_slice][
                    f"{ticker}_volume_krw"
                ].mean()
                arrays["second_half_vol"][idx] = df.iloc[second_half_slice][
                    f"{ticker}_volume_krw"
                ].mean()
                arrays["volume_diff"][idx] = (
                    arrays["first_half_vol"][idx] - arrays["second_half_vol"][idx]
                )

                # VCP signal
                if (
                    df.iloc[idx][f"{ticker}_ma_signal"] == 1
                    and arrays["std_diff"][idx] > 0
                    and arrays["volume_diff"][idx] > 0
                ):
                    if not arrays["vcp"][idx - 1]:
                        arrays["breakout_count"][idx] = (
                            arrays["breakout_count"][idx - 1] + 1
                        )
                    arrays["vcp"][idx] = True
                else:
                    arrays["vcp"][idx] = False

    # Assign arrays back to dataframe
    column_mapping = {
        "base_length": f"{ticker}_base_length",
        "base_created": f"{ticker}_base_created",
        "return": f"{ticker}_return",
        "first_half_std": f"{ticker}_first_half_std",
        "second_half_std": f"{ticker}_second_half_std",
        "std_diff": f"{ticker}_std_diff",
        "volume_diff": f"{ticker}_volume_diff",
        "base_start_price": f"{ticker}_base_start_price",
        "first_half_vol": f"{ticker}_first_half_volume",
        "second_half_vol": f"{ticker}_second_half_volume",
        "vcp": f"{ticker}_vcp",
        "breakout_count": f"{ticker}_breakout_count",
    }

    for arr_name, col_name in column_mapping.items():
        df[col_name] = arrays[arr_name]

    return df


def main():
    # Constants
    MIN_PER_DAY = 24 * 60
    TICKERS = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]

    # Load data
    df = load_data("backtests/mark01/data/merged_df.csv")

    # Calculate indicators for each ticker
    for ticker in TICKERS:
        df = calculate_moving_averages(df, ticker, MIN_PER_DAY)
        df = calculate_ma_signal(df, ticker)
        print(f"{ticker} MA signal counts:", df[f"{ticker}_ma_signal"].value_counts())

    # Drop NA values
    print(f"Before dropna: {len(df)}")
    for ticker in TICKERS:
        df = df.dropna(subset=[f"{ticker}_365ma"])
    print(f"After dropna: {len(df)}")

    # Calculate VCP metrics
    for ticker in TICKERS:
        df = initialize_vcp_columns(df, ticker)
        df = calculate_vcp_metrics(df, ticker)
        print(
            f"Max breakout count for {ticker}: {df[f'{ticker}_breakout_count'].max()}"
        )

    # Save results
    df.to_csv("backtests/mark01/data/vcp_signal.csv", index=True)


if __name__ == "__main__":
    main()
