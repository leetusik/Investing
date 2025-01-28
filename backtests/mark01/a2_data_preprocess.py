import pandas as pd

tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-DOGE"]
tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]

dfs = {}


def make_merged_df():
    for ticker in tickers:
        dfs[ticker] = pd.read_csv(f"backtests/mark01/data/{ticker}.csv", index_col=0)
        dfs[ticker].drop(columns=["market"], inplace=True)
        if "merged_df" not in locals():
            merged_df = dfs[ticker].rename(
                columns={
                    "open": f"{ticker}_open",
                    "high": f"{ticker}_high",
                    "low": f"{ticker}_low",
                    "close": f"{ticker}_close",
                    "volume_krw": f"{ticker}_volume_krw",
                    "volume_market": f"{ticker}_volume_market",
                }
            )
        else:
            temp_df = dfs[ticker].rename(
                columns={
                    "open": f"{ticker}_open",
                    "high": f"{ticker}_high",
                    "low": f"{ticker}_low",
                    "close": f"{ticker}_close",
                    "volume_krw": f"{ticker}_volume_krw",
                    "volume_market": f"{ticker}_volume_market",
                }
            )
            merged_df = pd.merge(merged_df, temp_df, on="time_utc", how="outer")

    return merged_df


def fill_missing_time_and_values(merged_df):
    # Convert 'time_utc' to datetime
    merged_df["time_utc"] = pd.to_datetime(merged_df["time_utc"])

    # Print originally empty cells count in each column before adding missing times
    original_null_values = merged_df.isnull().sum()
    print("Originally empty cells count in each column before adding missing times:")
    print(original_null_values[original_null_values > 0])

    # Determine the full time range and identify missing times
    full_time_range = pd.date_range(
        start=merged_df["time_utc"].min(), end=merged_df["time_utc"].max(), freq="min"
    )
    missing_times = full_time_range.difference(merged_df["time_utc"])

    # Create a DataFrame for the missing times with NaN values
    missing_df = pd.DataFrame(missing_times, columns=["time_utc"])

    # Merge the missing times DataFrame with the original merged DataFrame
    merged_df = pd.concat([merged_df, missing_df], ignore_index=True)

    # Sort the DataFrame by 'time_utc' to maintain chronological order
    merged_df = merged_df.sort_values(by="time_utc").copy()

    # Print originally empty cells count in each column before forward filling
    pre_ffill_null_values = merged_df.isnull().sum()
    print("Originally empty cells count in each column before forward filling:")
    print(pre_ffill_null_values[pre_ffill_null_values > 0])

    # Fill empty values with the very last value
    merged_df.ffill(inplace=True)

    # Drop rows with any remaining NaN values
    merged_df.dropna(inplace=True)

    # Reset the index
    merged_df.reset_index(drop=True, inplace=True)

    # Check if there are null values in the merged DataFrame
    null_values = merged_df.isnull().sum()
    print("Columns with null values and their counts:")
    print(null_values[null_values > 0])

    # Calculate and print the number of artificially added rows
    num_artificial_rows = len(missing_times)
    print(f"Number of artificially added time rows: {num_artificial_rows}")

    return merged_df


def time_utc_start_to_end(df, start_date, end_date):
    df["time_utc"] = pd.to_datetime(df["time_utc"])
    df = df[df["time_utc"] >= start_date]
    df = df[df["time_utc"] <= end_date]
    return df


merged_df = make_merged_df()


merged_df = fill_missing_time_and_values(merged_df)

merged_df = time_utc_start_to_end(merged_df, "2020-01-01", "2024-12-31")

merged_df.to_csv("backtests/mark01/data/merged_df.csv")

merged_df = pd.read_csv("backtests/mark01/data/merged_df.csv", index_col=0)

print(merged_df)
