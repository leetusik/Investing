import ast
import json
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


def remove_duplicates(dict_list):
    seen = set()
    unique_list = []
    for d in dict_list:
        # Create a frozenset from the dictionary items to make it hashable
        dict_frozen = frozenset(d.items())
        if dict_frozen not in seen:
            seen.add(dict_frozen)
            unique_list.append(d)
    return unique_list


def get_time_intervals(initial_time_str, interval, interval2):
    # Convert the initial time string to a datetime object
    initial_time = datetime.strptime(initial_time_str, "%Y-%m-%d %H:%M:%S")

    # Get the current time
    current_time = str(datetime.now(timezone.utc))
    current_time = current_time.split(".")[0]
    current_time = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")

    # Initialize an empty list to store the times
    time_intervals = []
    time_intervals.append(initial_time.strftime("%Y-%m-%d %H:%M:%S"))

    if interval == "minutes":
        # Generate times in 3-hour intervals until the current time
        if interval2 == "1":
            while initial_time <= current_time:
                if current_time - initial_time < timedelta(hours=3, minutes=20):
                    break
                initial_time += timedelta(hours=3, minutes=20)
                time_intervals.append(initial_time.strftime("%Y-%m-%d %H:%M:%S"))
        elif interval2 == "60":
            while initial_time <= current_time:
                if current_time - initial_time < timedelta(hours=200):
                    break
                initial_time += timedelta(hours=200)
                time_intervals.append(initial_time.strftime("%Y-%m-%d %H:%M:%S"))
    elif interval == "hours":
        # add later
        return

    time_intervals.append(current_time.strftime("%Y-%m-%d %H:%M:%S"))
    return time_intervals


def get_candles(
    interval="minutes",
    market="KRW-BTC",
    count="200",
    start=None,
    interval2="1",
):
    if start is None:
        # Default start to one year before the current date and time
        start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d %H:%M:%S")

    times = get_time_intervals(
        initial_time_str=start,
        interval=interval,
        interval2=interval2,
    )
    if len(times) == 2:
        current_time = datetime.strptime(times[1], "%Y-%m-%d %H:%M:%S")
        start_time = datetime.strptime(times[0], "%Y-%m-%d %H:%M:%S")
        time_difference = current_time - start_time
        # when interval minutes 1
        if time_difference < timedelta(hours=4):
            total_minutes = time_difference.total_seconds() // 60 + 1
        # when interval minutes 60
        else:
            total_minutes = time_difference.total_seconds() // 3600 + 1
        count = int(total_minutes)

    times = times[1:]

    lst = []
    for t in times:
        url = f"https://api.upbit.com/v1/candles/{interval}/{interval2}?market={market}&count={count}&to={t}"
        headers = {"accept": "application/json"}
        print(t)
        while True:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                try:
                    data = json.loads(response.text)
                    lst += data
                    break
                except json.JSONDecodeError as e:
                    print(f"JSON decoding error: {e}")
                    print(f"Response text: {response.text}")
                    break
            elif response.status_code == 429:
                pass
            else:
                print(f"Unexpected error: {response.status_code}")
                print(f"Response text: {response.text}")
                break

    lst = remove_duplicates(lst)
    sorted_list = sorted(lst, key=lambda x: x["candle_date_time_utc"])

    df = pd.DataFrame(sorted_list)
    selected_columns = [
        "market",
        "candle_date_time_utc",
        "opening_price",
        "high_price",
        "low_price",
        "trade_price",
        "candle_acc_trade_price",
        "candle_acc_trade_volume",
    ]
    df_selected = df[selected_columns].copy()

    df_selected.rename(
        columns={
            "candle_date_time_utc": "time_utc",
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_price": "volume_krw",
            "candle_acc_trade_volume": "volume_market",
        },
        inplace=True,
    )
    df_selected["time_utc"] = pd.to_datetime(df_selected["time_utc"])
    return df_selected


def make_merged_df(tickers):
    dfs = {}
    merged_df = None

    for ticker in tickers:
        dfs[ticker] = pd.read_csv(f"backtests/mark01/data/{ticker}.csv", index_col=0)
        dfs[ticker].drop(columns=["market"], inplace=True)

        renamed_columns = {
            "open": f"{ticker}_open",
            "high": f"{ticker}_high",
            "low": f"{ticker}_low",
            "close": f"{ticker}_close",
            "volume_krw": f"{ticker}_volume_krw",
            "volume_market": f"{ticker}_volume_market",
        }

        temp_df = dfs[ticker].rename(columns=renamed_columns)

        if merged_df is None:
            merged_df = temp_df
        else:
            merged_df = pd.merge(merged_df, temp_df, on="time_utc", how="outer")

    return merged_df


def fill_missing_time_and_values(merged_df):
    # Convert 'time_utc' to datetime
    merged_df["time_utc"] = pd.to_datetime(merged_df["time_utc"])

    # Print originally empty cells count
    original_null_values = merged_df.isnull().sum()
    print("Originally empty cells count in each column before adding missing times:")
    print(original_null_values[original_null_values > 0])

    # Create full time range and identify missing times
    full_time_range = pd.date_range(
        start=merged_df["time_utc"].min(), end=merged_df["time_utc"].max(), freq="min"
    )
    missing_times = full_time_range.difference(merged_df["time_utc"])

    # Add missing times
    missing_df = pd.DataFrame(missing_times, columns=["time_utc"])
    merged_df = pd.concat([merged_df, missing_df], ignore_index=True)
    merged_df = merged_df.sort_values(by="time_utc").copy()

    # Fill missing values and clean up
    merged_df.ffill(inplace=True)
    merged_df.dropna(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    # Print statistics
    null_values = merged_df.isnull().sum()
    print("\nColumns with null values and their counts:")
    print(null_values[null_values > 0])
    print(f"\nNumber of artificially added time rows: {len(missing_times)}")

    return merged_df


def time_utc_start_to_end(df, start_date, end_date):
    df["time_utc"] = pd.to_datetime(df["time_utc"])
    df = df[df["time_utc"] >= start_date]
    df = df[df["time_utc"] <= end_date]
    return df


def main():
    # Define tickers
    tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]

    # Get data for each ticker
    for ticker in tickers:
        df = get_candles(start="2020-01-01 00:00:00", interval2="1", market=ticker)
        df.to_csv(f"backtests/01_mark01/data/{ticker}.csv")

    # Process and merge data
    merged_df = make_merged_df(tickers)
    merged_df = fill_missing_time_and_values(merged_df)
    merged_df = time_utc_start_to_end(merged_df, "2020-01-01", "2024-12-31")

    # Save final result
    merged_df.to_csv("backtests/01_mark01/data/merged_df.csv")
    print("\nFinal merged dataframe:")
    print(merged_df)


if __name__ == "__main__":
    main()
