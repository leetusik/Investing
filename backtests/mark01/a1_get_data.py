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
        # This allows us to use it in a set
        dict_frozen = frozenset(d.items())
        if dict_frozen not in seen:
            seen.add(dict_frozen)
            unique_list.append(d)
    return unique_list


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
        # Get the current time
        current_time = times[1]
        start_time = times[0]
        current_time = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
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
                    # Use json.loads() instead of ast.literal_eval()
                    data = json.loads(response.text)
                    lst += data
                    break  # Exit the loop if the request is successful
                except json.JSONDecodeError as e:
                    print(f"JSON decoding error: {e}")
                    print(f"Response text: {response.text}")
                    break  # Exit the loop on JSON parsing error
            elif response.status_code == 429:
                pass
            else:
                print(f"Unexpected error: {response.status_code}")
                print(f"Response text: {response.text}")
                break

    lst = remove_duplicates(lst)
    # Sort the list of dictionaries by 'candle_date_time_utc'
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


def get_time_intervals(initial_time_str, interval, interval2):
    # Convert the initial time string to a datetime object
    initial_time = datetime.strptime(initial_time_str, "%Y-%m-%d %H:%M:%S")
    # initial_time = initial_time - timedelta(hours=9)

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


def concat_candles(long_df, short_df):
    concatenated_df = pd.concat([long_df, short_df])

    # Drop duplicates based on 'day_starting_at_4am', keeping the last occurrence (from df2)
    final_df = concatenated_df.drop_duplicates(subset="time_utc", keep="last")

    # Sort the DataFrame by 'day_starting_at_4am' to maintain chronological order
    final_df = final_df.sort_values(by="time_utc").copy()

    # Reset the index if needed
    final_df.reset_index(drop=True, inplace=True)

    return final_df


def resample_df(df, execution_time):
    df["time_utc"] = pd.to_datetime(df["time_utc"])

    df.set_index("time_utc", inplace=True)

    # Assuming execution_time is a datetime object, with specific hour and minute (e.g., 13:12)
    daily_df = (
        df.resample(
            "24h",
            offset=pd.Timedelta(
                hours=execution_time.hour, minutes=execution_time.minute
            ),
            origin="epoch",
        )
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume_krw": "sum",
                "volume_market": "sum",
            }
        )
        .reset_index()
    )

    return daily_df


tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-DOGE"]
tickers = ["KRW-XRP", "KRW-DOGE"]

for ticker in tickers:
    df = get_candles(start="2020-01-01 00:00:00", interval2="1", market=ticker)
    df.to_csv(f"backtests/mark01/data/{ticker}.csv")


# ticker = "KRW-BTC"
# df = get_candles(start="2024-09-14 00:00:00", interval2="60", market=ticker)
# df.to_csv("temp_period2.csv")


# df = get_candles(start="2015-01-01 00:00:00", interval2="60")
# df.to_csv("temp_new.csv")
# tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL"]
# for ticker in tickers:
# execution_time = datetime.strptime(f"02:00", "%H:%M")
# df = resample_df(df, execution_time=execution_time)
# df["coin_returns"] = (1 + df["close"].pct_change()).cumprod()
# df.to_csv(f"{ticker}_temp_3.csv")
# {
#     "market": "KRW-BTC",
#     "candle_date_time_utc": "2017-10-30T07:00:00",
#     "candle_date_time_kst": "2017-10-30T16:00:00",
#     "opening_price": 6948000.00000000,
#     "high_price": 6952000.00000000,
#     "low_price": 6925000.00000000,
#     "trade_price": 6952000.00000000,
#     "timestamp": 1509350393761,
#     "candle_acc_trade_price": 159378096.84944000,
#     "candle_acc_trade_volume": 22.95870288,
#     "unit": 60,
# },
# {
#     "market": "KRW-BTC",
#     "candle_date_time_utc": "2017-11-07T15:00:00",
#     "candle_date_time_kst": "2017-11-08T00:00:00",
#     "opening_price": 8104000.00000000,
#     "high_price": 8169000.00000000,
#     "low_price": 8073000.00000000,
#     "trade_price": 8167000.00000000,
#     "timestamp": 1510070395801,
#     "candle_acc_trade_price": 597480801.40376000,
#     "candle_acc_trade_volume": 73.49856848,
#     "unit": 60,
# },
