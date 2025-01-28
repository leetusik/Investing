import numpy as np
import pandas as pd


def load_data(file_path):
    """Load and return the CSV data"""
    return pd.read_csv(file_path, index_col=0)


def main():
    # Constants
    TICKERS = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]

    # Load data
    df = load_data("backtests/mark01/data/vcp_signal.csv")
    # Print all columns in the dataframe
    print("\nColumns in dataframe:")
    for col in df.columns:
        print(col)
    # Save results


if __name__ == "__main__":
    main()
