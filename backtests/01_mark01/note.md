
## a0_get_data.py summary
- Fetches minute-level candle data from Upbit API for multiple tickers (BTC, ETH, XRP)
- Key functions:
  - `get_candles()`: Retrieves raw candle data from API
  - `make_merged_df()`: Combines data from multiple tickers
  - `fill_missing_time_and_values()`: Ensures continuous minute-by-minute data
- Saves both individual ticker CSVs and final merged dataset

## b0_backtest.py summary
- Implements VCP (Volatility Contraction Pattern) trading strategy
- Key components:
  - Moving average signals using multiple periods (e.g., [365, 270, 90] days)
  - Base price tracking with adjustable reset thresholds
  - Position management with stop-loss and profit targets
- Backtest parameters:
  - Loss rate (e.g., 0.9 for 10% stop-loss)
  - Profit rate (e.g., 1.3 for 30% target)
  - Base reset thresholds (upper/lower)
  - Minimum base length (in days)
  - MA periods for trend confirmation
- Outputs performance metrics including final balance and max drawdown
