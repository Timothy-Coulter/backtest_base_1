"""Market data handler for loading and managing financial data.

This module provides functionality to load, validate, and manage market data
for backtesting purposes. It handles data from yfinance and other sources.
"""

import logging
from typing import Any

import pandas as pd
import yfinance as yf


class DataHandler:
    """Handles market data loading, validation, and preprocessing."""

    def __init__(
        self, config: dict[str, Any] | None = None, logger: logging.Logger | None = None
    ) -> None:
        """Initialize the data handler.

        Args:
            config: Optional configuration dictionary
            logger: Optional logger instance for logging operations.
        """
        self.config = config or {
            'data_source': 'yahoo',
            'cache_enabled': True,
            'validation_strict': False,
            'preprocess_data': False,
            'fill_missing': False,
        }
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

        # Data cache for performance
        self.data_cache: dict[str, pd.DataFrame] = {}

    def get_data(
        self,
        ticker: str,
        start_date: str = "1990-01-01",
        end_date: str = "2025-01-01",
        interval: str = "1mo",
    ) -> pd.DataFrame:
        """Load market data for a given ticker.

        Args:
            ticker: Stock/asset ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data frequency ('1d', '1wk', '1mo')

        Returns:
            DataFrame with OHLC data (Open, High, Low, Close)

        Raises:
            ValueError: If data cannot be loaded or is invalid
        """
        try:
            self.logger.info(f"Loading data for {ticker} from {start_date} to {end_date}")

            # Check cache first
            cache_key = f"{ticker}_{start_date}_{end_date}_{interval}"
            if self.config.get('cache_enabled', True) and cache_key in self.data_cache:
                self.logger.info(f"Using cached data for {ticker}")
                return self.data_cache[cache_key]

            # Download data from yfinance
            data: pd.DataFrame = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=False,
            )

            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")

            # Handle multi-index columns (when downloading multiple tickers)
            if isinstance(data.columns, pd.MultiIndex):
                data = pd.DataFrame(
                    {
                        "Open": data["Open"][ticker],
                        "High": data["High"][ticker],
                        "Low": data["Low"][ticker],
                        "Close": data["Close"][ticker],
                        "Volume": data["Volume"][ticker] if "Volume" in data.columns else 0,
                    }
                )
            else:
                # Ensure we have the required columns
                required_columns = ["Open", "High", "Low", "Close"]
                if not all(col in data.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in data.columns]
                    raise ValueError(f"Missing required columns: {missing}")

                # Select only OHLC columns and add volume if available
                columns = ["Open", "High", "Low", "Close"]
                if "Volume" in data.columns:
                    columns.append("Volume")
                data = data[columns]

            # Clean and validate data
            data = self._clean_data(data)
            data = self._validate_data(data)

            # Process data if configured
            if self.config.get('preprocess_data', False):
                data = self.process(data)

            # Cache the data
            if self.config.get('cache_enabled', True):
                self.data_cache[cache_key] = data

            self.logger.info(f"Successfully loaded {len(data)} records for {ticker}")
            return data

        except Exception as e:
            self.logger.error(f"Error loading data for {ticker}: {e}")
            raise ValueError(f"Failed to load data for {ticker}: {e}") from e

    def load_data(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """Load data for a single symbol (alias for get_data)."""
        return self.get_data(symbol, start_date, end_date, interval)

    def load_multiple_symbols(
        self, symbols: list[str], start_date: str, end_date: str, interval: str = "1d"
    ) -> dict[str, pd.DataFrame]:
        """Load data for multiple symbols."""
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_data(symbol, start_date, end_date, interval)
        return results

    def aggregate_data(
        self, data: pd.DataFrame, target_frequency: str, aggregation_method: str = "ohlcv"
    ) -> pd.DataFrame:
        """Aggregate data to target frequency."""
        if aggregation_method == "ohlcv":
            # Simple aggregation by resampling
            aggregated = (
                data.resample(target_frequency)
                .agg(
                    {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
                )
                .dropna()
            )
            return aggregated
        else:
            return data.resample(target_frequency).last().dropna()

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data according to configuration."""
        processed = data.copy()

        if self.config.get('fill_missing', False):
            processed = processed.ffill()

        if self.config.get('preprocess_data', False):
            processed = self.add_technical_indicators(processed)

        return processed

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the loaded data.

        Args:
            data: Raw DataFrame from yfinance

        Returns:
            Cleaned DataFrame
        """
        # Remove any rows with NaN values
        data = data.dropna()

        # Ensure data is sorted by date
        data = data.sort_index()

        # Convert numeric columns to float if needed
        numeric_columns = ["Open", "High", "Low", "Close"]
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        return data

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate the cleaned data for common issues.

        Args:
            data: Cleaned DataFrame

        Returns:
            Validated DataFrame

        Raises:
            ValueError: If data fails validation checks
        """
        if len(data) == 0:
            raise ValueError("Data is empty after cleaning")

        # Check for reasonable price ranges
        if "Close" in data.columns and (data["Close"] <= 0).any():
            self.logger.warning("Found non-positive prices in data")
            data = data[data["Close"] > 0]

        # Check for logical OHLC relationships
        if all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
            invalid_ohlc = (
                (data["High"] < data["Low"])
                | (data["High"] < data["Open"])
                | (data["High"] < data["Close"])
                | (data["Low"] > data["Open"])
                | (data["Low"] > data["Close"])
            )

            if invalid_ohlc.any():
                self.logger.warning(f"Found {invalid_ohlc.sum()} invalid OHLC records")
                if self.config.get('validation_strict', False):
                    raise ValueError("Data validation failed: invalid OHLC relationships")
                data = data[~invalid_ohlc]

        return data

    def compute_returns(self, data: pd.DataFrame, column: str = "Close") -> pd.Series:
        """Compute returns for a given price column.

        Args:
            data: DataFrame with price data
            column: Column name to compute returns for

        Returns:
            Series of returns
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        return data[column].pct_change().dropna()

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to the data.

        Args:
            data: DataFrame with OHLC data

        Returns:
            DataFrame with added technical indicators
        """
        data = data.copy()

        # Simple Moving Averages
        data["SMA_5"] = data["Close"].rolling(window=5).mean()
        data["SMA_10"] = data["Close"].rolling(window=10).mean()
        data["SMA_20"] = data["Close"].rolling(window=20).mean()

        # Exponential Moving Averages
        data["EMA_5"] = data["Close"].ewm(span=5).mean()
        data["EMA_10"] = data["Close"].ewm(span=10).mean()
        data["EMA_20"] = data["Close"].ewm(span=20).mean()

        # Price ratios
        data["Price_vs_SMA5"] = data["Close"] / data["SMA_5"]
        data["Price_vs_SMA20"] = data["Close"] / data["SMA_20"]

        # Volatility (rolling standard deviation)
        data["Volatility_20"] = data["Close"].rolling(window=20).std()

        # RSI (Relative Strength Index)
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))

        return data

    def export_data(self, data: pd.DataFrame, format: str = "csv", path: str | None = None) -> str:
        """Export data in specified format."""
        if format.lower() == "csv":
            filename = path or "data_export.csv"
            data.to_csv(filename)
            return filename
        elif format.lower() == "json":
            filename = path or "data_export.json"
            data.to_json(filename)
            return filename
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_data_info(self, data: pd.DataFrame) -> dict[str, Any]:
        """Get information about the data."""
        return {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict(),
            "date_range": {
                "start": data.index[0] if len(data) > 0 else None,
                "end": data.index[-1] if len(data) > 0 else None,
            },
            "missing_values": data.isnull().sum().to_dict(),
        }

    def get_statistics(self, data: pd.DataFrame) -> dict[str, Any]:
        """Get basic statistics about the loaded data.

        Args:
            data: DataFrame with market data

        Returns:
            Dictionary with statistics
        """
        stats = {
            "start_date": data.index[0],
            "end_date": data.index[-1],
            "total_periods": len(data),
            "columns": list(data.columns),
        }

        if "Close" in data.columns:
            stats.update(
                {
                    "initial_price": data["Close"].iloc[0],
                    "final_price": data["Close"].iloc[-1],
                    "min_price": data["Close"].min(),
                    "max_price": data["Close"].max(),
                    "total_return": (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100,
                }
            )

            returns = self.compute_returns(data, "Close")
            stats.update(
                {
                    "mean_return": returns.mean(),
                    "volatility": returns.std(),
                    "sharpe_ratio": returns.mean() / returns.std() if returns.std() > 0 else 0,
                }
            )

        return stats
