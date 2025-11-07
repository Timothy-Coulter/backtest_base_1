"""Unit tests for DataHandler class."""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from backtester.data.data_handler import DataHandler


class TestDataHandler:
    """Test DataHandler class methods."""

    @pytest.fixture
    def sample_ohlc_data(self) -> pd.DataFrame:
        """Create sample OHLC data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # Create realistic OHLC data
        np.random.seed(42)  # For reproducible tests
        base_price = 100

        data = pd.DataFrame(index=dates)
        data['Open'] = base_price + np.random.randn(100).cumsum() * 0.5
        data['Close'] = data['Open'] + np.random.randn(100) * 0.3
        data['High'] = np.maximum(data['Open'], data['Close']) + np.abs(np.random.randn(100)) * 0.2
        data['Low'] = np.minimum(data['Open'], data['Close']) - np.abs(np.random.randn(100)) * 0.2
        data['Volume'] = 1000000 + np.random.randint(0, 500000, 100)

        # Ensure High > Low and logical relationships
        data['High'] = np.maximum(data['High'], data['Low'] + 0.01)

        return data

    @pytest.fixture
    def handler(self) -> DataHandler:
        """Create DataHandler instance for testing."""
        config = {
            'data_source': 'yahoo',
            'cache_enabled': False,
            'validation_strict': False,
            'preprocess_data': False,
            'fill_missing': False,
        }
        logger = Mock(spec=logging.Logger)
        return DataHandler(config=config, logger=logger)

    def test_init_default_config(self) -> None:
        """Test DataHandler initialization with default config."""
        handler = DataHandler()

        assert handler.config['data_source'] == 'yahoo'
        assert handler.config['cache_enabled'] is True
        assert handler.config['validation_strict'] is False
        assert handler.config['preprocess_data'] is False
        assert handler.config['fill_missing'] is False
        assert isinstance(handler.data_cache, dict)
        assert isinstance(handler.logger, logging.Logger)

    def test_init_custom_config(self) -> None:
        """Test DataHandler initialization with custom config."""
        custom_config = {
            'data_source': 'test',
            'cache_enabled': False,
            'validation_strict': True,
            'preprocess_data': True,
            'fill_missing': True,
        }
        custom_logger = Mock(spec=logging.Logger)

        handler = DataHandler(config=custom_config, logger=custom_logger)

        assert handler.config == custom_config
        assert handler.logger == custom_logger

    def test_load_data_alias(self, handler: DataHandler, sample_ohlc_data: pd.DataFrame) -> None:
        """Test that load_data is an alias for get_data."""
        with patch.object(handler, 'get_data', return_value=sample_ohlc_data) as mock_get:
            result = handler.load_data('TEST', '2023-01-01', '2023-12-31', '1d')

            mock_get.assert_called_once_with('TEST', '2023-01-01', '2023-12-31', '1d')
            pd.testing.assert_frame_equal(result, sample_ohlc_data)

    @patch.object(DataHandler, 'get_data')
    def test_load_multiple_symbols(
        self, mock_get_data: Mock, handler: DataHandler, sample_ohlc_data: pd.DataFrame
    ) -> None:
        """Test loading multiple symbols."""
        mock_get_data.return_value = sample_ohlc_data

        symbols = ['AAPL', 'GOOGL', 'MSFT']
        result = handler.load_multiple_symbols(symbols, '2023-01-01', '2023-12-31')

        assert isinstance(result, dict)
        assert len(result) == 3

        for symbol in symbols:
            assert symbol in result
            pd.testing.assert_frame_equal(result[symbol], sample_ohlc_data)

        assert mock_get_data.call_count == 3

    def test_aggregate_data_ohlcv(
        self, handler: DataHandler, sample_ohlc_data: pd.DataFrame
    ) -> None:
        """Test data aggregation with OHLCV method."""
        # Create daily data
        daily_data = sample_ohlc_data.resample('D').last()

        # Aggregate to weekly
        weekly_data = handler.aggregate_data(daily_data, 'W', 'ohlcv')

        assert isinstance(weekly_data, pd.DataFrame)
        assert weekly_data.index.freq is not None  # Should have frequency

        # Check that OHLCV columns are present
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            assert col in weekly_data.columns

    def test_aggregate_data_simple(
        self, handler: DataHandler, sample_ohlc_data: pd.DataFrame
    ) -> None:
        """Test data aggregation with simple method."""
        daily_data = sample_ohlc_data.resample('D').last()

        # Aggregate to weekly with simple method
        weekly_data = handler.aggregate_data(daily_data, 'W', 'simple')

        assert isinstance(weekly_data, pd.DataFrame)
        assert weekly_data.index.freq is not None

    def test_clean_data(self, handler: DataHandler, sample_ohlc_data: pd.DataFrame) -> None:
        """Test data cleaning functionality."""
        # Create data with some issues
        dirty_data = sample_ohlc_data.copy()
        dirty_data.loc[dirty_data.index[:5], 'Open'] = np.nan
        dirty_data = dirty_data.sort_index(ascending=False)  # Wrong order

        cleaned_data = handler._clean_data(dirty_data)

        # Should remove NaN values and sort by index
        assert not cleaned_data.isnull().any().any()
        assert cleaned_data.index.is_monotonic_increasing

        # Check numeric conversion
        for col in ['Open', 'High', 'Low', 'Close']:
            assert pd.api.types.is_numeric_dtype(cleaned_data[col])

    def test_validate_data_empty(self, handler: DataHandler) -> None:
        """Test validation with empty data."""
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError, match="Data is empty after cleaning"):
            handler._validate_data(empty_data)

    def test_validate_data_negative_prices(
        self, handler: DataHandler, sample_ohlc_data: pd.DataFrame
    ) -> None:
        """Test validation with negative prices."""
        bad_data = sample_ohlc_data.copy()
        bad_data.loc[bad_data.index[0], 'Close'] = -10

        validated_data = handler._validate_data(bad_data)

        # Should filter out negative prices
        assert (validated_data['Close'] > 0).all()

    def test_validate_data_invalid_ohlc(
        self, handler: DataHandler, sample_ohlc_data: pd.DataFrame
    ) -> None:
        """Test validation with invalid OHLC relationships."""
        bad_data = sample_ohlc_data.copy()
        # Make High < Low
        bad_data.loc[bad_data.index[0], 'High'] = 95
        bad_data.loc[bad_data.index[0], 'Low'] = 105

        # Should not raise in non-strict mode
        validated_data = handler._validate_data(bad_data)

        # Should filter out invalid OHLC
        assert len(validated_data) < len(bad_data)

    def test_validate_data_strict_mode(
        self, handler: DataHandler, sample_ohlc_data: pd.DataFrame
    ) -> None:
        """Test validation in strict mode."""
        bad_data = sample_ohlc_data.copy()
        bad_data.loc[bad_data.index[0], 'High'] = 95
        bad_data.loc[bad_data.index[0], 'Low'] = 105

        config = {'validation_strict': True}
        strict_handler = DataHandler(config=config)

        with pytest.raises(ValueError, match="Data validation failed"):
            strict_handler._validate_data(bad_data)

    def test_compute_returns(self, handler: DataHandler, sample_ohlc_data: pd.DataFrame) -> None:
        """Test return calculation."""
        returns = handler.compute_returns(sample_ohlc_data, 'Close')

        assert isinstance(returns, pd.Series)
        assert len(returns) == len(sample_ohlc_data) - 1  # First value is NaN
        assert not returns.isnull().any()  # After dropna()

        # Check that it's actually returns
        expected_returns = sample_ohlc_data['Close'].pct_change().dropna()
        pd.testing.assert_series_equal(returns, expected_returns, check_names=False)

    def test_compute_returns_invalid_column(
        self, handler: DataHandler, sample_ohlc_data: pd.DataFrame
    ) -> None:
        """Test return calculation with invalid column."""
        with pytest.raises(ValueError, match="Column 'InvalidColumn' not found in data"):
            handler.compute_returns(sample_ohlc_data, 'InvalidColumn')

    def test_add_technical_indicators(
        self, handler: DataHandler, sample_ohlc_data: pd.DataFrame
    ) -> None:
        """Test technical indicators addition."""
        enhanced_data = handler.add_technical_indicators(sample_ohlc_data)

        # Check that all expected indicators were added
        expected_indicators = [
            'SMA_5',
            'SMA_10',
            'SMA_20',
            'EMA_5',
            'EMA_10',
            'EMA_20',
            'Price_vs_SMA5',
            'Price_vs_SMA20',
            'Volatility_20',
            'RSI',
        ]

        for indicator in expected_indicators:
            assert indicator in enhanced_data.columns, f"Missing indicator: {indicator}"

        # Check that original columns are preserved
        original_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in original_columns:
            assert col in enhanced_data.columns

    def test_export_data_csv(
        self, handler: DataHandler, sample_ohlc_data: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test data export to CSV."""
        csv_file = tmp_path / "test_data.csv"

        result_path = handler.export_data(sample_ohlc_data, 'csv', str(csv_file))

        assert csv_file.exists()
        assert result_path == str(csv_file)

        # Verify the exported data (basic check)
        imported_data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        assert len(imported_data) == len(sample_ohlc_data)

    def test_export_data_json(
        self, handler: DataHandler, sample_ohlc_data: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test data export to JSON."""
        json_file = tmp_path / "test_data.json"

        result_path = handler.export_data(sample_ohlc_data, 'json', str(json_file))

        assert json_file.exists()
        assert result_path == str(json_file)

    def test_export_data_invalid_format(
        self, handler: DataHandler, sample_ohlc_data: pd.DataFrame
    ) -> None:
        """Test data export with invalid format."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            handler.export_data(sample_ohlc_data, 'invalid_format')

    def test_get_data_info(self, handler: DataHandler, sample_ohlc_data: pd.DataFrame) -> None:
        """Test data information retrieval."""
        info = handler.get_data_info(sample_ohlc_data)

        assert isinstance(info, dict)
        assert info['shape'] == sample_ohlc_data.shape
        assert info['columns'] == list(sample_ohlc_data.columns)
        assert 'dtypes' in info
        assert 'date_range' in info
        assert 'missing_values' in info

        assert info['date_range']['start'] == sample_ohlc_data.index[0]
        assert info['date_range']['end'] == sample_ohlc_data.index[-1]

    def test_get_statistics(self, handler: DataHandler, sample_ohlc_data: pd.DataFrame) -> None:
        """Test statistics calculation."""
        stats = handler.get_statistics(sample_ohlc_data)

        expected_keys = [
            'start_date',
            'end_date',
            'total_periods',
            'columns',
            'initial_price',
            'final_price',
            'min_price',
            'max_price',
            'total_return',
            'mean_return',
            'volatility',
            'sharpe_ratio',
        ]

        for key in expected_keys:
            assert key in stats, f"Missing statistic: {key}"

        # Verify return calculations
        assert stats['initial_price'] == sample_ohlc_data['Close'].iloc[0]
        assert stats['final_price'] == sample_ohlc_data['Close'].iloc[-1]
        assert stats['total_periods'] == len(sample_ohlc_data)

    def test_process_method(self, handler: DataHandler, sample_ohlc_data: pd.DataFrame) -> None:
        """Test data processing method."""
        config = {'fill_missing': True, 'preprocess_data': True}
        processing_handler = DataHandler(config=config)

        processed_data = processing_handler.process(sample_ohlc_data)

        # Should have technical indicators
        expected_indicators = ['SMA_5', 'EMA_5', 'RSI']
        for indicator in expected_indicators:
            assert indicator in processed_data.columns

    def test_process_method_no_preprocessing(
        self, handler: DataHandler, sample_ohlc_data: pd.DataFrame
    ) -> None:
        """Test data processing with preprocessing disabled."""
        config = {'fill_missing': False, 'preprocess_data': False}
        processing_handler = DataHandler(config=config)

        processed_data = processing_handler.process(sample_ohlc_data)

        # Should not have technical indicators
        expected_indicators = ['SMA_5', 'EMA_5', 'RSI']
        for indicator in expected_indicators:
            assert indicator not in processed_data.columns
