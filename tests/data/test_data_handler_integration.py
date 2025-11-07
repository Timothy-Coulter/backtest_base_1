"""Tests for DataHandler integration with DataPreprocessor functionality."""

import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import Mock
from backtester.data.data_handler import DataHandler


@pytest.fixture
def sample_data():
    """Create sample OHLC data for testing."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    data = pd.DataFrame(
        {
            'Open': 100 + pd.Series(range(len(dates))) * 0.1 + np.random.normal(0, 0.5, len(dates)),
            'High': 101 + pd.Series(range(len(dates))) * 0.1 + np.random.normal(0, 0.5, len(dates)),
            'Low': 99 + pd.Series(range(len(dates))) * 0.1 + np.random.normal(0, 0.5, len(dates)),
            'Close': 100
            + pd.Series(range(len(dates))) * 0.1
            + np.random.normal(0, 0.5, len(dates)),
            'Volume': pd.Series(range(len(dates))) * 1000 + 50000,
        },
        index=dates,
    )

    # Ensure logical OHLC relationships
    data['High'] = data[['Open', 'Close']].max(axis=1) + 0.5
    data['Low'] = data[['Open', 'Close']].min(axis=1) - 0.5

    return data


@pytest.fixture
def handler():
    """Create DataHandler instance for testing."""
    config = {
        'preprocess_data': True,
        'fill_missing': True,
        'cache_enabled': False,
    }
    logger = Mock(spec=logging.Logger)
    return DataHandler(config=config, logger=logger)


class TestDataHandlerIntegration:
    """Test DataHandler integration after removing DataPreprocessor."""

    def test_data_preprocessor_attribute_removed(self):
        """Test that data_preprocessor attribute no longer exists."""
        handler = DataHandler()

        # This should raise AttributeError
        with pytest.raises(AttributeError):
            _ = handler.data_preprocessor

    def test_process_method_integration(self, handler, sample_data):
        """Test that process method works correctly with integrated functionality."""
        processed_data = handler.process(sample_data)

        # Check that technical indicators were added
        expected_indicators = ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 'EMA_20']

        for indicator in expected_indicators:
            assert indicator in processed_data.columns, f"Technical indicator {indicator} not found"

    def test_add_technical_indicators_method(self, handler, sample_data):
        """Test that add_technical_indicators method works correctly."""
        enhanced_data = handler.add_technical_indicators(sample_data)

        # Check that all expected indicators were added
        expected_indicators = ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 'EMA_20']

        for indicator in expected_indicators:
            assert indicator in enhanced_data.columns, f"Indicator {indicator} missing"

        # Check that original columns are preserved
        original_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in original_columns:
            assert col in enhanced_data.columns, f"Original column {col} missing"

    def test_process_with_preprocess_disabled(self, sample_data):
        """Test that process method works when preprocessing is disabled."""
        config = {'preprocess_data': False, 'fill_missing': False}
        logger = Mock(spec=logging.Logger)
        handler = DataHandler(config=config, logger=logger)

        processed_data = handler.process(sample_data)

        # Should not have technical indicators when preprocessing is disabled
        expected_indicators = ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 'EMA_20']

        for indicator in expected_indicators:
            assert indicator not in processed_data.columns, (
                f"Indicator {indicator} should not be present"
            )

    def test_no_duplicate_functionality(self, handler, sample_data):
        """Test that there's no duplication of functionality."""
        # Test that process and add_technical_indicators produce consistent results
        processed_data = handler.process(sample_data)
        indicators_data = handler.add_technical_indicators(sample_data)

        # Both should have the same indicators
        processed_indicators = [
            col
            for col in processed_data.columns
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']
        ]
        indicators_indicators = [
            col
            for col in indicators_data.columns
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']
        ]

        assert set(processed_indicators) == set(indicators_indicators), (
            "Inconsistent indicators between methods"
        )


if __name__ == "__main__":
    # Run with pytest if executed directly
    pytest.main([__file__, "-v"])
