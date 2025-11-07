#!/usr/bin/env python3
"""Simple test script to verify DataHandler integration works correctly."""

import logging

import pandas as pd

from backtester.data.data_handler import DataHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_handler_integration() -> bool:
    """Test that DataHandler works correctly after integrating DataPreprocessor functionality."""
    logger.info("Testing DataHandler integration...")

    # Create DataHandler with preprocessing enabled
    config = {
        'preprocess_data': True,
        'fill_missing': True,
        'cache_enabled': False,  # Disable cache for testing
    }

    handler = DataHandler(config=config, logger=logger)

    # Create sample data to test with
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    sample_data = pd.DataFrame(
        {
            'Open': 100 + pd.Series(range(len(dates))) * 0.1 + pd.random.normal(0, 1, len(dates)),
            'High': 101 + pd.Series(range(len(dates))) * 0.1 + pd.random.normal(0, 1, len(dates)),
            'Low': 99 + pd.Series(range(len(dates))) * 0.1 + pd.random.normal(0, 1, len(dates)),
            'Close': 100 + pd.Series(range(len(dates))) * 0.1 + pd.random.normal(0, 1, len(dates)),
            'Volume': pd.Series(range(len(dates))) * 1000 + 50000,
        },
        index=dates,
    )

    # Ensure OHLC relationships are logical
    sample_data['High'] = sample_data[['Open', 'Close']].max(axis=1) + 0.5
    sample_data['Low'] = sample_data[['Open', 'Close']].min(axis=1) - 0.5

    try:
        # Test the process method which was causing the issue
        logger.info("Testing process method...")
        processed_data = handler.process(sample_data)

        # Check that technical indicators were added
        expected_indicators = ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 'EMA_20']
        for indicator in expected_indicators:
            if indicator in processed_data.columns:
                logger.info(f"✓ Technical indicator {indicator} was added correctly")
            else:
                logger.error(f"✗ Technical indicator {indicator} was NOT found")
                return False

        # Test other methods
        logger.info("Testing add_technical_indicators method...")
        indicators_added = handler.add_technical_indicators(sample_data)
        for indicator in expected_indicators:
            if indicator in indicators_added.columns:
                logger.info(f"✓ Method add_technical_indicators includes {indicator}")
            else:
                logger.error(f"✗ Method add_technical_indicators missing {indicator}")
                return False

        logger.info("✓ All integration tests passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        return False


def test_broken_reference() -> bool:
    """Test that the old broken reference no longer exists."""
    logger.info("Testing that broken data_preprocessor reference is fixed...")

    handler = DataHandler()

    # This should not raise an AttributeError anymore
    try:
        # The old broken code would be: handler.data_preprocessor.calculate_technical_indicators
        # This should raise AttributeError, which is expected
        _ = handler.data_preprocessor  # type: ignore[attr-defined]
        logger.error("✗ data_preprocessor attribute still exists (unexpected)")
        return False
    except AttributeError:
        logger.info("✓ data_preprocessor attribute correctly removed")
        return True


if __name__ == "__main__":
    print("Testing DataHandler Integration")
    print("=" * 50)

    test1_passed = test_data_handler_integration()
    test2_passed = test_broken_reference()

    if test1_passed and test2_passed:
        print("\n✓ All tests PASSED - Integration successful!")
    else:
        print("\n✗ Some tests FAILED - Integration needs work")
