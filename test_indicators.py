#!/usr/bin/env python3
"""Test script to verify indicator system integration."""

import numpy as np
import pandas as pd

from backtester.indicators import IndicatorConfig, IndicatorFactory


def test_indicators() -> bool:
    """Test the indicator system integration."""
    print("Starting indicator system integration test...")

    # Create sample data with proper OHLC relationships
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    np.random.seed(42)

    # Generate base prices
    price_changes = np.random.randn(50) * 2
    closes = 100 + price_changes.cumsum()
    opens = closes + np.random.randn(50) * 0.5
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(50) * 0.5)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(50) * 0.5)

    data = pd.DataFrame(
        {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': np.random.randint(1000, 10000, 50),
        },
        index=dates,
    )

    print(f"Created sample data with {len(data)} periods")

    # Test RSI indicator
    rsi_config = IndicatorConfig(
        indicator_name='test_rsi',
        indicator_type='momentum',
        period=14,
        overbought_threshold=70.0,
        oversold_threshold=30.0,
    )

    rsi_indicator = IndicatorFactory.create('rsi', rsi_config)
    result = rsi_indicator.calculate(data)
    signals = rsi_indicator.generate_signals(result)

    print(f'RSI calculated successfully. Generated {len(signals)} signals.')
    rsi_columns = [col for col in result.columns if 'rsi' in col]
    print(f'RSI columns: {rsi_columns}')

    # Test SMA indicator
    sma_config = IndicatorConfig(indicator_name='test_sma', indicator_type='trend', period=10)

    sma_indicator = IndicatorFactory.create('sma', sma_config)
    sma_result = sma_indicator.calculate(data)
    sma_signals = sma_indicator.generate_signals(sma_result)

    print(f'SMA calculated successfully. Generated {len(sma_signals)} signals.')
    sma_columns = [col for col in sma_result.columns if 'sma' in col]
    print(f'SMA columns: {sma_columns}')

    # Test MACD indicator
    macd_config = IndicatorConfig(
        indicator_name='test_macd',
        indicator_type='momentum',
        fast_period=12,
        slow_period=26,
        signal_period=9,
    )

    macd_indicator = IndicatorFactory.create('macd', macd_config)
    macd_result = macd_indicator.calculate(data)
    macd_signals = macd_indicator.generate_signals(macd_result)

    print(f'MACD calculated successfully. Generated {len(macd_signals)} signals.')
    macd_columns = [col for col in macd_result.columns if 'macd' in col]
    print(f'MACD columns: {macd_columns}')

    # Test factory registration
    available_indicators = IndicatorFactory.get_available_indicators()
    print(f"Available indicators: {available_indicators}")

    print('All integration tests passed successfully!')
    return True


if __name__ == "__main__":
    test_indicators()
