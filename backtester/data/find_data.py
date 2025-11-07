"""Data retrieval utility functions using findatapy library.

This module provides simple utility functions for fetching market data
using the findatapy library with basic configuration examples.
"""

from findatapy.market import Market, MarketDataGenerator, MarketDataRequest

market = Market(market_data_generator=MarketDataGenerator())

# Get you FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html
fred_api_key = "WRITE YOUR KEY HERE"

md_request = MarketDataRequest(
    start_date='year',
    category='fx',
    data_source='alfred',
    tickers=['AUDJPY'],
    fred_api_key=fred_api_key,
)

df = market.fetch_market(md_request)
print(df.tail(n=10))
