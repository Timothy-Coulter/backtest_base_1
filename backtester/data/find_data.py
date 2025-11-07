from findatapy.market import Market, MarketDataRequest, MarketDataGenerator

market = Market(market_data_generator=MarketDataGenerator())

# Get you FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html
fred_api_key = "WRITE YOUR KEY HERE" 

md_request = MarketDataRequest(start_date='year', category='fx', data_source='alfred', tickers=['AUDJPY'],
                               fred_api_key=fred_api_key)

df = market.fetch_market(md_request)
print(df.tail(n=10))