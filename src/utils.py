"""Source code utilities.

This module provides utility functions for financial data analysis and portfolio optimization.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def get_data(
    ticker: str,
    start_date: str = "1990-01-01",
    end_date: str = "2025-01-01",
    interval: str = "1mo",
) -> pd.DataFrame:
    """Download and process stock data from Yahoo Finance.

    Args:
        ticker: Stock symbol to download
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d', '1wk', '1mo', etc.)

    Returns:
        DataFrame with processed OHLC data
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval="1mo", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = pd.DataFrame(
            {
                "Close": data["Close"][ticker],
                "Low": data["Low"][ticker],
                "High": data["High"][ticker],
            }
        )
    else:
        data = data[["Close", "Low", "High"]]
    data = data.dropna()

    return data


def kelly_leverage(mu: float, sigma: float, risk_free_rate: float = 0.0) -> float:
    """Compute the Kelly-optimal leverage for a stock or strategy.

    Args:
        mu: Expected return (annualized)
        sigma: Volatility (annualized)
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        Kelly fraction (can be >1 for leveraged positions)
    """
    # Kelly fraction formula
    kelly_fraction = (mu - risk_free_rate) / sigma**2
    return kelly_fraction


def compute_stock_mu_sigma(
    ticker: str, start_date: str, end_date: str, interval: str = '1d'
) -> dict[str, float]:
    """Compute annualized expected return (mu) and volatility (sigma) from historical stock data.

    Args:
        ticker: Stock symbol
        start_date: Start date in 'YYYY-MM-DD'
        end_date: End date in 'YYYY-MM-DD'
        interval: Data interval ('1d', '1wk', '1mo', etc.)

    Returns:
        Dictionary with 'mu': expected return, 'sigma': volatility
    """
    # Download historical price data
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    # If multiple tickers were passed, take only the first column
    if isinstance(data['Close'], pd.DataFrame):
        adj_close = data['Close'].iloc[:, 0]
    else:
        adj_close = data['Close']

    # Compute returns
    returns = adj_close.pct_change().dropna()

    # Annualization factor
    if interval == '1d':
        periods_per_year = 252
    elif interval == '1wk':
        periods_per_year = 52
    elif interval == '1mo':
        periods_per_year = 12
    else:
        raise ValueError("Unsupported interval. Use '1d', '1wk', or '1mo'.")

    # Annualized expected return and volatility
    mu = returns.mean() * periods_per_year
    sigma = returns.std() * np.sqrt(periods_per_year)

    return {'mu': float(mu), 'sigma': float(sigma)}  # ensure float


def optimal_leverage_split(
    sigma: float, mu: float, leverage: float, target_vol: float, capital: float = 1.0
) -> dict[str, float]:
    """Compute optimal allocation between unleveraged and leveraged positions.

    Args:
        sigma: Stock volatility (annualized, e.g., 0.15 for 15%)
        mu: Expected stock return (annualized, e.g., 0.08 for 8%)
        leverage: Leverage factor for leveraged position (e.g., 3.0)
        target_vol: Target portfolio volatility (annualized)
        capital: Total capital (default 1.0, can scale later)

    Returns:
        Dictionary with allocation results:
        - 'w_unleveraged': Weight in unleveraged position
        - 'w_leveraged': Weight in leveraged position
        - 'effective_exposure': Total portfolio exposure
        - 'expected_portfolio_return': Expected portfolio return
    """
    # Effective exposure required to meet target volatility
    target_exposure = target_vol / sigma

    # Solve w_leveraged = (target_exposure - w_unleveraged) / leverage
    # If we want to maximize expected return, allocate as much as possible to leveraged
    # But w_leveraged must be >= 0, so w_unleveraged <= target_exposure
    w_unleveraged = max(0, min(capital, target_exposure))
    w_leveraged = max(0, (target_exposure - w_unleveraged) / leverage)

    effective_exposure = w_unleveraged + w_leveraged * leverage
    expected_portfolio_return = effective_exposure * mu * capital

    return {
        'w_unleveraged': w_unleveraged,
        'w_leveraged': w_leveraged,
        'effective_exposure': effective_exposure,
        'expected_portfolio_return': expected_portfolio_return,
    }


if __name__ == "__main__":
    ticker = 'NDAQ'
    start_date = '1990-01-01'
    end_date = '2025-01-01'

    stats = compute_stock_mu_sigma(ticker, start_date, end_date)
    print(f"{ticker} expected return (mu): {stats['mu']:.4f}")
    print(f"{ticker} volatility (sigma): {stats['sigma']:.4f}")

    sigma = stats['sigma']
    mu = stats['mu']
    leverage = 3
    target_vol = 1.0

    result = optimal_leverage_split(sigma, mu, leverage, target_vol)
    print("Optimal Split:")
    print(result)

    kelly = kelly_leverage(mu=mu, sigma=sigma)
    print(kelly)
