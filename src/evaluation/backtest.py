import vectorbt as vbt
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def run_dummy_backtest(ticker: str = "AAPL", days: int = 100):
    """
    Shows how vectorbt can be used to perform massive parallel backtests 
    to evaluate the strategy logic over historical timelines.
    """
    logger.info(f"Starting vectorbt backtest for {ticker} over {days} days.")
    
    # Mocking price data since we don't have historical data ingested yet
    np.random.seed(42)
    price = np.cumprod(1 + np.random.normal(0, 0.01, days)) * 100
    price_series = pd.Series(price, index=pd.date_range("2023-01-01", periods=days))
    
    # Mocking strategy entries/exits (which in reality would be driven by the agent)
    # We create a random boolean array for daily buy/sell signals
    entries = pd.Series(np.random.choice([True, False], size=days, p=[0.1, 0.9]), index=price_series.index)
    exits = pd.Series(np.random.choice([True, False], size=days, p=[0.1, 0.9]), index=price_series.index)
    
    # Run the backtest using vectorbt Portfolio
    portfolio = vbt.Portfolio.from_signals(price_series, entries, exits, init_cash=10000, fees=0.001)
    
    # Dump out summary statistics
    stats = portfolio.stats()
    
    print("\n--- Backtest Complete ---")
    print(f"Total Return [%]: {stats.get('Total Return [%]', 'N/A'):.2f}%")
    print(f"Win Rate [%]: {stats.get('Win Rate [%]', 'N/A'):.2f}%")
    print(f"Sharpe Ratio: {stats.get('Sharpe Ratio', 'N/A')}")
    print("-------------------------\n")
    
    return stats

if __name__ == "__main__":
    run_dummy_backtest()
