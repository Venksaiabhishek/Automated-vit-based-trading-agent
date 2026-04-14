import os
import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logger = logging.getLogger(__name__)

class AlpacaExecutionManager:
    """
    Handles secure order routing to the Alpaca Paper trading API.
    """
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        # Paper trading defaults to true for safety
        self.paper = True 
        
        if self.api_key and self.secret_key:
            self.client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
            logger.info("Alpaca Trading Client initialized successfully.")
        else:
            self.client = None
            logger.warning("Alpaca Trading Client not initialized: Missing API Keys.")

    def get_account_capital(self) -> float:
        """Fetches the current buying power from the paper account."""
        if not self.client:
            return 10000.0  # Mock capital if keys aren't set
            
        try:
            account = self.client.get_account()
            return float(account.buying_power)
        except Exception as e:
            logger.error(f"Failed to fetch Alpaca account details: {e}")
            return 0.0

    def execute_market_order(self, symbol: str, quantity: float, side: str = "BUY") -> dict:
        """
        Executes a market order via Alpaca.
        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            quantity: Notional amount in USD or shares. Using notional for ease with Kelly criterion.
            side: "BUY" or "SELL"
        """
        if not self.client:
            logger.warning(f"Mock Order: Simulated {side} {quantity} of {symbol}. (Keys not configured)")
            return {"status": "success", "mock": True, "details": "Alpaca integration omitted for safety."}

        # Convert simple string to Enums
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        
        try:
            # We use notional value (fractional shares) to precisely map the Kelly fraction
            order_data = MarketOrderRequest(
                symbol=symbol,
                notional=quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.client.submit_order(order_data=order_data)
            logger.info(f"Order executed successfully: {side} {quantity} Notional of {symbol}. Order ID: {order.id}")
            
            return {
                "status": "success",
                "mock": False,
                "order_id": str(order.id),
                "filled_qty": order.filled_qty
            }
        except Exception as e:
            logger.error(f"Failed to submit market order for {symbol}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

alpaca_manager = AlpacaExecutionManager()
