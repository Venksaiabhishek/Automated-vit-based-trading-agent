"""
Portfolio Manager Module

Handles active user position tracking, profit/loss calculations,
and the intelligent liquidation recommendation engine.
Data is persisted locally to portfolio.json.
"""
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages the user's active holdings and trade execution log."""
    
    def __init__(self, db_path: str = "portfolio.json"):
        self.db_path = db_path
        self.portfolio = self._load_portfolio()
    
    def _load_portfolio(self) -> dict:
        """Loads portfolio state from JSON, creating it if it doesn't exist."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load portfolio: {e}")
                return {"balance": 100000.0, "holdings": {}, "history": []}
        return {"balance": 100000.0, "holdings": {}, "history": []}
    
    def _save_portfolio(self):
        """Persists the portfolio state to JSON."""
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.portfolio, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save portfolio: {e}")

    def buy_stock(self, ticker: str, amount_usd: float, current_price: float) -> dict:
        """Executes a BUY command, updating holdings and tracking entry date."""
        ticker = ticker.upper()
        if self.portfolio["balance"] < amount_usd:
            return {"status": "FAILED", "reason": "Insufficient cash balance"}
        
        shares = amount_usd / current_price
        
        # Add to holdings (averaging entry price if it already exists)
        if ticker in self.portfolio["holdings"]:
            holding = self.portfolio["holdings"][ticker]
            total_shares = holding["shares"] + shares
            total_cost = (holding["shares"] * holding["avg_entry_price"]) + amount_usd
            holding["avg_entry_price"] = total_cost / total_shares
            holding["shares"] = total_shares
            holding["last_purchase"] = datetime.now().isoformat()
        else:
            self.portfolio["holdings"][ticker] = {
                "shares": shares,
                "avg_entry_price": current_price,
                "first_purchase": datetime.now().isoformat(),
                "last_purchase": datetime.now().isoformat()
            }
        
        # Deduct cash
        self.portfolio["balance"] -= amount_usd
        
        # Log history
        trade_log = {
            "date": datetime.now().isoformat(),
            "action": "BUY",
            "ticker": ticker,
            "shares": shares,
            "price": current_price,
            "amount_usd": amount_usd
        }
        self.portfolio["history"].append(trade_log)
        self._save_portfolio()
        logger.info(f"Bought {shares:.4f} shares of {ticker} for ${amount_usd:.2f}")
        return {"status": "SUCCESS", "log": trade_log}

    def sell_stock(self, ticker: str, shares_to_sell: float, current_price: float) -> dict:
        """Executes a SELL command, liquidating shares and realizing PnL."""
        ticker = ticker.upper()
        if ticker not in self.portfolio["holdings"]:
            return {"status": "FAILED", "reason": f"No holdings for {ticker}"}
        
        holding = self.portfolio["holdings"][ticker]
        if shares_to_sell > holding["shares"]:
            return {"status": "FAILED", "reason": "Insufficient shares to sell"}
        
        # Calculate PnL
        cost_basis = shares_to_sell * holding["avg_entry_price"]
        sale_proceeds = shares_to_sell * current_price
        realized_pnl = sale_proceeds - cost_basis
        
        # Update holding
        holding["shares"] -= shares_to_sell
        if holding["shares"] < 1e-6:  # Precision floating point check
            del self.portfolio["holdings"][ticker]
        
        # Update cash
        self.portfolio["balance"] += sale_proceeds
        
        # Log history
        trade_log = {
            "date": datetime.now().isoformat(),
            "action": "SELL",
            "ticker": ticker,
            "shares": shares_to_sell,
            "price": current_price,
            "amount_usd": sale_proceeds,
            "realized_pnl": realized_pnl
        }
        self.portfolio["history"].append(trade_log)
        self._save_portfolio()
        logger.info(f"Sold {shares_to_sell:.4f} shares of {ticker} for ${sale_proceeds:.2f}. PnL: ${realized_pnl:.2f}")
        return {"status": "SUCCESS", "log": trade_log}

    def get_portfolio_summary(self, live_prices: dict) -> dict:
        """
        Generates a huge data dump of the portfolio's current standing,
        used directly by the Dashboard UI.
        `live_prices` is a dict of {ticker: current_price}
        """
        total_invested = 0.0
        total_current_value = 0.0
        details = []
        
        for ticker, data in self.portfolio["holdings"].items():
            shares = data["shares"]
            avg_price = data["avg_entry_price"]
            cost = shares * avg_price
            total_invested += cost
            
            # Use live price or fallback to avg_price if not available
            current_price = live_prices.get(ticker, avg_price)
            current_val = shares * current_price
            total_current_value += current_val
            
            # Days held
            entry_date = datetime.fromisoformat(data["first_purchase"])
            days_held = (datetime.now() - entry_date).days
            
            details.append({
                "ticker": ticker,
                "shares": shares,
                "avg_price": avg_price,
                "current_price": current_price,
                "cost_basis": cost,
                "current_value": current_val,
                "unrealized_pnl": current_val - cost,
                "pnl_pct": ((current_price - avg_price) / avg_price) * 100 if avg_price > 0 else 0,
                "days_held": days_held
            })
        
        total_portfolio_value = self.portfolio["balance"] + total_current_value
        total_pnl = total_current_value - total_invested
        
        return {
            "cash_balance": self.portfolio["balance"],
            "total_invested": total_invested,
            "total_holdings_value": total_current_value,
            "total_portfolio_value": total_portfolio_value,
            "total_unrealized_pnl": total_pnl,
            "total_pnl_pct": (total_pnl / total_invested * 100) if total_invested > 0 else 0.0,
            "holdings": details
        }

    def recommend_liquidation(self, target_cash: float, live_prices: dict, agent_signals: dict = None) -> dict:
        """
        Intelligent Liquidation Engine.
        Finds the optimal mix of stocks to sell to raise `target_cash`.
        Prioritizes stocks that have active BEARISH signals or worst PnL.
        """
        total_value = sum([h['shares'] * live_prices.get(t, h['avg_entry_price']) for t, h in self.portfolio["holdings"].items()])
        if target_cash > total_value:
            return {"status": "FAILED", "reason": f"Target amount ${target_cash:,.2f} exceeds total portfolio value of ${total_value:,.2f}"}

        # Structure holdings for sorting
        sale_candidates = []
        for ticker, data in self.portfolio["holdings"].items():
            curr_price = live_prices.get(ticker, data["avg_entry_price"])
            cost = data["shares"] * data["avg_entry_price"]
            val = data["shares"] * curr_price
            pnl_pct = ((curr_price - data["avg_entry_price"]) / data["avg_entry_price"]) * 100
            
            # Penalty logic: Bearish signals increase priority to sell
            signal_weight = 0
            if agent_signals and ticker in agent_signals:
                sig = agent_signals[ticker].upper()
                if "BEAR" in sig or "SELL" in sig:
                    signal_weight = -100  # Sell these first
                elif "BULL" in sig or "BUY" in sig:
                    signal_weight = 100   # Hold these
            
            # Sort score: Lower score means sell first (Bearish signals go first, then worst performing stocks)
            score = signal_weight + pnl_pct
            
            sale_candidates.append({
                "ticker": ticker,
                "shares_available": data["shares"],
                "current_price": curr_price,
                "current_value": val,
                "score": score
            })
        
        # Sort candidates (lowest score sells first)
        sale_candidates.sort(key=lambda x: x["score"])
        
        # Construct sale orders
        orders = []
        amount_needed = target_cash
        
        for cand in sale_candidates:
            if amount_needed <= 0:
                break
                
            value_available = cand["current_value"]
            if value_available >= amount_needed:
                # We can fulfill the rest of the target just from this holding
                shares_to_sell = amount_needed / cand["current_price"]
                orders.append({
                    "ticker": cand["ticker"],
                    "shares": shares_to_sell,
                    "price": cand["current_price"],
                    "amount_usd": amount_needed
                })
                amount_needed = 0
            else:
                # Sell entire holding and move to next
                orders.append({
                    "ticker": cand["ticker"],
                    "shares": cand["shares_available"],
                    "price": cand["current_price"],
                    "amount_usd": value_available
                })
                amount_needed -= value_available
                
        return {
            "status": "SUCCESS",
            "target_cash": target_cash,
            "orders": orders,
            "message": f"Successfully calculated optimal liquidation mix to raise ${target_cash:,.2f}"
        }

# Singleton accessor
portfolio_manager = PortfolioManager()
