"""
Risk Management & Circuit Breaker Module
Implements:
1. Kelly Criterion guardrail (existing, enhanced)
2. Circuit Breaker — automatic halt at 2% drawdown (new)
3. Post-Mortem report generation (new)
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class RiskGuardrail:
    """
    Deterministic Safety Circuit.
    A non-AI, logic-driven module that acts as a final supervisor.
    Uses the Kelly Criterion to mathematically prevent over-exposure.
    """
    def __init__(self, max_capital_exposure: float = 0.05, win_probability: float = 0.55, win_loss_ratio: float = 2.0):
        self.max_capital_exposure = max_capital_exposure
        self.p_win = win_probability
        self.b = win_loss_ratio
        
    def calculate_kelly_fraction(self, p_win: float = None) -> float:
        """
        Calculates the optimal Kelly bet size (f*).
        Kelly formula: f* = p - (q / b)
        Uses Half-Kelly for reduced volatility.
        """
        p = p_win if p_win is not None else self.p_win
        q = 1.0 - p
        
        if self.b <= 0:
            return 0.0
            
        f_star = p - (q / self.b)
        fraction = max(0.0, f_star * 0.5)  # Half-Kelly
        return fraction
        
    def validate_trade(self, confidence_score: float, current_capital: float, requested_amount: float) -> dict:
        """Evaluates an agent's intended trade against the Kelly Criterion guardrail."""
        adjusted_p_win = max(0.01, min(0.99, self.p_win * confidence_score))
        kelly_fraction = self.calculate_kelly_fraction(adjusted_p_win)
        
        max_allowed_fraction = min(kelly_fraction, self.max_capital_exposure)
        max_allowed_position = current_capital * max_allowed_fraction
        
        approved = False
        message = ""
        
        if kelly_fraction <= 0:
            message = "Rejecting Trade: Negative or Zero Kelly Expectancy indicating math disadvantage."
        elif requested_amount > max_allowed_position:
            message = f"Rejecting Trade: Requested amount ({requested_amount}) exceeds max guardrail allowed ({max_allowed_position:.2f})."
        else:
            approved = True
            message = f"Trade Approved. Inside Kelly constraints (Max Allowed: {max_allowed_position:.2f})"
            
        logger.info(f"Risk Circuit Evaluation: {message}")
        
        return {
            "approved": approved,
            "max_allowed": max_allowed_position,
            "reason": message
        }


class CircuitBreaker:
    """
    Automated Circuit Breaker.
    If the agent loses more than a threshold percentage of capital in a session,
    it automatically halts trading and generates a Post-Mortem report.
    """
    def __init__(self, drawdown_limit: float = 0.02):
        """
        :param drawdown_limit: Maximum drawdown fraction before circuit trips (default: 2%)
        """
        self.drawdown_limit = drawdown_limit
        self.session_trades = []
        self.triggered = False
        self.trigger_time = None
        self.post_mortem = ""
    
    def check(self, session_pnl: float, initial_capital: float) -> dict:
        """
        Check if the circuit breaker should trip.
        Returns dict with 'triggered', 'drawdown_pct', 'message'.
        """
        if initial_capital <= 0:
            return {"triggered": False, "drawdown_pct": 0.0, "message": "No capital to measure against."}
        
        drawdown_pct = abs(min(0, session_pnl)) / initial_capital
        
        if drawdown_pct >= self.drawdown_limit:
            self.triggered = True
            self.trigger_time = datetime.now()
            
            message = (
                f"🚨 CIRCUIT BREAKER TRIGGERED 🚨\n"
                f"Session drawdown: {drawdown_pct*100:.2f}% (limit: {self.drawdown_limit*100:.1f}%)\n"
                f"Session PnL: {session_pnl:+.2f} on {initial_capital:.2f} capital\n"
                f"All trading activity HALTED at {self.trigger_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            logger.warning(message)
            
            return {
                "triggered": True,
                "drawdown_pct": drawdown_pct,
                "message": message
            }
        
        return {
            "triggered": False,
            "drawdown_pct": drawdown_pct,
            "message": f"Circuit OK. Drawdown: {drawdown_pct*100:.2f}% / {self.drawdown_limit*100:.1f}% limit"
        }
    
    def record_trade(self, ticker: str, decision: str, amount: float, pnl: float):
        """Record a trade for post-mortem analysis."""
        self.session_trades.append({
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "decision": decision,
            "amount": amount,
            "pnl": pnl
        })
    
    def generate_post_mortem(self, session_pnl: float, initial_capital: float) -> str:
        """
        Generate a structured Post-Mortem report when the circuit breaker trips.
        Written as Markdown for dashboard display.
        """
        drawdown_pct = abs(min(0, session_pnl)) / initial_capital * 100 if initial_capital > 0 else 0
        
        report = f"""# 🚨 Circuit Breaker Post-Mortem Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Session Summary
| Metric | Value |
|--------|-------|
| Initial Capital | ${initial_capital:,.2f} |
| Session PnL | ${session_pnl:+,.2f} |
| Drawdown | {drawdown_pct:.2f}% |
| Drawdown Limit | {self.drawdown_limit*100:.1f}% |
| Total Trades | {len(self.session_trades)} |

## Trade Log
"""
        if self.session_trades:
            report += "| Time | Ticker | Decision | Amount | PnL |\n"
            report += "|------|--------|----------|--------|-----|\n"
            for t in self.session_trades:
                report += f"| {t['timestamp'][:19]} | {t['ticker']} | {t['decision']} | ${t['amount']:,.2f} | ${t['pnl']:+,.2f} |\n"
            
            # Analysis
            losing_trades = [t for t in self.session_trades if t['pnl'] < 0]
            winning_trades = [t for t in self.session_trades if t['pnl'] > 0]
            worst_trade = min(self.session_trades, key=lambda x: x['pnl']) if self.session_trades else None
            
            report += f"""
## Analysis
- **Winning Trades**: {len(winning_trades)} / {len(self.session_trades)}
- **Losing Trades**: {len(losing_trades)} / {len(self.session_trades)}
- **Worst Trade**: {worst_trade['ticker']} ({worst_trade['decision']}) — ${worst_trade['pnl']:+,.2f}
"""
        else:
            report += "\n*No trades recorded in this session.*\n"
        
        report += """
## Recommendations
1. Review the worst-performing trade for pattern analysis
2. Consider tightening position sizing via Kelly Criterion adjustments
3. Verify that signal conflicts were properly resolved by the Deep Search Agent
4. Agent will remain halted until manual reset or next session

---
*This report was automatically generated by the Circuit Breaker module.*
"""
        self.post_mortem = report
        logger.info("Post-mortem report generated")
        return report
    
    def reset(self):
        """Reset the circuit breaker for a new session."""
        self.triggered = False
        self.trigger_time = None
        self.session_trades = []
        self.post_mortem = ""
        logger.info("Circuit breaker reset for new session")


# Singleton instances
risk_guardrail = RiskGuardrail()
circuit_breaker = CircuitBreaker(drawdown_limit=0.02)
