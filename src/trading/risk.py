import logging

logger = logging.getLogger(__name__)

class RiskGuardrail:
    """
    Deterministic Safety Circuit.
    A non-AI, logic-driven module that acts as a final supervisor.
    Uses the Kelly Criterion to mathematically prevent over-exposure.
    """
    def __init__(self, max_capital_exposure: float = 0.05, win_probability: float = 0.55, win_loss_ratio: float = 2.0):
        """
        :param max_capital_exposure: Maximum percentage of total capital allowed per trade (e.g., 5%).
        :param win_probability: Estimated probability of a winning trade (p). 
        :param win_loss_ratio: Average risk/reward ratio (b). (E.g., 2.0 means winning $200 for every $100 risked).
        """
        self.max_capital_exposure = max_capital_exposure
        self.p_win = win_probability
        self.b = win_loss_ratio
        
    def calculate_kelly_fraction(self, p_win: float = None) -> float:
        """
        Calculates the optimal Kelly bet size (f*).
        Kelly formula: f* = p - (q / b)
        where q = 1 - p
        """
        p = p_win if p_win is not None else self.p_win
        q = 1.0 - p
        
        if self.b <= 0:
            return 0.0
            
        f_star = p - (q / self.b)
        
        # Kelly can be aggressive; traders often use "Half-Kelly" to reduce volatility
        fraction = max(0.0, f_star * 0.5) 
        return fraction
        
    def validate_trade(self, confidence_score: float, current_capital: float, requested_amount: float) -> dict:
        """
        Evaluates an agent's intended trade against the Kelly Criterion guardrail.
        :param confidence_score: Agent's confidence (0.0 to 1.0), used to dynamically adjust win probability.
        :param current_capital: Total capital available.
        :param requested_amount: The size of the position the agent wants to open.
        """
        # Dynamically adjust win probability based on agent's confidence
        # (Assuming baseline p_win is scaled by confidence)
        adjusted_p_win = max(0.01, min(0.99, self.p_win * confidence_score))
        
        kelly_fraction = self.calculate_kelly_fraction(adjusted_p_win)
        
        # Absolute structural limit
        max_allowed_fraction = min(kelly_fraction, self.max_capital_exposure)
        max_allowed_position = current_capital * max_allowed_fraction
        
        approved = False
        message = ""
        
        if kelly_fraction <= 0:
            message = "Rejecting Trade: Negative or Zero Kelly Expectancy indicating math disadvantage."
        elif requested_amount > max_allowed_position:
            message = f"Rejecting Trade: Requested amount ({requested_amount}) exceeds max guardrail allowed ({max_allowed_position})."
        else:
            approved = True
            message = f"Trade Approved. Inside Kelly constraints (Max Allowed: {max_allowed_position})"
            
        logger.info(f"Risk Circuit Evaluation: {message}")
        
        return {
            "approved": approved,
            "max_allowed": max_allowed_position,
            "reason": message
        }

# Singleton instance
risk_guardrail = RiskGuardrail()
