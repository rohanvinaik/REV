from dataclasses import dataclass


@dataclass(frozen=True)
class ModeParams:
    """Parameters for REV verification modes"""
    alpha: float      # Significance level
    gamma: float      # Threshold for "same" decision
    eta: float        # Relative precision requirement
    delta_star: float # Threshold for "different" decision
    eps_diff: float   # Relative margin for different decision
    n_min: int        # Minimum samples before decision
    n_max: int        # Maximum samples


class TestingMode:
    """Predefined testing modes for REV verification"""
    QUICK = ModeParams(
        alpha=0.025, gamma=0.15, eta=0.50, 
        delta_star=0.80, eps_diff=0.15, 
        n_min=10, n_max=120
    )
    AUDIT = ModeParams(
        alpha=0.010, gamma=0.10, eta=0.50, 
        delta_star=1.00, eps_diff=0.10, 
        n_min=30, n_max=400
    )
    EXTENDED = ModeParams(
        alpha=0.001, gamma=0.08, eta=0.40, 
        delta_star=1.10, eps_diff=0.08, 
        n_min=50, n_max=800
    )
    # REV-specific mode for memory-bounded verification
    REV_DEFAULT = ModeParams(
        alpha=0.01, gamma=0.05, eta=0.25,
        delta_star=0.20, eps_diff=0.10,
        n_min=20, n_max=500
    )