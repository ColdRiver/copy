from dataclasses import dataclass
from typing import Dict


@dataclass
class BilevelConfig:
    """
    Configuration for the bilevel extension.

    Inner level:
        Multi-agent policy optimization.

    Outer level:
        Mechanism (subsidy/tax) optimization.
    """

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    inner_lr: float = 1e-3
    outer_lr: float = 1e-4

    inner_updates_per_outer: int = 10
    outer_updates: int = 1

    max_grad_norm: float = 1.0

    # ------------------------------------------------------------------
    # Unrolling
    # ------------------------------------------------------------------

    unroll_length: int = 10
    truncate_unroll: bool = False

    # ------------------------------------------------------------------
    # Episode configuration
    # ------------------------------------------------------------------

    rollout_horizon: int = 100
    mechanism_update_frequency: int = 1

    # ------------------------------------------------------------------
    # Upper-level objective weights
    # ------------------------------------------------------------------

    env_weight: float = 0.40
    econ_weight: float = 0.35
    equity_weight: float = 0.25

    # ------------------------------------------------------------------
    # Mechanism bounds
    # ------------------------------------------------------------------

    subsidy_min: float = -0.50
    subsidy_max: float = 1.00

    tax_min: float = 0.00
    tax_max: float = 1.00

    # ------------------------------------------------------------------
    # Numerical stability
    # ------------------------------------------------------------------

    reward_scale: float = 1e-9
    eps: float = 1e-8

    # ------------------------------------------------------------------
    # Logging / checkpointing
    # ------------------------------------------------------------------

    log_gradient_norms: bool = True
    log_mechanism_values: bool = True

    checkpoint_frequency: int = 100
    save_best_only: bool = True


def get_upper_level_weights(cfg: BilevelConfig) -> Dict[str, float]:
    """
    Returns the scalarization weights used by the
    upper-level objective.
    """
    return {
        "env": cfg.env_weight,
        "econ": cfg.econ_weight,
        "equity": cfg.equity_weight,
    }


def build_bilevel_config(**kwargs) -> BilevelConfig:
    """
    Convenience factory allowing selective overrides.

    Example
    -------
    cfg = build_bilevel_config(
        inner_updates_per_outer=20,
        outer_lr=5e-5
    )
    """
    cfg = BilevelConfig()

    for key, value in kwargs.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown bilevel config parameter: {key}")
        setattr(cfg, key, value)

    return cfg
