import torch
import torch.nn as nn


class MechanismModule(nn.Module):
    """
    Differentiable mechanism operator.

    Stores current subsidy/tax outputs produced by
    the upper-level mechanism network and applies
    them inside the simulator.

    IMPORTANT:
    Never detach stored mechanisms.
    """

    def __init__(
        self,
        num_commodities,
        subsidy_min=-0.50,
        subsidy_max=1.00,
        tax_min=0.00,
        tax_max=1.00,
        use_exponential_coupling=True,
    ):
        super().__init__()

        self.num_commodities = num_commodities

        self.subsidy_min = subsidy_min
        self.subsidy_max = subsidy_max

        self.tax_min = tax_min
        self.tax_max = tax_max

        self.use_exponential_coupling = use_exponential_coupling

        self.subsidies = None
        self.taxes = None

    def update_mechanisms(
        self,
        subsidies,
        taxes,
    ):
        """
        Store references directly.

        DO NOT:
            detach
            clone.detach
            numpy conversion

        because outer gradients must propagate.
        """

        self.subsidies = torch.clamp(
            subsidies,
            min=self.subsidy_min,
            max=self.subsidy_max,
        )

        self.taxes = torch.clamp(
            taxes,
            min=self.tax_min,
            max=self.tax_max,
        )

    def has_mechanisms(self):
        return (
            self.subsidies is not None
            and self.taxes is not None
        )

    def apply_to_price(
        self,
        price,
        commodity_type="regular",
    ):
        """
        Parameters
        ----------
        price:
            [num_agents, num_commodities]

        Returns
        -------
        Mechanism-adjusted price.
        """

        if not self.has_mechanisms():
            return price

        subsidies = self.subsidies
        taxes = self.taxes

        if subsidies.dim() > 1:
            subsidies = subsidies.squeeze(0)

        if taxes.dim() > 1:
            taxes = taxes.squeeze(0)

        if self.use_exponential_coupling:

            if commodity_type == "regular":
                return price * torch.exp(taxes)

            return price * torch.exp(-subsidies)

        else:

            if commodity_type == "regular":
                return price * (1.0 + taxes)

            return price * (1.0 - subsidies)

    def apply_to_reward(
        self,
        reward,
        agent_idx=None,
        commodity_type="regular",
    ):
        """
        Optional reward shaping.

        By default we leave rewards untouched.

        Upper-level effects should primarily enter
        through market prices rather than ad-hoc
        reward multipliers.
        """

        return reward

    def get_current_mechanisms(self):
        return {
            "subsidies": self.subsidies,
            "taxes": self.taxes,
        }

    def reset(self):
        self.subsidies = None
        self.taxes = None
