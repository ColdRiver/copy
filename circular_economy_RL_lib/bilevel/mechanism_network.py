import torch
import torch.nn as nn


class MechanismNetwork(nn.Module):
    """
    Upper-level mechanism policy.

    Input:
        Aggregated market state.

    Output:
        subsidies : [-subsidy_scale, subsidy_scale]
        taxes     : [0, tax_scale]
    """

    def __init__(
        self,
        state_dim,
        commodity_dim,
        hidden_dim=256,
        init_subsidy_scale=0.5,
        init_tax_scale=0.5,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.commodity_dim = commodity_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.subsidy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, commodity_dim),
        )

        self.tax_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, commodity_dim),
        )

        # Learnable global scales.
        self.subsidy_scale = nn.Parameter(
            torch.tensor(float(init_subsidy_scale))
        )
        self.tax_scale = nn.Parameter(
            torch.tensor(float(init_tax_scale))
        )

    def forward(self, state):
        """
        Parameters
        ----------
        state : Tensor
            Shape:
                [state_dim] or
                [batch_size, state_dim]

        Returns
        -------
        subsidies : Tensor
            Shape [batch_size, commodity_dim]

        taxes : Tensor
            Shape [batch_size, commodity_dim]
        """

        if state.dim() == 1:
            state = state.unsqueeze(0)

        latent = self.encoder(state)

        subsidy_logits = self.subsidy_head(latent)
        tax_logits = self.tax_head(latent)

        subsidies = torch.tanh(subsidy_logits)
        subsidies = subsidies * torch.clamp(
            self.subsidy_scale,
            min=0.0,
            max=1.0,
        )

        taxes = torch.sigmoid(tax_logits)
        taxes = taxes * torch.clamp(
            self.tax_scale,
            min=0.0,
            max=1.0,
        )

        return subsidies, taxes

    @torch.no_grad()
    def predict(self, state):
        """
        Convenience inference wrapper.

        Returns detached tensors for evaluation only.
        Never use this inside differentiable rollouts.
        """
        self.eval()
        subsidies, taxes = self.forward(state)
        return subsidies.detach(), taxes.detach()

    def reset_output_scales(
        self,
        subsidy_scale=0.5,
        tax_scale=0.5,
    ):
        """
        Utility for resetting learnable output scales.
        """
        with torch.no_grad():
            self.subsidy_scale.fill_(subsidy_scale)
            self.tax_scale.fill_(tax_scale)
