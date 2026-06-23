# circular_economy_RL_lib/market_authority.py

import torch
import torch.nn as nn
from torch.distributions import Normal

class MarketAuthority(nn.Module):
    """
    Stochastic Leader policy parameterizing continuous mechanism decisions.
    Produces: spot_multiplier, cost_multiplier, tx_multiplier, and waste_penalty.
    """
    def __init__(self, num_commodities, hidden_dim=128):
        super(MarketAuthority, self).__init__()
        self.num_commodities = num_commodities
        
        # State dimension is num_commodities * 5 (representing average aggregates)
        self.encoder = nn.Sequential(
            nn.Linear(num_commodities * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Continuous heads for mechanism means
        self.mean_head = nn.Linear(hidden_dim, 4)
        # Learnable log standard deviations for continuous sampling
        self.log_std = nn.Parameter(torch.zeros(4))

    def forward(self, market_state):
        if isinstance(market_state, torch.Tensor) is False:
            market_state = torch.tensor(market_state, dtype=torch.float32)
            
        x = self.encoder(market_state)
        raw_means = self.mean_head(x)
        
        # Project means to reasonable, bounded physical spaces
        spot_mean = 0.5 + torch.sigmoid(raw_means[0])
        uc_mean = 0.5 + torch.sigmoid(raw_means[1])
        tx_mean = 0.5 + torch.sigmoid(raw_means[2])
        waste_mean = torch.sigmoid(raw_means[3])
        
        means = torch.stack([spot_mean, uc_mean, tx_mean, waste_mean])
        stds = torch.exp(self.log_std)
        
        dist = Normal(means, stds)
        sampled_mechanism = dist.sample()
        
        # Clamping samples to enforce physical viability
        spot_mult = torch.clamp(sampled_mechanism[0], min=0.1, max=3.0)
        uc_mult = torch.clamp(sampled_mechanism[1], min=0.1, max=3.0)
        tx_mult = torch.clamp(sampled_mechanism[2], min=0.1, max=3.0)
        waste_penalty = torch.clamp(sampled_mechanism[3], min=0.0, max=1.0)
        
        # Log-probability for policy gradient calculations
        log_prob = dist.log_prob(sampled_mechanism).sum()
        
        mechanism_np = {
            "spot_mult": float(spot_mult.detach().cpu()),
            "uc_mult": float(uc_mult.detach().cpu()),
            "tx_mult": float(tx_mult.detach().cpu()),
            "waste_penalty": float(waste_penalty.detach().cpu())
        }
        
        return mechanism_np, log_prob
