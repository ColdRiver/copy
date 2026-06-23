import torch
import torch.nn as nn


class MarketAuthority(nn.Module):

    def __init__(self,
                 num_commodities,
                 hidden_dim=128):

        super().__init__()

        self.num_commodities = num_commodities

        self.encoder = nn.Sequential(
            nn.Linear(num_commodities * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.head = nn.Linear(hidden_dim, 4)

    def forward(self, market_state):

        x = self.encoder(market_state)

        raw = self.head(x)

        spot_mult = 0.5 + torch.sigmoid(raw[0])

        uc_mult = 0.5 + torch.sigmoid(raw[1])

        tx_mult = 0.5 + torch.sigmoid(raw[2])

        waste_penalty = torch.sigmoid(raw[3])

        return {
            "spot_mult": spot_mult,
            "uc_mult": uc_mult,
            "tx_mult": tx_mult,
            "waste_penalty": waste_penalty
        }
