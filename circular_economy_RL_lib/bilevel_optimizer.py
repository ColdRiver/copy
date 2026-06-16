import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class UpperLevelNetwork(nn.Module):
    def __init__(self, state_dim, subsidy_dim, tax_dim, hidden_dim=256):
        super(UpperLevelNetwork, self).__init__()
        self.subsidy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, subsidy_dim),
            nn.Sigmoid()
        )
        
        self.tax_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, tax_dim),
            nn.Sigmoid()
        )
    
    def forward(self, state):
        subsidies = self.subsidy_net(state) * 2.0 - 1.0
        taxes = self.tax_net(state)
        return subsidies, taxes


class BilevelOptimizer:
    def __init__(self, num_commodities, num_agents, state_dim, lr_upper=1e-4, lr_lower=1e-3):
        self.num_commodities = num_commodities
        self.num_agents = num_agents
        
        self.upper_network = UpperLevelNetwork(
            state_dim=state_dim,
            subsidy_dim=num_commodities,
            tax_dim=num_commodities
        )
        
        self.upper_optimizer = optim.Adam(self.upper_network.parameters(), lr=lr_upper)
        self.lr_lower = lr_lower
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.upper_network.to(self.device)
    
    def compute_mechanisms(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        subsidies, taxes = self.upper_network(state_tensor)
        return subsidies.detach().cpu().numpy(), taxes.detach().cpu().numpy()
    
    def upper_level_loss(self, environmental_metrics, economic_metrics, equity_metrics,
                        weights={'env': 0.4, 'econ': 0.35, 'equity': 0.25}):
        env_loss = torch.tensor(environmental_metrics, dtype=torch.float32).to(self.device).mean()
        econ_loss = torch.tensor(economic_metrics, dtype=torch.float32).to(self.device).mean()
        equity_loss = torch.tensor(equity_metrics, dtype=torch.float32).to(self.device).mean()
        
        total_loss = (weights['env'] * env_loss + 
                     weights['econ'] * econ_loss + 
                     weights['equity'] * equity_loss)
        
        return total_loss
    
    def step(self, trajectory_batch, weights={'env': 0.4, 'econ': 0.35, 'equity': 0.25}):
        states = trajectory_batch['states']
        environmental_metrics = trajectory_batch['env_metrics']
        economic_metrics = trajectory_batch['econ_metrics']
        equity_metrics = trajectory_batch['equity_metrics']
        
        loss = self.upper_level_loss(environmental_metrics, economic_metrics, 
                                     equity_metrics, weights)
        
        self.upper_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.upper_network.parameters(), max_norm=1.0)
        self.upper_optimizer.step()
        
        return loss.item()
    
    def save_checkpoint(self, filepath):
        torch.save(self.upper_network.state_dict(), filepath)
    
    def load_checkpoint(self, filepath):
        self.upper_network.load_state_dict(torch.load(filepath))


class MechanismModule:
    def __init__(self, num_commodities, num_agents):
        self.num_commodities = num_commodities
        self.num_agents = num_agents
        self.subsidy_rates = np.zeros((num_commodities,))
        self.tax_rates = np.zeros((num_commodities,))
    
    def update_mechanisms(self, subsidies, taxes):
        self.subsidy_rates = np.clip(subsidies, -0.5, 1.0)
        self.tax_rates = np.clip(taxes, 0.0, 1.0)
    
    def apply_to_price(self, price, commodity_type='regular'):
        if commodity_type == 'regular':
            return price * (1 + self.tax_rates)
        else:
            return price * (1 - self.subsidy_rates)
    
    def apply_to_reward(self, reward, agent_idx, commodity_type='regular'):
        if commodity_type == 'waste':
            return reward * (1 + self.subsidy_rates)
        return reward
