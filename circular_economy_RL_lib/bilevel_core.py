import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MechanismNetwork(nn.Module):
    def __init__(self, state_dim, commodity_dim, hidden_dim=256):
        super().__init__()
        
        self.subsidy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, commodity_dim)
        )
        
        self.tax_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, commodity_dim)
        )
    
    def forward(self, state):
        subsidies = torch.tanh(self.subsidy_net(state))
        taxes = torch.sigmoid(self.tax_net(state))
        return subsidies, taxes
    
    def get_mechanisms(self, state_np):
        with torch.no_grad():
            state_t = torch.tensor(state_np, dtype=torch.float32)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            subsidies, taxes = self(state_t)
        return subsidies.numpy(), taxes.numpy()


class DifferentiableMarketState:
    def __init__(self, num_agents, num_commodities, episode_length, history_length):
        self.num_agents = num_agents
        self.num_commodities = num_commodities
        self.episode_length = episode_length
        self.history_length = history_length
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.price = torch.zeros(num_agents, num_commodities, dtype=torch.float32, device=self.device, requires_grad=False)
        self.waste_price = torch.zeros(num_agents, num_commodities, dtype=torch.float32, device=self.device, requires_grad=False)
        self.spot_price = torch.ones(num_commodities, dtype=torch.float32, device=self.device, requires_grad=False)
        
        self.inv = torch.ones(num_agents, num_commodities, dtype=torch.float32, device=self.device, requires_grad=False) * 100
        self.waste_inv = torch.ones(num_agents, num_commodities, dtype=torch.float32, device=self.device, requires_grad=False) * 50
        
        self.q = torch.zeros(num_agents, num_agents, num_commodities, dtype=torch.float32, device=self.device, requires_grad=False)
        self.waste_q = torch.zeros(num_agents, num_agents, num_commodities, dtype=torch.float32, device=self.device, requires_grad=False)
        self.spot_q = torch.zeros(num_agents, num_commodities, dtype=torch.float32, device=self.device, requires_grad=False)
        
        self.actual_d = torch.zeros(num_agents, num_agents, num_commodities, dtype=torch.float32, device=self.device, requires_grad=False)
        self.waste_actual_d = torch.zeros(num_agents, num_agents, num_commodities, dtype=torch.float32, device=self.device, requires_grad=False)
        
        self.eco_u = torch.zeros(num_agents, num_commodities, dtype=torch.float32, device=self.device, requires_grad=False)
        self.tx_u = torch.zeros(num_agents, num_commodities, dtype=torch.float32, device=self.device, requires_grad=False)
        
        self.wastewater = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=False)
        
        self.subsidies = torch.zeros(num_commodities, dtype=torch.float32, device=self.device, requires_grad=True)
        self.taxes = torch.zeros(num_commodities, dtype=torch.float32, device=self.device, requires_grad=True)
    
    def apply_mechanisms(self, price, waste_price):
        adjusted_price = price * (1.0 + self.taxes.unsqueeze(0))
        adjusted_waste_price = waste_price * (1.0 - self.subsidies.unsqueeze(0))
        return adjusted_price, adjusted_waste_price
    
    def update_mechanisms(self, subsidies, taxes):
        self.subsidies = subsidies.clone().detach().requires_grad_(True)
        self.taxes = taxes.clone().detach().requires_grad_(True)


class BilevelDifferentiableSimulator:
    def __init__(self, mechanism_network, num_agents=3, num_commodities=12, 
                 episode_length=1000, history_length=5):
        
        self.mechanism_network = mechanism_network
        self.num_agents = num_agents
        self.num_commodities = num_commodities
        self.episode_length = episode_length
        self.history_length = history_length
        
        self.state = DifferentiableMarketState(
            num_agents, num_commodities, episode_length, history_length
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mechanism_network.to(self.device)
        
        self.UC = 0.5
        self.TX_P = 0.5
        self.delta = 0.5
        self.LAMBDA = 0.5
        self.RWD_SCALE = 1e-9
        self.INIT_INV = 100
        
        self.t = 0
    
    def reset(self):
        self.t = self.history_length
        
        self.state.price.zero_()
        self.state.waste_price.zero_()
        self.state.spot_price.fill_(0.5)
        
        self.state.inv.fill_(self.INIT_INV)
        self.state.waste_inv.fill_(50)
        
        self.state.q.zero_()
        self.state.waste_q.zero_()
        self.state.spot_q.zero_()
        
        self.state.actual_d.zero_()
        self.state.waste_actual_d.zero_()
        
        self.state.eco_u.zero_()
        self.state.tx_u.zero_()
        
        self.state.wastewater.zero_()
        
        return self.get_seller_state().detach().cpu().numpy()
    
    def get_seller_state(self):
        state_components = [
            self.state.spot_price,
            self.state.price.view(-1),
            self.state.waste_price.view(-1),
            self.state.inv.view(-1),
            self.state.waste_inv.view(-1),
            self.state.q.view(-1),
            self.state.waste_q.view(-1),
        ]
        state = torch.cat(state_components)
        
        if state.dim() == 1:
            state = state.unsqueeze(0).expand(self.num_agents, -1)
        
        return state
    
    def action_conversion(self, actions, keys_lengths):
        converted = {}
        for i, (key, length) in enumerate(keys_lengths.items()):
            converted[key] = actions[:, i*length:(i+1)*length]
        return converted
    
    def step_sell(self, actions_seller):
        keys_lengths = {'price': self.num_commodities, 'waste_price': self.num_commodities}
        actions = self.action_conversion(actions_seller, keys_lengths)
        
        adjusted_price, adjusted_waste_price = self.state.apply_mechanisms(
            actions['price'], actions['waste_price']
        )
        
        self.state.price = torch.clamp(adjusted_price, 0.0, 1000.0)
        self.state.waste_price = torch.clamp(adjusted_waste_price, 0.0, 1000.0)
        
        buyer_state = self.get_seller_state()
        return buyer_state
    
    def step_buy(self, actions_buyer):
        keys_lengths = {
            'q': (self.num_agents - 1) * self.num_commodities,
            'waste_q': (self.num_agents - 1) * self.num_commodities,
            'spot_q': self.num_commodities
        }
        actions = self.action_conversion(actions_buyer, keys_lengths)
        
        q_reshaped = actions['q'].view(self.num_agents, self.num_agents - 1, self.num_commodities)
        waste_q_reshaped = actions['waste_q'].view(self.num_agents, self.num_agents - 1, self.num_commodities)
        
        self.state.q.zero_()
        self.state.waste_q.zero_()
        
        for i in range(self.num_agents):
            i_list = [j for j in range(self.num_agents) if j != i]
            self.state.q[i, i_list, :] = q_reshaped[i]
            self.state.waste_q[i, i_list, :] = waste_q_reshaped[i]
        
        self.state.spot_q = torch.clamp(actions['spot_q'], 0.0, 100.0)
        
        seller_reward = self.compute_seller_reward()
        buyer_reward = self.compute_buyer_reward()
        
        trans_state = self.get_seller_state()
        return trans_state, buyer_reward, seller_reward
    
    def step_trans(self, actions_trans):
        keys_lengths = {'tx_u': self.num_commodities, 'eco_u': self.num_commodities}
        actions = self.action_conversion(actions_trans, keys_lengths)
        
        self.state.tx_u = torch.clamp(actions['tx_u'], 0.0, 50.0)
        self.state.eco_u = torch.clamp(actions['eco_u'], 0.0, 50.0)
        
        self.state.wastewater = torch.sum(self.state.tx_u)
        
        self.state.inv = torch.clamp(
            self.state.inv - self.state.tx_u - self.state.eco_u, 
            min=0.0
        )
        self.state.waste_inv = (1.0 - self.delta) * (self.state.waste_inv + self.state.tx_u)
        
        trans_reward = self.compute_trans_reward()
        
        self.t += 1
        done = (self.t >= self.episode_length)
        
        seller_state = self.get_seller_state()
        return seller_state, trans_reward, done
    
    def compute_seller_reward(self):
        revenue_regular = torch.sum(self.state.price * self.state.actual_d, dim=(1, 2))
        revenue_waste = torch.sum(self.state.waste_price * self.state.waste_actual_d, dim=(1, 2))
        reward = revenue_regular + revenue_waste
        return reward * self.RWD_SCALE
    
    def compute_buyer_reward(self):
        cost_regular = torch.sum(self.state.price.unsqueeze(0) * self.state.actual_d, dim=(1, 2))
        cost_waste = torch.sum(self.state.waste_price.unsqueeze(0) * self.state.waste_actual_d, dim=(1, 2))
        spot_cost = torch.sum(self.state.spot_price.unsqueeze(0) * self.state.spot_q, dim=1)
        
        reward = -(cost_regular + cost_waste + spot_cost)
        return reward * self.RWD_SCALE
    
    def compute_trans_reward(self):
        uc_cost = torch.sum(self.state.eco_u * 0.5, dim=1)
        tx_cost = torch.sum(self.state.tx_u * 0.5, dim=1)
        reward = uc_cost - tx_cost
        return reward * self.RWD_SCALE
    
    def collect_trajectory_metrics(self):
        env_metric = torch.sum(self.state.wastewater) + 0.5 * torch.sum(self.state.waste_q)
        
        total_revenue = torch.sum(self.state.price * self.state.actual_d) + \
                       torch.sum(self.state.waste_price * self.state.waste_actual_d)
        total_cost = torch.sum(self.state.spot_price * self.state.spot_q)
        econ_metric = -(total_revenue - total_cost)
        
        agent_utilities = torch.sum(self.state.actual_d, dim=(1, 2))
        equity_metric = torch.var(agent_utilities)
        
        return env_metric, econ_metric, equity_metric
    
    def get_mechanism_state(self):
        state_flat = torch.cat([
            self.state.spot_price,
            self.state.inv.mean(dim=0),
            self.state.waste_inv.mean(dim=0),
            self.state.price.mean(dim=0),
            self.state.waste_price.mean(dim=0)
        ])
        return state_flat.detach()


class BilevelUpperLevelOptimizer:
    def __init__(self, mechanism_network, num_commodities, lr=1e-4):
        self.mechanism_network = mechanism_network
        self.num_commodities = num_commodities
        self.optimizer = optim.Adam(mechanism_network.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def compute_upper_level_loss(self, env_metric, econ_metric, equity_metric,
                                weights={'env': 0.4, 'econ': 0.35, 'equity': 0.25}):
        loss = (weights['env'] * env_metric + 
               weights['econ'] * econ_metric + 
               weights['equity'] * equity_metric)
        return loss
    
    def step(self, metrics_dict, weights={'env': 0.4, 'econ': 0.35, 'equity': 0.25}):
        env_metrics = torch.stack(metrics_dict['env'])
        econ_metrics = torch.stack(metrics_dict['econ'])
        equity_metrics = torch.stack(metrics_dict['equity'])
        
        loss = self.compute_upper_level_loss(
            env_metrics.mean(),
            econ_metrics.mean(),
            equity_metrics.mean(),
            weights
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mechanism_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.detach().item()
