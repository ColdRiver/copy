import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import grad


class MechanismPolicyNetwork(nn.Module):
    def __init__(self, state_dim, commodity_dim, hidden_dim=256):
        super().__init__()
        self.commodity_dim = commodity_dim
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.subsidy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, commodity_dim),
            nn.Tanh()
        )
        
        self.tax_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, commodity_dim),
            nn.Sigmoid()
        )
    
    def forward(self, state):
        encoded = self.shared_encoder(state)
        subsidies = self.subsidy_head(encoded)
        taxes = self.tax_head(encoded)
        return subsidies, taxes


class LowerLevelPolicies(nn.Module):
    def __init__(self, num_agents, num_commodities, state_dim, hidden_dim=128):
        super().__init__()
        self.num_agents = num_agents
        self.num_commodities = num_commodities
        
        self.seller_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * num_commodities),
            nn.Softplus()
        )
        
        self.buyer_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * num_commodities),
            nn.Softplus()
        )
        
        self.trans_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * num_commodities),
            nn.Softplus()
        )
    
    def forward(self, state, stage):
        if stage == 'sell':
            return self.seller_policy(state)
        elif stage == 'buy':
            return self.buyer_policy(state)
        elif stage == 'trans':
            return self.trans_policy(state)


class DifferentiableMarketEnvironment:
    def __init__(self, mechanism_policy, lower_level_policies, num_agents=3, 
                 num_commodities=12, device='cpu'):
        
        self.mechanism_policy = mechanism_policy
        self.lower_level_policies = lower_level_policies
        self.num_agents = num_agents
        self.num_commodities = num_commodities
        self.device = device
        
        self.UC = 0.5
        self.TX_P = 0.5
        self.delta = 0.5
        self.LAMBDA = 0.5
        self.RWD_SCALE = 1e-9
        self.INIT_INV = 100.0
        
        self.reset_state()
    
    def reset_state(self):
        self.price = torch.zeros(self.num_agents, self.num_commodities, 
                                dtype=torch.float32, device=self.device)
        self.waste_price = torch.zeros(self.num_agents, self.num_commodities,
                                      dtype=torch.float32, device=self.device)
        self.spot_price = torch.ones(self.num_commodities, 
                                    dtype=torch.float32, device=self.device) * 0.5
        
        self.inv = torch.ones(self.num_agents, self.num_commodities,
                            dtype=torch.float32, device=self.device) * self.INIT_INV
        self.waste_inv = torch.ones(self.num_agents, self.num_commodities,
                                   dtype=torch.float32, device=self.device) * 50.0
        
        self.q = torch.zeros(self.num_agents, self.num_agents, self.num_commodities,
                            dtype=torch.float32, device=self.device)
        self.waste_q = torch.zeros(self.num_agents, self.num_agents, self.num_commodities,
                                  dtype=torch.float32, device=self.device)
        self.spot_q = torch.zeros(self.num_agents, self.num_commodities,
                                 dtype=torch.float32, device=self.device)
        
        self.actual_d = torch.zeros(self.num_agents, self.num_agents, self.num_commodities,
                                   dtype=torch.float32, device=self.device)
        self.waste_actual_d = torch.zeros(self.num_agents, self.num_agents, self.num_commodities,
                                         dtype=torch.float32, device=self.device)
        
        self.eco_u = torch.zeros(self.num_agents, self.num_commodities,
                               dtype=torch.float32, device=self.device)
        self.tx_u = torch.zeros(self.num_agents, self.num_commodities,
                              dtype=torch.float32, device=self.device)
        
        self.wastewater = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        
        self.subsidies = None
        self.taxes = None
        
        self.trajectory_rewards = {'seller': [], 'buyer': [], 'trans': []}
        self.trajectory_metrics = {'env': [], 'econ': [], 'equity': []}
    
    def get_mechanism_state(self):
        state = torch.cat([
            self.spot_price,
            self.price.mean(dim=0),
            self.waste_price.mean(dim=0),
            self.inv.mean(dim=0),
            self.waste_inv.mean(dim=0)
        ])
        return state
    
    def compute_mechanisms(self, state):
        subsidies, taxes = self.mechanism_policy(state)
        self.subsidies = subsidies
        self.taxes = taxes
        return subsidies, taxes
    
    def get_seller_state(self):
        state = torch.cat([
            self.spot_price,
            self.price.view(-1),
            self.waste_price.view(-1),
            self.inv.view(-1),
            self.waste_inv.view(-1),
            self.q.view(-1),
            self.waste_q.view(-1),
        ])
        return state.unsqueeze(0).expand(self.num_agents, -1)
    
    def step_sell(self, seller_state):
        seller_actions = self.lower_level_policies(seller_state, 'sell')
        seller_actions = torch.clamp(seller_actions, min=0.01, max=100.0)
        
        price = seller_actions[:, :self.num_commodities]
        waste_price = seller_actions[:, self.num_commodities:]
        
        if self.subsidies is not None and self.taxes is not None:
            price = price * (1.0 + self.taxes.unsqueeze(0))
            waste_price = waste_price * (1.0 - self.subsidies.unsqueeze(0))
        
        self.price = torch.clamp(price, min=0.0, max=1000.0)
        self.waste_price = torch.clamp(waste_price, min=0.0, max=1000.0)
        
        buyer_state = self.get_seller_state()
        return buyer_state
    
    def step_buy(self, buyer_state):
        buyer_actions = self.lower_level_policies(buyer_state, 'buy')
        buyer_actions = torch.clamp(buyer_actions, min=0.01, max=100.0)
        
        q_dim = (self.num_agents - 1) * self.num_commodities
        
        q = buyer_actions[:, :q_dim]
        waste_q = buyer_actions[:, q_dim:2*q_dim]
        spot_q = buyer_actions[:, 2*q_dim:]
        
        q = q.reshape(self.num_agents, self.num_agents - 1, self.num_commodities)
        waste_q = waste_q.reshape(self.num_agents, self.num_agents - 1, self.num_commodities)
        
        self.q.zero_()
        self.waste_q.zero_()
        
        for i in range(self.num_agents):
            i_list = [j for j in range(self.num_agents) if j != i]
            self.q[i, i_list, :] = q[i]
            self.waste_q[i, i_list, :] = waste_q[i]
        
        self.spot_q = torch.clamp(spot_q, min=0.0, max=100.0)
        
        self.actual_d = self.q.clone().detach()
        self.waste_actual_d = self.waste_q.clone().detach()
        
        seller_reward = self.compute_seller_reward()
        buyer_reward = self.compute_buyer_reward()
        
        self.trajectory_rewards['seller'].append(seller_reward.detach().mean().item())
        self.trajectory_rewards['buyer'].append(buyer_reward.detach().mean().item())
        
        trans_state = self.get_seller_state()
        return trans_state, buyer_reward, seller_reward
    
    def step_trans(self, trans_state):
        trans_actions = self.lower_level_policies(trans_state, 'trans')
        trans_actions = torch.clamp(trans_actions, min=0.01, max=100.0)
        
        eco_u = trans_actions[:, :self.num_commodities]
        tx_u = trans_actions[:, self.num_commodities:]
        
        self.eco_u = torch.clamp(eco_u, min=0.0, max=50.0)
        self.tx_u = torch.clamp(tx_u, min=0.0, max=50.0)
        
        self.wastewater = torch.sum(self.tx_u)
        
        self.inv = torch.clamp(self.inv - self.eco_u - self.tx_u, min=0.0)
        self.waste_inv = (1.0 - self.delta) * (self.waste_inv + self.tx_u)
        
        trans_reward = self.compute_trans_reward()
        self.trajectory_rewards['trans'].append(trans_reward.detach().mean().item())
        
        seller_state = self.get_seller_state()
        return seller_state, trans_reward
    
    def compute_seller_reward(self):
        revenue = torch.sum(self.price.unsqueeze(0) * self.actual_d, dim=(1, 2))
        revenue += torch.sum(self.waste_price.unsqueeze(0) * self.waste_actual_d, dim=(1, 2))
        return revenue * self.RWD_SCALE
    
    def compute_buyer_reward(self):
        cost = torch.sum(self.price.unsqueeze(0) * self.actual_d, dim=(1, 2))
        cost += torch.sum(self.waste_price.unsqueeze(0) * self.waste_actual_d, dim=(1, 2))
        cost += torch.sum(self.spot_price.unsqueeze(0) * self.spot_q, dim=1)
        return -cost * self.RWD_SCALE
    
    def compute_trans_reward(self):
        reward = torch.sum(self.eco_u * 0.5, dim=1) - torch.sum(self.tx_u * 0.5, dim=1)
        return reward * self.RWD_SCALE
    
    def compute_metrics(self):
        env_metric = self.wastewater + 0.5 * torch.sum(self.waste_q)
        
        total_revenue = torch.sum(self.price * self.actual_d) + torch.sum(self.waste_price * self.waste_actual_d)
        total_cost = torch.sum(self.spot_price * self.spot_q)
        econ_metric = -(total_revenue - total_cost)
        
        agent_utilities = torch.sum(self.actual_d, dim=(1, 2))
        equity_metric = torch.var(agent_utilities)
        
        self.trajectory_metrics['env'].append(env_metric)
        self.trajectory_metrics['econ'].append(econ_metric)
        self.trajectory_metrics['equity'].append(equity_metric)
        
        return env_metric, econ_metric, equity_metric
    
    def rollout_episode(self):
        self.reset_state()
        
        mechanism_state = self.get_mechanism_state()
        self.compute_mechanisms(mechanism_state)
        
        seller_state = self.get_seller_state()
        
        for step in range(100):
            buyer_state = self.step_sell(seller_state)
            trans_state, buyer_reward, seller_reward = self.step_buy(buyer_state)
            seller_state, trans_reward = self.step_trans(trans_state)
            env_m, econ_m, equity_m = self.compute_metrics()
        
        avg_env = torch.stack(self.trajectory_metrics['env']).mean()
        avg_econ = torch.stack(self.trajectory_metrics['econ']).mean()
        avg_equity = torch.stack(self.trajectory_metrics['equity']).mean()
        
        return avg_env, avg_econ, avg_equity


class BilevelOptimizationFramework:
    def __init__(self, mechanism_policy, lower_level_policies, num_agents=3, 
                 num_commodities=12, outer_lr=1e-4, inner_lr=1e-3, device='cpu'):
        
        self.mechanism_policy = mechanism_policy.to(device)
        self.lower_level_policies = lower_level_policies.to(device)
        self.env = DifferentiableMarketEnvironment(
            mechanism_policy, lower_level_policies, num_agents, num_commodities, device
        )
        
        self.outer_optimizer = optim.Adam(mechanism_policy.parameters(), lr=outer_lr)
        self.inner_optimizer = optim.Adam(lower_level_policies.parameters(), lr=inner_lr)
        
        self.device = device
        self.weights = {'env': 0.4, 'econ': 0.35, 'equity': 0.25}
    
    def compute_upper_level_loss(self, env_m, econ_m, equity_m):
        loss = (self.weights['env'] * env_m + 
               self.weights['econ'] * econ_m + 
               self.weights['equity'] * equity_m)
        return loss
    
    def bilevel_step(self):
        inner_steps = 5
        for _ in range(inner_steps):
            self.inner_optimizer.zero_grad()
            env_m, econ_m, equity_m = self.env.rollout_episode()
            lower_loss = torch.sum(env_m + econ_m + equity_m)
            lower_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.lower_level_policies.parameters(), max_norm=1.0)
            self.inner_optimizer.step()
        
        self.outer_optimizer.zero_grad()
        env_m, econ_m, equity_m = self.env.rollout_episode()
        upper_loss = self.compute_upper_level_loss(env_m, econ_m, equity_m)
        upper_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mechanism_policy.parameters(), max_norm=1.0)
        self.outer_optimizer.step()
        
        return upper_loss.detach().item(), env_m.item(), econ_m.item(), equity_m.item()
