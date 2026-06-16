import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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


class AgentPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus()
        )
    
    def forward(self, state):
        return self.net(state)


class SellerAgent(nn.Module):
    def __init__(self, state_dim, commodity_dim, hidden_dim=128):
        super().__init__()
        self.policy = AgentPolicyNetwork(state_dim, 2 * commodity_dim, hidden_dim)
        self.commodity_dim = commodity_dim
    
    def forward(self, state):
        actions = self.policy(state)
        price = torch.clamp(actions[:, :self.commodity_dim], 0.01, 100.0)
        waste_price = torch.clamp(actions[:, self.commodity_dim:], 0.01, 100.0)
        return price, waste_price


class BuyerAgent(nn.Module):
    def __init__(self, state_dim, commodity_dim, num_agents, hidden_dim=128):
        super().__init__()
        self.policy = AgentPolicyNetwork(
            state_dim, 
            3 * (num_agents - 1) * commodity_dim + commodity_dim,
            hidden_dim
        )
        self.commodity_dim = commodity_dim
        self.num_agents = num_agents
    
    def forward(self, state):
        actions = self.policy(state)
        q_dim = (self.num_agents - 1) * self.commodity_dim
        
        q = torch.clamp(actions[:, :q_dim], 0.01, 100.0)
        waste_q = torch.clamp(actions[:, q_dim:2*q_dim], 0.01, 100.0)
        spot_q = torch.clamp(actions[:, 2*q_dim:], 0.01, 100.0)
        
        return q, waste_q, spot_q


class TransformationAgent(nn.Module):
    def __init__(self, state_dim, commodity_dim, hidden_dim=128):
        super().__init__()
        self.policy = AgentPolicyNetwork(state_dim, 2 * commodity_dim, hidden_dim)
        self.commodity_dim = commodity_dim
    
    def forward(self, state):
        actions = self.policy(state)
        eco_u = torch.clamp(actions[:, :self.commodity_dim], 0.01, 50.0)
        tx_u = torch.clamp(actions[:, self.commodity_dim:], 0.01, 50.0)
        return eco_u, tx_u


class DifferentiableBilevelEnvironment:
    def __init__(self, mechanism_policy, seller_agents, buyer_agents, trans_agents,
                 num_agents=3, num_commodities=12, device='cpu'):
        
        self.mechanism_policy = mechanism_policy
        self.seller_agents = seller_agents
        self.buyer_agents = buyer_agents
        self.trans_agents = trans_agents
        
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
    
    def compute_mechanisms(self, state):
        subsidies, taxes = self.mechanism_policy(state)
        self.subsidies = subsidies
        self.taxes = taxes
        return subsidies, taxes
    
    def step_sell(self, seller_state):
        prices = []
        waste_prices = []
        
        for i in range(self.num_agents):
            price, waste_price = self.seller_agents[i](seller_state[i:i+1])
            
            if self.subsidies is not None and self.taxes is not None:
                price = price * (1.0 + self.taxes.unsqueeze(0))
                waste_price = waste_price * (1.0 - self.subsidies.unsqueeze(0))
            
            price = torch.clamp(price, 0.0, 1000.0)
            waste_price = torch.clamp(waste_price, 0.0, 1000.0)
            
            prices.append(price)
            waste_prices.append(waste_price)
        
        self.price = torch.cat(prices, dim=0)
        self.waste_price = torch.cat(waste_prices, dim=0)
        
        buyer_state = self.get_seller_state()
        return buyer_state
    
    def step_buy(self, buyer_state):
        qs = []
        waste_qs = []
        spot_qs = []
        
        for i in range(self.num_agents):
            q, waste_q, spot_q = self.buyer_agents[i](buyer_state[i:i+1])
            qs.append(q)
            waste_qs.append(waste_q)
            spot_qs.append(spot_q)
        
        q_batch = torch.cat(qs, dim=0)
        waste_q_batch = torch.cat(waste_qs, dim=0)
        spot_q_batch = torch.cat(spot_qs, dim=0)
        
        q_dim = (self.num_agents - 1) * self.num_commodities
        q_batch = q_batch.reshape(self.num_agents, self.num_agents - 1, self.num_commodities)
        waste_q_batch = waste_q_batch.reshape(self.num_agents, self.num_agents - 1, self.num_commodities)
        
        self.q.zero_()
        self.waste_q.zero_()
        
        for i in range(self.num_agents):
            i_list = [j for j in range(self.num_agents) if j != i]
            self.q[i, i_list, :] = q_batch[i]
            self.waste_q[i, i_list, :] = waste_q_batch[i]
        
        self.spot_q = torch.clamp(spot_q_batch, 0.0, 100.0)
        
        self.actual_d = self.q
        self.waste_actual_d = self.waste_q
        
        seller_reward = self.compute_seller_reward()
        buyer_reward = self.compute_buyer_reward()
        
        trans_state = self.get_seller_state()
        return trans_state, buyer_reward, seller_reward
    
    def step_trans(self, trans_state):
        eco_us = []
        tx_us = []
        
        for i in range(self.num_agents):
            eco_u, tx_u = self.trans_agents[i](trans_state[i:i+1])
            eco_us.append(eco_u)
            tx_us.append(tx_u)
        
        self.eco_u = torch.cat(eco_us, dim=0)
        self.tx_u = torch.cat(tx_us, dim=0)
        
        self.wastewater = torch.sum(self.tx_u)
        
        self.inv = torch.clamp(self.inv - self.eco_u - self.tx_u, min=0.0)
        self.waste_inv = (1.0 - self.delta) * (self.waste_inv + self.tx_u)
        
        trans_reward = self.compute_trans_reward()
        
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
        
        self.trajectory_metrics['env'].append(env_metric.detach())
        self.trajectory_metrics['econ'].append(econ_metric.detach())
        self.trajectory_metrics['equity'].append(equity_metric.detach())
        
        return env_metric, econ_metric, equity_metric
    
    def rollout_episode(self, episode_length=100):
        self.reset_state()
        
        mechanism_state = self.get_mechanism_state()
        self.compute_mechanisms(mechanism_state)
        
        seller_state = self.get_seller_state()
        
        total_rewards = {'seller': [], 'buyer': [], 'trans': []}
        
        for step in range(episode_length):
            buyer_state = self.step_sell(seller_state)
            trans_state, buyer_reward, seller_reward = self.step_buy(buyer_state)
            seller_state, trans_reward = self.step_trans(trans_state)
            
            total_rewards['seller'].append(seller_reward.detach().mean())
            total_rewards['buyer'].append(buyer_reward.detach().mean())
            total_rewards['trans'].append(trans_reward.detach().mean())
            
            self.compute_metrics()
        
        avg_env = torch.stack(self.trajectory_metrics['env']).mean()
        avg_econ = torch.stack(self.trajectory_metrics['econ']).mean()
        avg_equity = torch.stack(self.trajectory_metrics['equity']).mean()
        
        return avg_env, avg_econ, avg_equity, total_rewards


class BilevelOptimizer:
    def __init__(self, mechanism_policy, seller_agents, buyer_agents, trans_agents,
                 num_agents=3, num_commodities=12, outer_lr=1e-4, inner_lr=1e-3,
                 device='cpu'):
        
        self.mechanism_policy = mechanism_policy.to(device)
        self.seller_agents = [agent.to(device) for agent in seller_agents]
        self.buyer_agents = [agent.to(device) for agent in buyer_agents]
        self.trans_agents = [agent.to(device) for agent in trans_agents]
        
        self.env = DifferentiableBilevelEnvironment(
            mechanism_policy, seller_agents, buyer_agents, trans_agents,
            num_agents, num_commodities, device
        )
        
        self.outer_optimizer = optim.Adam(mechanism_policy.parameters(), lr=outer_lr)
        
        self.inner_optimizers = {
            'seller': [optim.Adam(agent.parameters(), lr=inner_lr) for agent in seller_agents],
            'buyer': [optim.Adam(agent.parameters(), lr=inner_lr) for agent in buyer_agents],
            'trans': [optim.Adam(agent.parameters(), lr=inner_lr) for agent in trans_agents]
        }
        
        self.device = device
        self.weights = {'env': 0.4, 'econ': 0.35, 'equity': 0.25}
    
    def compute_upper_level_loss(self, env_m, econ_m, equity_m):
        loss = (self.weights['env'] * env_m + 
               self.weights['econ'] * econ_m + 
               self.weights['equity'] * equity_m)
        return loss
    
    def inner_loop_step(self, num_rollouts=3):
        inner_loss_total = 0.0
        
        for _ in range(num_rollouts):
            env_m, econ_m, equity_m, rewards = self.env.rollout_episode()
            
            lower_loss = -(torch.stack(rewards['seller']).mean() +
                          torch.stack(rewards['buyer']).mean() +
                          torch.stack(rewards['trans']).mean())
            
            for optimizer_list in self.inner_optimizers.values():
                for optimizer in optimizer_list:
                    optimizer.zero_grad()
            
            lower_loss.backward()
            
            for optimizer_list in self.inner_optimizers.values():
                for optimizer in optimizer_list:
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)
                    optimizer.step()
            
            inner_loss_total += lower_loss.detach().item()
        
        return inner_loss_total / num_rollouts
    
    def outer_loop_step(self):
        env_m, econ_m, equity_m, rewards = self.env.rollout_episode()
        
        upper_loss = self.compute_upper_level_loss(env_m, econ_m, equity_m)
        
        self.outer_optimizer.zero_grad()
        upper_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mechanism_policy.parameters(), max_norm=1.0)
        self.outer_optimizer.step()
        
        return upper_loss.detach().item(), env_m.item(), econ_m.item(), equity_m.item()
    
    def bilevel_iteration(self, inner_steps=3):
        inner_loss = self.inner_loop_step(inner_steps)
        outer_loss, env_m, econ_m, equity_m = self.outer_loop_step()
        
        return {
            'outer_loss': outer_loss,
            'inner_loss': inner_loss,
            'env_metric': env_m,
            'econ_metric': econ_m,
            'equity_metric': equity_m
        }
