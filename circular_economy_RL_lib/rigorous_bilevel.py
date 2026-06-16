import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from collections import defaultdict


class UnrolledBilevelSimulator(nn.Module):
    
    def __init__(self, mechanism_net, num_agents=3, num_commodities=12, 
                 episode_steps=100, device='cpu'):
        super().__init__()
        self.mechanism_net = mechanism_net
        self.num_agents = num_agents
        self.num_commodities = num_commodities
        self.episode_steps = episode_steps
        self.device = device
        
        self.register_buffer('init_inv', torch.full((num_agents, num_commodities), 100.0))
        self.register_buffer('init_waste_inv', torch.full((num_agents, num_commodities), 50.0))
        self.register_buffer('spot_price_init', torch.full((num_commodities,), 0.5))
        
        self.delta = 0.5
        self.rwd_scale = 1e-9
        self.UC = 0.5
        self.TX_P = 0.5
    
    def forward(self, agent_actions_sequence, mechanism_state):
        batch_size = mechanism_state.shape[0] if mechanism_state.dim() > 1 else 1
        
        subsidies, taxes = self.mechanism_net(mechanism_state)
        
        prices_sequence = []
        waste_prices_sequence = []
        inventories_sequence = []
        waste_inventories_sequence = []
        actual_demands_sequence = []
        waste_demands_sequence = []
        metrics_sequence = []
        rewards_sequence = {'seller': [], 'buyer': [], 'trans': []}
        
        price = torch.zeros(self.num_agents, self.num_commodities, device=self.device)
        waste_price = torch.zeros(self.num_agents, self.num_commodities, device=self.device)
        inv = self.init_inv.clone()
        waste_inv = self.init_waste_inv.clone()
        
        for step in range(self.episode_steps):
            seller_actions = agent_actions_sequence['seller'][step]
            buyer_actions = agent_actions_sequence['buyer'][step]
            trans_actions = agent_actions_sequence['trans'][step]
            
            price_raw = seller_actions[:, :self.num_commodities]
            waste_price_raw = seller_actions[:, self.num_commodities:]
            
            price = torch.clamp(price_raw * (1.0 + taxes), 0.0, 1000.0)
            waste_price = torch.clamp(waste_price_raw * (1.0 - subsidies), 0.0, 1000.0)
            
            prices_sequence.append(price)
            waste_prices_sequence.append(waste_price)
            
            q_dim = (self.num_agents - 1) * self.num_commodities
            q = buyer_actions[:, :q_dim].reshape(self.num_agents, self.num_agents - 1, self.num_commodities)
            waste_q = buyer_actions[:, q_dim:2*q_dim].reshape(self.num_agents, self.num_agents - 1, self.num_commodities)
            spot_q = buyer_actions[:, 2*q_dim:]
            
            full_q = torch.zeros(self.num_agents, self.num_agents, self.num_commodities, device=self.device)
            full_waste_q = torch.zeros(self.num_agents, self.num_agents, self.num_commodities, device=self.device)
            
            for i in range(self.num_agents):
                i_list = [j for j in range(self.num_agents) if j != i]
                full_q[i, i_list, :] = q[i]
                full_waste_q[i, i_list, :] = waste_q[i]
            
            actual_d = full_q
            waste_actual_d = full_waste_q
            
            actual_demands_sequence.append(actual_d)
            waste_demands_sequence.append(waste_actual_d)
            
            seller_rev = torch.sum(price.unsqueeze(0) * actual_d, dim=(1, 2))
            seller_rev_waste = torch.sum(waste_price.unsqueeze(0) * waste_actual_d, dim=(1, 2))
            seller_reward = (seller_rev + seller_rev_waste) * self.rwd_scale
            rewards_sequence['seller'].append(seller_reward.mean())
            
            buyer_cost = torch.sum(price.unsqueeze(0) * actual_d, dim=(1, 2))
            buyer_cost_waste = torch.sum(waste_price.unsqueeze(0) * waste_actual_d, dim=(1, 2))
            buyer_cost_spot = torch.sum(self.spot_price_init.unsqueeze(0) * spot_q, dim=1)
            buyer_reward = -(buyer_cost + buyer_cost_waste + buyer_cost_spot) * self.rwd_scale
            rewards_sequence['buyer'].append(buyer_reward.mean())
            
            eco_u = trans_actions[:, :self.num_commodities]
            tx_u = trans_actions[:, self.num_commodities:]
            
            inv = torch.clamp(inv - eco_u - tx_u, min=0.0)
            waste_inv = (1.0 - self.delta) * (waste_inv + tx_u)
            
            trans_reward = torch.sum(eco_u * self.UC, dim=1) - torch.sum(tx_u * self.TX_P, dim=1)
            trans_reward = trans_reward * self.rwd_scale
            rewards_sequence['trans'].append(trans_reward.mean())
            
            inventories_sequence.append(inv)
            waste_inventories_sequence.append(waste_inv)
            
            wastewater = torch.sum(tx_u)
            env_metric = wastewater + 0.5 * torch.sum(waste_q)
            total_revenue = torch.sum(price * actual_d) + torch.sum(waste_price * waste_actual_d)
            total_cost = torch.sum(self.spot_price_init * spot_q)
            econ_metric = -(total_revenue - total_cost)
            agent_utilities = torch.sum(actual_d, dim=(1, 2))
            equity_metric = torch.var(agent_utilities) if self.num_agents > 1 else torch.tensor(0.0, device=self.device)
            
            metrics_sequence.append({
                'env': env_metric,
                'econ': econ_metric,
                'equity': equity_metric
            })
        
        avg_env_metric = torch.stack([m['env'] for m in metrics_sequence]).mean()
        avg_econ_metric = torch.stack([m['econ'] for m in metrics_sequence]).mean()
        avg_equity_metric = torch.stack([m['equity'] for m in metrics_sequence]).mean()
        
        avg_seller_reward = torch.stack(rewards_sequence['seller']).mean()
        avg_buyer_reward = torch.stack(rewards_sequence['buyer']).mean()
        avg_trans_reward = torch.stack(rewards_sequence['trans']).mean()
        
        return {
            'metrics': {
                'env': avg_env_metric,
                'econ': avg_econ_metric,
                'equity': avg_equity_metric
            },
            'rewards': {
                'seller': avg_seller_reward,
                'buyer': avg_buyer_reward,
                'trans': avg_trans_reward
            },
            'trajectories': {
                'prices': torch.stack(prices_sequence),
                'waste_prices': torch.stack(waste_prices_sequence),
                'inventories': torch.stack(inventories_sequence),
                'waste_inventories': torch.stack(waste_inventories_sequence),
                'actual_demands': torch.stack(actual_demands_sequence),
                'waste_demands': torch.stack(waste_demands_sequence)
            },
            'mechanisms': {
                'subsidies': subsidies,
                'taxes': taxes
            }
        }


class MechanismNetwork(nn.Module):
    
    def __init__(self, state_dim, commodity_dim, hidden_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.subsidy_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, commodity_dim)
        )
        
        self.tax_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, commodity_dim)
        )
        
        self.subsidy_scale = nn.Parameter(torch.tensor(0.5))
        self.tax_scale = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, state):
        encoded = self.encoder(state)
        
        subsidy_raw = self.subsidy_decoder(encoded)
        tax_raw = self.tax_decoder(encoded)
        
        subsidies = torch.tanh(subsidy_raw) * self.subsidy_scale
        taxes = torch.sigmoid(tax_raw) * self.tax_scale
        
        return subsidies, taxes


class AgentPolicyNetwork(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.action_scale = nn.Parameter(torch.ones(action_dim))
    
    def forward(self, state):
        raw_action = self.net(state)
        action = torch.sigmoid(raw_action) * self.action_scale
        return action


class MultiAgentPolicies(nn.Module):
    
    def __init__(self, num_agents, num_commodities, state_dim, hidden_dim=128, device='cpu'):
        super().__init__()
        self.num_agents = num_agents
        self.num_commodities = num_commodities
        self.device = device
        
        self.seller_policies = nn.ModuleList([
            AgentPolicyNetwork(state_dim, 2 * num_commodities, hidden_dim)
            for _ in range(num_agents)
        ])
        
        q_action_dim = (num_agents - 1) * num_commodities
        self.buyer_policies = nn.ModuleList([
            AgentPolicyNetwork(state_dim, q_action_dim * 2 + num_commodities, hidden_dim)
            for _ in range(num_agents)
        ])
        
        self.trans_policies = nn.ModuleList([
            AgentPolicyNetwork(state_dim, 2 * num_commodities, hidden_dim)
            for _ in range(num_agents)
        ])
    
    def get_seller_actions(self, state):
        actions = []
        for i, policy in enumerate(self.seller_policies):
            action = policy(state[i:i+1])
            actions.append(action)
        return torch.cat(actions, dim=0)
    
    def get_buyer_actions(self, state):
        actions = []
        for i, policy in enumerate(self.buyer_policies):
            action = policy(state[i:i+1])
            actions.append(action)
        return torch.cat(actions, dim=0)
    
    def get_trans_actions(self, state):
        actions = []
        for i, policy in enumerate(self.trans_policies):
            action = policy(state[i:i+1])
            actions.append(action)
        return torch.cat(actions, dim=0)


class RigourousBilevelOptimizer:
    
    def __init__(self, mechanism_net, agent_policies, num_agents=3, num_commodities=12,
                 outer_lr=1e-4, inner_lr=1e-3, device='cpu'):
        
        self.mechanism_net = mechanism_net.to(device)
        self.agent_policies = agent_policies.to(device)
        self.unrolled_sim = UnrolledBilevelSimulator(
            mechanism_net, num_agents, num_commodities, device=device
        ).to(device)
        
        self.outer_optimizer = optim.Adam(mechanism_net.parameters(), lr=outer_lr)
        self.inner_optimizer = optim.Adam(agent_policies.parameters(), lr=inner_lr)
        
        self.device = device
        self.num_agents = num_agents
        self.num_commodities = num_commodities
        
        self.upper_weights = torch.tensor(
            [0.4, 0.35, 0.25], device=device
        )
    
    def get_mechanism_state(self, episode_length=100):
        avg_price = torch.ones(self.num_commodities, device=self.device) * 0.5
        avg_waste_price = torch.ones(self.num_commodities, device=self.device) * 0.3
        avg_inv = torch.ones(self.num_commodities, device=self.device) * 100.0
        avg_waste_inv = torch.ones(self.num_commodities, device=self.device) * 50.0
        spot_price = torch.ones(self.num_commodities, device=self.device) * 0.5
        
        state = torch.cat([spot_price, avg_price, avg_waste_price, avg_inv, avg_waste_inv])
        return state
    
    def generate_agent_actions(self, episode_length=100):
        state_dim = 5 * self.num_commodities
        batch_state = torch.randn(self.num_agents, state_dim, device=self.device)
        
        seller_actions_seq = []
        buyer_actions_seq = []
        trans_actions_seq = []
        
        for step in range(episode_length):
            seller_acts = self.agent_policies.get_seller_actions(batch_state)
            buyer_acts = self.agent_policies.get_buyer_actions(batch_state)
            trans_acts = self.agent_policies.get_trans_actions(batch_state)
            
            seller_actions_seq.append(seller_acts)
            buyer_actions_seq.append(buyer_acts)
            trans_actions_seq.append(trans_acts)
        
        return {
            'seller': seller_actions_seq,
            'buyer': buyer_actions_seq,
            'trans': trans_actions_seq
        }
    
    def compute_upper_loss(self, metrics):
        env_m = metrics['env']
        econ_m = metrics['econ']
        equity_m = metrics['equity']
        
        loss = (0.4 * env_m + 0.35 * econ_m + 0.25 * equity_m)
        return loss
    
    def compute_lower_loss(self, rewards):
        lower_loss = -(rewards['seller'] + rewards['buyer'] + rewards['trans'])
        return lower_loss
    
    def bilevel_step(self):
        mechanism_state = self.get_mechanism_state()
        agent_actions = self.generate_agent_actions(episode_length=100)
        
        inner_rollout = self.unrolled_sim(agent_actions, mechanism_state)
        lower_loss = self.compute_lower_loss(inner_rollout['rewards'])
        
        self.inner_optimizer.zero_grad()
        lower_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.agent_policies.parameters(), max_norm=1.0)
        self.inner_optimizer.step()
        
        agent_actions_for_outer = self.generate_agent_actions(episode_length=100)
        outer_rollout = self.unrolled_sim(agent_actions_for_outer, mechanism_state)
        upper_loss = self.compute_upper_loss(outer_rollout['metrics'])
        
        self.outer_optimizer.zero_grad()
        upper_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mechanism_net.parameters(), max_norm=1.0)
        self.outer_optimizer.step()
        
        return {
            'upper_loss': upper_loss.detach().item(),
            'lower_loss': lower_loss.detach().item(),
            'env_metric': outer_rollout['metrics']['env'].detach().item(),
            'econ_metric': outer_rollout['metrics']['econ'].detach().item(),
            'equity_metric': outer_rollout['metrics']['equity'].detach().item()
        }
