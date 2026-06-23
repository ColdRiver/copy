"""
Bilevel Optimization Framework for Circular Economy
=====================================================

Upper Level (Leader): Market Authority designing incentive & pricing mechanisms
  - Objectives: Environmental sustainability, Economic efficiency, Equity
  - Variables: Subsidies (waste reduction), Taxes (pollution), Pricing signals

Lower Level (Follower): Individual agents (Sellers, Buyers, Transformers) responding to mechanisms
  - Objectives: Individual profit maximization
  - Variables: Prices, Purchase quantities, Transformation activities
  
Reference: arXiv:2503.17644 - Bilevel Reinforcement Learning framework
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List
import logging


class MechanismDesigner(nn.Module):
    """
    Upper-level optimizer: Design market mechanisms (subsidies, taxes, price signals)
    
    Mechanism variables:
    - subsidy_waste: incentive for waste reduction [0, 1] → applied to waste prices
    - tax_pollution: penalty on high transformation [0, 1] → applied to tx_u costs
    - price_signal_adjustment: dynamic price multipliers [0, 1] → applied to spot prices
    - inventory_holding_tax: penalty on holding costs → scales inventory carrying costs
    """
    
    def __init__(self, num_commodities: int, hidden_dim: int = 256, device: str = 'cpu'):
        super().__init__()
        self.num_commodities = num_commodities
        self.device = device
        
        # Shared encoder: learns aggregate market state
        self.encoder = nn.Sequential(
            nn.Linear(5 * num_commodities, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Subsidy decoder: waste incentives per commodity
        self.subsidy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_commodities),
            nn.Sigmoid()  # [0, 1]
        )
        
        # Tax decoder: pollution penalties per commodity
        self.tax_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_commodities),
            nn.Sigmoid()  # [0, 1]
        )
        
        # Price signal decoder: global price adjustment
        self.price_signal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_commodities),
            nn.Sigmoid()  # [0.5, 1.5] via scaling
        )
        
        # Inventory tax decoder: carrying cost penalties
        self.inv_tax_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_commodities),
            nn.Sigmoid()  # [0, 1]
        )
        
        # Learnable mechanism scales
        self.subsidy_scale = nn.Parameter(torch.tensor(0.3, device=device))
        self.tax_scale = nn.Parameter(torch.tensor(0.2, device=device))
        self.price_scale = nn.Parameter(torch.tensor(0.1, device=device))
        self.inv_tax_scale = nn.Parameter(torch.tensor(0.15, device=device))
    
    def forward(self, market_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            market_state: (5 * num_commodities,) aggregated market observations
                - spot_price, avg_price, avg_waste_price, avg_inv, avg_waste_inv
        
        Returns:
            mechanism dict with keys: subsidies, taxes, price_signals, inv_taxes
        """
        encoded = self.encoder(market_state)
        
        subsidies = self.subsidy_head(encoded) * self.subsidy_scale
        taxes = self.tax_head(encoded) * self.tax_scale
        price_signals = 0.5 + self.price_signal_head(encoded) * self.price_scale
        inv_taxes = self.inv_tax_head(encoded) * self.inv_tax_scale
        
        return {
            'subsidies': subsidies,      # reduces waste_price
            'taxes': taxes,               # increases tx_u cost
            'price_signals': price_signals,  # scales spot_price
            'inv_taxes': inv_taxes        # scales carrying cost
        }


class LowerLevelAgent(nn.Module):
    """
    Lower-level policy: Agent responds to mechanisms by optimizing own utility
    
    Agents are organized by role:
    - Seller: sets prices, decides inventory carryover
    - Buyer: decides purchase quantities from different sources
    - Transformer: decides transformation vs. economic utility amounts
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, 
                 device: str = 'cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.action_scale = nn.Parameter(torch.ones(action_dim, device=device))
    
    def forward(self, state: torch.Tensor, mechanism: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            state: agent's local observation
            mechanism: mechanism parameters from upper level
        
        Returns:
            action: scaled and bounded action
        """
        raw_action = self.policy_net(state)
        action = torch.sigmoid(raw_action) * self.action_scale
        return action


class MultiAgentBilevelPolicies(nn.Module):
    """
    Manages all agent policies for lower level
    """
    
    def __init__(self, num_agents: int, num_commodities: int, state_dim: int, 
                 hidden_dim: int = 128, device: str = 'cpu'):
        super().__init__()
        self.num_agents = num_agents
        self.num_commodities = num_commodities
        self.state_dim = state_dim
        self.device = device
        
        # Seller policies: output (price, waste_price) for each commodity
        self.seller_policies = nn.ModuleList([
            LowerLevelAgent(state_dim, 2 * num_commodities, hidden_dim, device)
            for _ in range(num_agents)
        ])
        
        # Buyer policies: output quantities (q, waste_q, spot_q)
        q_action_dim = (num_agents - 1) * num_commodities
        self.buyer_policies = nn.ModuleList([
            LowerLevelAgent(state_dim, q_action_dim * 2 + num_commodities, hidden_dim, device)
            for _ in range(num_agents)
        ])
        
        # Transformer policies: output (tx_u, eco_u) for each commodity
        self.trans_policies = nn.ModuleList([
            LowerLevelAgent(state_dim, 2 * num_commodities, hidden_dim, device)
            for _ in range(num_agents)
        ])
    
    def get_actions(self, states: torch.Tensor, role: int, 
                   mechanism: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """
        role: 0=seller, 1=buyer, 2=transformer
        
        Returns: (num_agents, action_dim)
        """
        actions = []
        
        if role == 0:  # Seller
            for i, policy in enumerate(self.seller_policies):
                action = policy(states[i:i+1], mechanism)
                actions.append(action)
        elif role == 1:  # Buyer
            for i, policy in enumerate(self.buyer_policies):
                action = policy(states[i:i+1], mechanism)
                actions.append(action)
        elif role == 2:  # Transformer
            for i, policy in enumerate(self.trans_policies):
                action = policy(states[i:i+1], mechanism)
                actions.append(action)
        
        return torch.cat(actions, dim=0)


class BilevelSimulator:
    """
    Simulates the circular economy under given mechanisms
    Computes upper-level metrics and lower-level rewards
    """
    
    def __init__(self, num_agents: int = 3, num_commodities: int = 12, 
                 device: str = 'cpu'):
        super().__init__()
        self.num_agents = num_agents
        self.num_commodities = num_commodities
        self.device = device
        
        # System constants
        self.delta = 0.5  # waste decay rate
        self.rwd_scale = 1e-9
        self.UC = 0.5  # unit cost of economic utility
        self.TX_P = 0.5  # unit cost of transformation
        self.LAMBDA = 0.5  # demand satisfaction reward weight
        
        # Initial conditions
        self.init_inv = torch.full((num_agents, num_commodities), 100.0, device=device)
        self.init_waste_inv = torch.full((num_agents, num_commodities), 50.0, device=device)
        self.spot_price_init = torch.full((num_commodities,), 0.5, device=device)
    
    def simulate_step(self, mechanism: Dict[str, torch.Tensor],
                     seller_actions: torch.Tensor,
                     buyer_actions: torch.Tensor,
                     trans_actions: torch.Tensor,
                     inv: torch.Tensor,
                     waste_inv: torch.Tensor) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """
        Single timestep simulation under mechanism
        
        Returns:
            - metrics: dict with env, econ, equity scores
            - new_inv: updated inventory
            - new_waste_inv: updated waste inventory
        """
        
        # ---- SELLER STAGE ----
        price_raw = seller_actions[:, :self.num_commodities]
        waste_price_raw = seller_actions[:, self.num_commodities:]
        
        # Apply mechanism: taxes increase raw prices, subsidies reduce waste prices
        taxes = mechanism['taxes']
        subsidies = mechanism['subsidies']
        
        price = torch.clamp(price_raw * (1.0 + taxes), 0.0, 1000.0)
        waste_price = torch.clamp(waste_price_raw * (1.0 - subsidies), 0.0, 1000.0)
        
        # ---- BUYER STAGE ----
        q_dim = (self.num_agents - 1) * self.num_commodities
        q = buyer_actions[:, :q_dim].reshape(self.num_agents, self.num_agents - 1, self.num_commodities)
        waste_q = buyer_actions[:, q_dim:2*q_dim].reshape(self.num_agents, self.num_agents - 1, self.num_commodities)
        spot_q = buyer_actions[:, 2*q_dim:]
        
        # Reconstruct full quantity matrices
        full_q = torch.zeros(self.num_agents, self.num_agents, self.num_commodities, device=self.device)
        full_waste_q = torch.zeros(self.num_agents, self.num_agents, self.num_commodities, device=self.device)
        
        for i in range(self.num_agents):
            i_list = [j for j in range(self.num_agents) if j != i]
            full_q[i, i_list, :] = q[i]
            full_waste_q[i, i_list, :] = waste_q[i]
        
        actual_d = full_q
        waste_actual_d = full_waste_q
        
        # ---- REWARDS: LOWER LEVEL ----
        seller_rev = torch.sum(price.unsqueeze(0) * actual_d, dim=(1, 2))
        seller_rev_waste = torch.sum(waste_price.unsqueeze(0) * waste_actual_d, dim=(1, 2))
        seller_rewards = (seller_rev + seller_rev_waste) * self.rwd_scale
        
        buyer_cost = torch.sum(price.unsqueeze(0) * actual_d, dim=(1, 2))
        buyer_cost_waste = torch.sum(waste_price.unsqueeze(0) * waste_actual_d, dim=(1, 2))
        buyer_cost_spot = torch.sum(self.spot_price_init * spot_q, dim=1)
        buyer_rewards = -(buyer_cost + buyer_cost_waste + buyer_cost_spot) * self.rwd_scale
        
        # ---- TRANSFORMER STAGE ----
        eco_u = trans_actions[:, :self.num_commodities]
        tx_u = trans_actions[:, self.num_commodities:]
        
        # Apply mechanism: taxes increase transformation costs
        tx_cost_multiplier = 1.0 + mechanism['taxes']
        trans_cost = torch.sum(tx_u * self.TX_P * tx_cost_multiplier, dim=1)
        trans_rewards = torch.sum(eco_u * self.UC, dim=1) - trans_cost * self.rwd_scale
        
        # ---- INVENTORY UPDATES ----
        new_inv = torch.clamp(inv - eco_u - tx_u, min=0.0)
        new_waste_inv = (1.0 - self.delta) * (waste_inv + tx_u)
        
        # Apply inventory tax to carrying costs (reflected in metrics)
        inv_tax = mechanism['inv_taxes']
        
        # ---- UPPER LEVEL METRICS ----
        # Environmental: wastewater + waste inventory
        wastewater = torch.sum(tx_u)
        env_metric = wastewater + 0.5 * torch.sum(waste_q)
        
        # Economic: total surplus (maximize revenue - cost)
        total_revenue = torch.sum(price * actual_d) + torch.sum(waste_price * waste_actual_d)
        total_cost = torch.sum(self.spot_price_init * spot_q)
        econ_metric = total_revenue - total_cost
        
        # Equity: agent utility variance (lower is more equitable)
        agent_utilities = torch.sum(actual_d, dim=(1, 2))
        equity_metric = torch.var(agent_utilities) if self.num_agents > 1 else torch.tensor(0.0, device=self.device)
        
        metrics = {
            'env': env_metric,
            'econ': econ_metric,
            'equity': equity_metric
        }
        
        lower_rewards = {
            'seller': seller_rewards,
            'buyer': buyer_rewards,
            'trans': trans_rewards
        }
        
        return metrics, lower_rewards, new_inv, new_waste_inv, {
            'prices': price,
            'waste_prices': waste_price,
            'actual_demands': actual_d,
            'waste_demands': waste_actual_d
        }


class BilevelOptimizer:
    """
    Bilevel optimization trainer
    
    Upper level: Optimize mechanism parameters (subsidies, taxes, etc.)
    Lower level: Optimize agent policies given mechanism
    
    Uses implicit differentiation or unrolled optimization
    """
    
    def __init__(self, mechanism_net: MechanismDesigner,
                 agent_policies: MultiAgentBilevelPolicies,
                 num_agents: int = 3,
                 num_commodities: int = 12,
                 outer_lr: float = 1e-4,
                 inner_lr: float = 1e-3,
                 inner_steps: int = 5,
                 device: str = 'cpu'):
        
        self.mechanism_net = mechanism_net.to(device)
        self.agent_policies = agent_policies.to(device)
        self.simulator = BilevelSimulator(num_agents, num_commodities, device)
        
        self.outer_optimizer = optim.Adam(mechanism_net.parameters(), lr=outer_lr)
        self.inner_optimizer = optim.Adam(agent_policies.parameters(), lr=inner_lr)
        
        self.device = device
        self.num_agents = num_agents
        self.num_commodities = num_commodities
        self.inner_steps = inner_steps
        
        # Upper level weights: [env, econ, equity]
        self.upper_weights = torch.tensor([0.4, 0.35, 0.25], device=device)
        
        self.logger = logging.getLogger('BilevelOptimizer')
    
    def get_market_state(self, episode_length: int = 100) -> torch.Tensor:
        """
        Construct aggregated market state for mechanism designer
        """
        avg_price = torch.ones(self.num_commodities, device=self.device) * 0.5
        avg_waste_price = torch.ones(self.num_commodities, device=self.device) * 0.3
        avg_inv = torch.ones(self.num_commodities, device=self.device) * 100.0
        avg_waste_inv = torch.ones(self.num_commodities, device=self.device) * 50.0
        spot_price = torch.ones(self.num_commodities, device=self.device) * 0.5
        
        state = torch.cat([spot_price, avg_price, avg_waste_price, avg_inv, avg_waste_inv])
        return state
    
    def inner_optimization_step(self, mechanism: Dict[str, torch.Tensor],
                               market_state: torch.Tensor,
                               num_steps: int = 5):
        """
        Inner level: Optimize agent policies for fixed mechanism
        """
        total_lower_loss = 0.0
        
        for inner_step in range(num_steps):
            self.inner_optimizer.zero_grad()
            
            # Generate agent states and actions
            batch_states = torch.randn(self.num_agents, 
                                      5 * self.num_commodities, 
                                      device=self.device)
            
            seller_acts = self.agent_policies.get_actions(batch_states, role=0, mechanism=mechanism)
            buyer_acts = self.agent_policies.get_actions(batch_states, role=1, mechanism=mechanism)
            trans_acts = self.agent_policies.get_actions(batch_states, role=2, mechanism=mechanism)
            
            # Simulate one step
            inv = self.simulator.init_inv.clone()
            waste_inv = self.simulator.init_waste_inv.clone()
            
            metrics, lower_rewards, _, _, _ = self.simulator.simulate_step(
                mechanism, seller_acts, buyer_acts, trans_acts, inv, waste_inv
            )
            
            # Lower level loss: negative of sum of agent rewards
            lower_loss = -(lower_rewards['seller'].mean() + 
                          lower_rewards['buyer'].mean() + 
                          lower_rewards['trans'].mean())
            
            lower_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.agent_policies.parameters(), max_norm=1.0)
            self.inner_optimizer.step()
            
            total_lower_loss += lower_loss.detach().item()
        
        return total_lower_loss / num_steps
    
    def outer_optimization_step(self, market_state: torch.Tensor) -> Dict[str, float]:
        """
        Outer level: Optimize mechanism parameters
        
        After inner optimization converges, compute upper level loss and update mechanism
        """
        # Get mechanism
        mechanism = self.mechanism_net(market_state)
        
        # Run inner optimization
        inner_loss = self.inner_optimization_step(mechanism, market_state, self.inner_steps)
        
        # Compute upper level loss
        self.outer_optimizer.zero_grad()
        
        batch_states = torch.randn(self.num_agents, 
                                  5 * self.num_commodities, 
                                  device=self.device)
        
        seller_acts = self.agent_policies.get_actions(batch_states, role=0, mechanism=mechanism)
        buyer_acts = self.agent_policies.get_actions(batch_states, role=1, mechanism=mechanism)
        trans_acts = self.agent_policies.get_actions(batch_states, role=2, mechanism=mechanism)
        
        inv = self.simulator.init_inv.clone()
        waste_inv = self.simulator.init_waste_inv.clone()
        
        metrics, lower_rewards, _, _, _ = self.simulator.simulate_step(
            mechanism, seller_acts, buyer_acts, trans_acts, inv, waste_inv
        )
        
        # Upper level loss: weighted combination of metrics
        upper_loss = (self.upper_weights[0] * metrics['env'] +
                     self.upper_weights[1] * metrics['econ'] +
                     self.upper_weights[2] * metrics['equity'])
        
        upper_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mechanism_net.parameters(), max_norm=1.0)
        self.outer_optimizer.step()
        
        return {
            'upper_loss': upper_loss.detach().item(),
            'lower_loss': inner_loss,
            'env_metric': metrics['env'].detach().item(),
            'econ_metric': metrics['econ'].detach().item(),
            'equity_metric': metrics['equity'].detach().item(),
            'subsidies_mean': mechanism['subsidies'].mean().detach().item(),
            'taxes_mean': mechanism['taxes'].mean().detach().item()
        }
    
    def train(self, num_epochs: int = 100, log_freq: int = 10):
        """
        Main training loop
        """
        for epoch in range(num_epochs):
            market_state = self.get_market_state()
            
            results = self.outer_optimization_step(market_state)
            
            if (epoch + 1) % log_freq == 0:
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
                for key, val in results.items():
                    self.logger.info(f"  {key}: {val:.6f}")
