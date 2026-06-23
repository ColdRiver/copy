"""
Extended Bilevel Simulator
===========================

Integration bridge between bilevel mechanism design and circular economy dynamics.
Extends the base Manufacturing_Simulator to support mechanism-driven agent behavior.

Key Features:
- Accepts mechanism parameters from upper level
- Applies mechanisms to agent actions and rewards
- Computes both lower-level (agent) and upper-level (social) metrics
- Compatible with existing simulator state structures
"""

import numpy as np
import torch
from typing import Dict, Tuple, Any
from config import config, init_historical_data, SELLER, BUYER, TRANSFORM, stages
from simulator import Manufacturing_Simulator
from surrogate_models import SurrogateModel


class BilevelManufacturingSimulator(Manufacturing_Simulator):
    """
    Extended simulator that applies bilevel mechanisms to the circular economy model.
    
    Inherits from Manufacturing_Simulator but adds:
    - Mechanism parameter application
    - Upper-level metric computation
    - Support for mechanism-aware reward calculation
    """
    
    def __init__(self, mechanisms: Dict[str, np.ndarray] = None):
        """
        Initialize extended bilevel simulator
        
        Args:
            mechanisms: Dictionary with keys {subsidies, taxes, price_signals, inv_taxes}
                       None to use neutral/default mechanisms
        """
        super().__init__()
        
        self.mechanisms = mechanisms if mechanisms is not None else self._init_neutral_mechanisms()
        
        # Track metrics for upper level
        self.episode_metrics = {
            'environmental': [],
            'economic': [],
            'equity': []
        }
        
        self.episode_lower_rewards = {
            SELLER: [],
            BUYER: [],
            TRANSFORM: []
        }
    
    def _init_neutral_mechanisms(self) -> Dict[str, np.ndarray]:
        """Initialize neutral (identity) mechanisms"""
        return {
            'subsidies': np.zeros(self.num_commodities),
            'taxes': np.zeros(self.num_commodities),
            'price_signals': np.ones(self.num_commodities),
            'inv_taxes': np.zeros(self.num_commodities)
        }
    
    def set_mechanisms(self, mechanisms: Dict[str, np.ndarray]):
        """Update mechanism parameters"""
        self.mechanisms = mechanisms
        self.logger.debug(f"Mechanisms updated - Subsidies: {mechanisms['subsidies'][:3]}, "
                         f"Taxes: {mechanisms['taxes'][:3]}")
    
    def step_sell_bilevel(self, seller_states: np.ndarray, seller_actions: np.ndarray) -> np.ndarray:
        """
        Extended seller step with mechanism application
        
        Mechanisms applied:
        - taxes: increase effective prices (tax multiplier on price)
        
        Args:
            seller_states: (num_agents, state_dim)
            seller_actions: (num_agents, 2*num_commodities) - [price, waste_price]
        
        Returns:
            buyer_states: (num_agents, state_dim)
        """
        # Parse seller actions
        keys = ['price', 'waste_price']
        key_len_dict = {k: self.num_commodities for k in keys}
        seller_actions_dict = self.action_conversion(key_len_dict, seller_actions)
        
        # Apply mechanisms to prices
        # Mechanism: taxes increase effective prices (penalty on sellers)
        taxes = self.mechanisms['taxes']  # shape: (num_commodities,)
        
        # Original price + tax: p_final = p_raw * (1 + tax)
        seller_actions_dict['price'] = seller_actions_dict['price'] * (1.0 + taxes)
        
        # Subsidies reduce waste prices (incentive for waste exchange)
        subsidies = self.mechanisms['subsidies']  # shape: (num_commodities,)
        seller_actions_dict['waste_price'] = seller_actions_dict['waste_price'] * (1.0 - subsidies)
        
        # Apply price signals (global price multiplier)
        price_signals = self.mechanisms['price_signals']
        seller_actions_dict['price'] = seller_actions_dict['price'] * price_signals
        seller_actions_dict['waste_price'] = seller_actions_dict['waste_price'] * price_signals
        
        # Clamp prices to valid range
        for key in keys:
            seller_actions_dict[key] = np.clip(seller_actions_dict[key], 0.0, 1000.0)
        
        # Update prices in simulator state
        for key, value in seller_actions_dict.items():
            getattr(self, key)[..., self.t] = value
        
        # Get buyer states (same as original)
        buyer_states = self.get_buyer_state(keys, seller_states, seller_actions_dict)
        
        return buyer_states
    
    def step_buy_bilevel(self, buyer_states: np.ndarray, buyer_actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extended buyer step - same as original but with mechanisms already applied via prices
        
        Returns:
            trans_states: (num_agents, state_dim)
            buyer_rewards: (num_agents,)
            seller_rewards: (num_agents,)
        """
        # Original buyer step (mechanisms already reflected in prices)
        keys = ['q', 'waste_q', 'spot_q']
        nc = (self.num_agents - 1) * self.num_commodities
        lengths = [nc, nc, self.num_commodities]
        key_len_dict = {k: v for k, v in zip(keys, lengths)}
        buyer_actions_dict = self.action_conversion(key_len_dict, buyer_actions)
        
        # Reshape buyer actions
        for k, arr in buyer_actions_dict.items():
            if k == 'spot_q':
                continue
            new_actions = np.zeros((self.num_agents, self.num_agents, self.num_commodities))
            arr = arr.reshape(self.num_agents, self.num_agents - 1, self.num_commodities)
            for i in range(self.num_agents):
                i_list = list(range(self.num_agents))
                i_list.remove(i)
                new_actions[i, i_list] = arr[i]
            buyer_actions_dict[k] = new_actions
        
        # Update buyer actions in state
        for key, value in buyer_actions_dict.items():
            getattr(self, key)[..., self.t] = value
        
        # Get transformation states and rewards
        trans_states = self.get_trans_state(keys, buyer_states, buyer_actions_dict)
        buyer_rewards = self.get_buyer_reward()
        seller_rewards = self.get_seller_reward()
        
        return trans_states, buyer_rewards, seller_rewards
    
    def step_trans_bilevel(self, trans_states: np.ndarray, trans_actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Extended transformer step with mechanism application
        
        Mechanisms applied:
        - taxes: increase transformation costs (pollution penalty)
        - inv_taxes: increase inventory carrying costs
        
        Args:
            trans_states: (num_agents, state_dim)
            trans_actions: (num_agents, 2*num_commodities) - [tx_u, eco_u]
        
        Returns:
            seller_states: (num_agents, state_dim) for next step
            trans_rewards: (num_agents,) with mechanism adjustments
            done: bool
        """
        # Parse transformation actions
        keys = ['tx_u', 'eco_u']
        key_len_dict = {k: self.num_commodities for k in keys}
        trans_actions_dict = self.action_conversion(key_len_dict, trans_actions)
        
        # Apply action constraints (inventory limits)
        trans_actions_dict['tx_u'] = np.minimum(trans_actions_dict['tx_u'], 0.5 * self.inv_buy[..., self.t])
        trans_actions_dict['eco_u'] = np.minimum(trans_actions_dict['eco_u'], 0.5 * self.inv_buy[..., self.t])
        
        # Update state with transformation actions
        for key, value in trans_actions_dict.items():
            getattr(self, key)[..., self.t] = value
        
        # Apply surrogate model to get transformation outputs
        u_bot, w_bot = self.apply_agent_surrogate(trans_actions_dict['tx_u'])
        
        # Update inventories for next step
        self.inv[:, :, self.t + 1] = np.maximum(
            self.inv_buy[:, :, self.t] - trans_actions_dict['tx_u'][:, :] - 
            trans_actions_dict['eco_u'][:, :] + u_bot, 
            0.
        )
        self.waste_inv[:, :, self.t + 1] = (1 - self.delta) * (self.waste_inv_buy[:, :, self.t] + w_bot)
        
        # Compute transformation rewards with mechanisms
        trans_rewards = self.get_trans_reward_bilevel(trans_actions_dict)
        
        # Increment time
        self.t += 1
        
        # Get seller states for next step
        seller_states = self.get_seller_state()
        
        # Check termination
        done = (self.t >= self.episode_length)
        
        return seller_states, trans_rewards, done
    
    def get_trans_reward_bilevel(self, trans_actions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute transformation rewards with tax mechanisms applied
        
        Mechanism: taxes increase transformation costs (pollution penalty)
        
        Args:
            trans_actions: dict with 'tx_u' and 'eco_u' keys
        
        Returns:
            rewards: (num_agents,) adjusted for mechanisms
        """
        uc_p = self.uc_p[:, self.t].reshape(1, self.num_commodities)
        tx_p = self.tx_p[:, self.t].reshape(1, self.num_commodities)
        
        # Base reward
        reward = np.sum(self.eco_u[:, :, self.t] * uc_p, axis=1)
        
        # Apply mechanism: taxes increase transformation cost
        taxes = self.mechanisms['taxes']  # shape: (num_commodities,)
        tax_cost_multiplier = 1.0 + taxes
        tx_cost = np.sum(self.tx_u[:, :, self.t] * tx_p * tax_cost_multiplier, axis=1)
        
        reward -= tx_cost
        
        # Apply mechanism: inventory holding tax
        inv_taxes = self.mechanisms['inv_taxes']  # shape: (num_commodities,)
        inv_holding_penalty = np.sum(self.inv[:, :, self.t] * inv_taxes, axis=1)
        
        reward -= inv_holding_penalty
        
        return reward * self.RWD_SCALE
    
    def compute_upper_level_metrics(self) -> Dict[str, float]:
        """
        Compute upper-level (social/aggregated) metrics at current timestep
        
        Metrics:
        1. Environmental: wastewater produced + waste inventory held
        2. Economic: total revenue - total cost (net economic efficiency)
        3. Equity: variance of agent utilities (lower = more equitable)
        
        Returns:
            metrics: dict with 'environmental', 'economic', 'equity' keys
        """
        # Environmental metric: penalize wastewater and waste
        wastewater = np.sum(self.tx_u[:, :, self.t])
        waste_inventory = np.sum(self.waste_inv[:, :, self.t])
        env_metric = wastewater + 0.5 * waste_inventory
        
        # Economic metric: maximize net surplus
        total_revenue = (np.sum(self.price[:, :, self.t] * self.actual_d[:, :, :, self.t]) +
                        np.sum(self.waste_price[:, :, self.t] * self.waste_actual_d[:, :, :, self.t]))
        total_cost = np.sum(self.spot_price[:, self.t] * self.spot_q[:, :, self.t])
        econ_metric = -(total_revenue - total_cost)  # Negative for minimization
        
        # Equity metric: minimize variance in agent utilities
        agent_utilities = np.sum(self.actual_d[:, :, :, self.t], axis=(1, 2))
        equity_metric = np.var(agent_utilities) if self.num_agents > 1 else 0.0
        
        return {
            'environmental': env_metric,
            'economic': econ_metric,
            'equity': equity_metric
        }
    
    def reset_episode_tracking(self):
        """Reset episode-level metric tracking"""
        self.episode_metrics = {
            'environmental': [],
            'economic': [],
            'equity': []
        }
        self.episode_lower_rewards = {
            SELLER: [],
            BUYER: [],
            TRANSFORM: []
        }
    
    def record_step_metrics(self, metrics: Dict[str, float], rewards: Dict[str, float]):
        """Record metrics at end of timestep"""
        for key, val in metrics.items():
            self.episode_metrics[key].append(val)
        
        for role, val in rewards.items():
            self.episode_lower_rewards[role].append(val)
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """
        Summarize metrics over entire episode
        
        Returns:
            dict with aggregated metrics and rewards
        """
        summary = {
            'metrics': {},
            'lower_rewards': {}
        }
        
        # Average upper-level metrics
        for metric_type in ['environmental', 'economic', 'equity']:
            if self.episode_metrics[metric_type]:
                summary['metrics'][metric_type] = np.mean(self.episode_metrics[metric_type])
        
        # Average lower-level rewards
        for role in [SELLER, BUYER, TRANSFORM]:
            if self.episode_lower_rewards[role]:
                summary['lower_rewards'][role] = np.mean(self.episode_lower_rewards[role])
        
        return summary


class BilevelTrajectoryCollector:
    """
    Collects trajectories under bilevel optimization
    
    Coordinates between:
    - Mechanism designer (upper level)
    - Agent policies (lower level)
    - Extended bilevel simulator
    """
    
    def __init__(self, num_agents: int = 3, num_commodities: int = 12):
        self.num_agents = num_agents
        self.num_commodities = num_commodities
        self.simulator = BilevelManufacturingSimulator()
    
    def rollout_episode(self, agent_pool, mechanisms: Dict[str, np.ndarray], 
                       episode_length: int = 1000) -> Tuple[Dict, Dict]:
        """
        Collect one full episode with given mechanisms
        
        Args:
            agent_pool: AgentPool object for policy evaluation
            mechanisms: dict with mechanism parameters
            episode_length: length of episode
        
        Returns:
            trajectories: dict with observations, actions, rewards
            metrics: dict with upper and lower level metrics
        """
        self.simulator.set_mechanisms(mechanisms)
        self.simulator.reset_episode_tracking()
        
        trajectories = {
            'observations': {SELLER: [], BUYER: [], TRANSFORM: []},
            'actions': {SELLER: [], BUYER: [], TRANSFORM: []},
            'rewards': {SELLER: [], BUYER: [], TRANSFORM: []}
        }
        
        # Reset environment
        obs_s = self.simulator.reset()
        
        for step in range(episode_length):
            # Collect seller data
            trajectories['observations'][SELLER].append(obs_s)
            action_s, log_prob_s = agent_pool.get_actions(obs_s, SELLER)
            trajectories['actions'][SELLER].append(action_s)
            
            # Seller step (with mechanisms applied)
            obs_b = self.simulator.step_sell_bilevel(obs_s, action_s)
            
            # Collect buyer data
            trajectories['observations'][BUYER].append(obs_b)
            action_b, log_prob_b = agent_pool.get_actions(obs_b, BUYER)
            trajectories['actions'][BUYER].append(action_b)
            
            # Buyer step
            obs_t, rew_b, rew_s = self.simulator.step_buy_bilevel(obs_b, action_b)
            trajectories['rewards'][SELLER].append(rew_s)
            trajectories['rewards'][BUYER].append(rew_b)
            
            # Collect transformer data
            trajectories['observations'][TRANSFORM].append(obs_t)
            action_t, log_prob_t = agent_pool.get_actions(obs_t, TRANSFORM)
            trajectories['actions'][TRANSFORM].append(action_t)
            
            # Transformer step (with mechanisms applied)
            obs_s, rew_t, done = self.simulator.step_trans_bilevel(obs_t, action_t)
            trajectories['rewards'][TRANSFORM].append(rew_t)
            
            # Compute upper-level metrics
            metrics = self.simulator.compute_upper_level_metrics()
            self.simulator.record_step_metrics(metrics, {
                SELLER: rew_s.mean(),
                BUYER: rew_b.mean(),
                TRANSFORM: rew_t.mean()
            })
            
            if done:
                break
        
        # Get episode summary
        episode_summary = self.simulator.get_episode_summary()
        
        return trajectories, episode_summary
