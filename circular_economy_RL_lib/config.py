# circular_economy_RL_lib/config.py

import numpy as np
from dataclasses import dataclass

## Settings for the stages
SELLER = 0
BUYER = 1
TRANSFORM = 2
stages = {SELLER, BUYER, TRANSFORM}

config = {
    # System parameters
    'num_agents': 3,
    'num_commodities': 12,
    
    # Optimization parameters
    'alpha': 0.5,
    'beta': 0.5,
    'delta': 0.5,
    'LAMBDA': 0.5,
    'UC': 0.5,
    'TX_P': 0.5,
    'INIT_INV': 100,
    'RWD_SCALE': 1e-9,
    
    # Training parameters (Lower Level)
    'gamma': 0.99,
    'num_steps': 1000, 
    'episode_length': 1000,
    'num_epochs': 100,
    'history_length': 5,
    'decay_factor': 0.1,
    'n_updates_per_iteration': 10,
    'lr': 3e-4,
    'clip': 0.2,
    'save_freq': 5,
    'seed': 2024,
    'price_factor': 1.0,
    
    # Bilevel Hyperparameters (Upper Level - Gaur et al. 2025)
    'upper_lr': 1e-4,
    'penalty_sigma': 5.0,            # Penalty parameter (sigma) for proxy objective
    'inner_optimization_steps': 5,   # Number of inner PPO epochs per outer iteration (K)
    'upper_weight_profit': 1.0,      # Weight of profits in G
    'upper_weight_wastewater': 0.1,  # Weight of wastewater in G
    'upper_weight_inventory': 0.05,  # Weight of waste inventory in G
    'upper_weight_market_balance': 0.05, # Weight of balance mismatch in G
}

def init_historical_data():
    historic_data = {}
    historic_data['spot_price'] = np.array([
        [config['price_factor'] * 0.5], [0.8], [1.], [3.], [20.], [4.], 
        [8.], [100.], [0.2], [1.2], [0.15], [1.173]
    ])
    return historic_data
