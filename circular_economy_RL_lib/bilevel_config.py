"""
Bilevel Optimization Configuration
===================================

Bilevel-specific hyperparameters and system parameters for circular economy optimization.

Upper Level (Mechanism Designer):
- Designs market mechanisms: subsidies, taxes, price signals
- Objectives: Environmental sustainability, Economic efficiency, Equity

Lower Level (Agent Policies):
- Individual sellers, buyers, transformers optimize own utility
- Respond to upper-level mechanisms
"""

import numpy as np
from dataclasses import dataclass

##########################################################################################
# BILEVEL STRUCTURE CONFIGURATION

# Role IDs
SELLER = 0
BUYER = 1
TRANSFORMER = 2
roles = {SELLER, BUYER, TRANSFORMER}

# Number of agents and commodities
num_agents = 3
num_commodities = 12

##########################################################################################
# MECHANISM PARAMETERS (Upper Level)

# Market authority objectives and weights
mechanism_objectives = {
    'environmental_sustainability': 0.40,    # Weight for environment metric
    'economic_efficiency': 0.35,             # Weight for economic metric
    'equity': 0.25                           # Weight for equity metric
}

# Mechanism variable bounds and scales
mechanism_config = {
    # Subsidy bounds: [0, max_subsidy] reduces waste prices
    'max_subsidy': 0.3,                      # Maximum subsidy per commodity
    'subsidy_granularity': 100,              # Number of subsidy levels
    
    # Tax bounds: [0, max_tax] increases transformation costs
    'max_tax': 0.2,                          # Maximum tax per commodity
    'tax_granularity': 100,                  # Number of tax levels
    
    # Price signal bounds: [0.5, 1.5] multiplies spot prices
    'price_signal_min': 0.5,
    'price_signal_max': 1.5,
    'price_signal_granularity': 100,
    
    # Inventory holding tax: [0, max_inv_tax] penalizes inventory carrying
    'max_inv_tax': 0.15,
    'inv_tax_granularity': 100,
}

##########################################################################################
# MARKET STATE AGGREGATION

# Market state dimensions for mechanism designer input
market_state_config = {
    'spot_price_dim': num_commodities,           # Current spot prices
    'avg_agent_price_dim': num_commodities,      # Average agent prices
    'avg_waste_price_dim': num_commodities,      # Average waste prices
    'avg_inventory_dim': num_commodities,        # Average inventories
    'avg_waste_inventory_dim': num_commodities,  # Average waste inventories
}

market_state_total_dim = sum(market_state_config.values())  # 5 * num_commodities = 60

##########################################################################################
# AGENT POLICY PARAMETERS (Lower Level)

# Agent observation dimensions (per role)
agent_obs_config = {
    'seller_obs_dim': num_commodities * 5 * (6 + num_agents * 8) + 2 * num_commodities,
    'buyer_obs_dim': None,  # Computed dynamically
    'transformer_obs_dim': None,  # Computed dynamically
}

# Agent action dimensions (per role)
agent_action_config = {
    'seller_action_dim': 2 * num_commodities,           # (price, waste_price)
    'buyer_action_dim': 2 * (num_agents - 1) * num_commodities + num_commodities,  # (q, waste_q, spot_q)
    'transformer_action_dim': 2 * num_commodities,      # (tx_u, eco_u)
}

##########################################################################################
# LOWER-LEVEL DYNAMICS

# Inventory dynamics
inventory_config = {
    'delta': 0.5,                    # Waste decay rate: waste_inv *= (1 - delta)
    'init_inv': 100.0,               # Initial inventory level per commodity
    'init_waste_inv': 50.0,          # Initial waste inventory per commodity
}

# Transformation and utility
transformation_config = {
    'UC': 0.5,                       # Unit cost of economic utility
    'TX_P': 0.5,                     # Unit cost of transformation (influenced by taxes)
    'LAMBDA': 0.5,                   # Demand satisfaction reward weight
    'rwd_scale': 1e-9,               # Reward scaling factor
}

# Initial commodities prices and availability
commodities = [
    'water',                    # 0
    'costly_water',             # 1
    'acetic_acid',              # 2
    'hydrogen',                 # 3
    'nitrobenzene',             # 4
    'PAP',                       # 5
    'acetaminophen',            # 6
    'acetic_anhydride',         # 7
    'oxygen',                   # 8
    'ammonia',                  # 9
    'sulfuric_acid',            # 10
    'aniline'                   # 11
]

spot_price_initial = np.array([0.5, 0.8, 1.0, 3.0, 20.0, 4.0, 8.0, 100.0, 0.2, 1.2, 0.15, 1.173])

##########################################################################################
# BILEVEL OPTIMIZATION HYPERPARAMETERS

bilevel_training_config = {
    # Outer level (mechanism designer) learning
    'outer_learning_rate': 1e-4,
    'outer_optimizer': 'adam',
    'outer_grad_clip': 1.0,
    
    # Inner level (agent policies) learning
    'inner_learning_rate': 1e-3,
    'inner_optimizer': 'adam',
    'inner_grad_clip': 1.0,
    'inner_optimization_steps': 5,           # Number of inner steps per outer step
    
    # Training schedule
    'num_epochs': 100,
    'num_inner_rollouts_per_epoch': 1,
    'episode_length': 100,
    'log_frequency': 10,
    'checkpoint_frequency': 10,
    
    # Batch configuration
    'batch_size': 1,                         # Number of market states per epoch
    'num_agents_in_episode': num_agents,
    'num_commodities_in_episode': num_commodities,
}

##########################################################################################
# NEURAL NETWORK ARCHITECTURE

network_config = {
    # Mechanism Designer Network
    'mechanism_designer': {
        'encoder_hidden_dim': 256,
        'encoder_layers': 2,
        'encoder_activation': 'relu',
        'encoder_norm': 'layer_norm',
        'encoder_dropout': 0.1,
        
        'subsidy_head_hidden_dim': 128,
        'tax_head_hidden_dim': 128,
        'price_signal_head_hidden_dim': 128,
        'inv_tax_head_hidden_dim': 128,
    },
    
    # Lower-Level Agent Policies
    'agent_policy': {
        'hidden_dim': 128,
        'num_layers': 2,
        'activation': 'relu',
        'norm': 'layer_norm',
        'dropout': 0.1,
        'action_scaling': 'sigmoid',
    },
}

##########################################################################################
# EVALUATION METRICS (Upper Level Objectives)

metrics_config = {
    # Environmental Sustainability Metric
    'environmental': {
        'wastewater_component': 1.0,         # Weight for wastewater produced
        'waste_inventory_component': 0.5,    # Weight for waste inventory held
        'minimize': True,                    # Lower is better
        'description': 'Minimize wastewater and waste inventory'
    },
    
    # Economic Efficiency Metric
    'economic': {
        'revenue_component': 1.0,            # Weight for revenue
        'cost_component': -1.0,              # Weight for cost (negative = minimize)
        'maximize': True,                    # Higher is better
        'description': 'Maximize total economic surplus'
    },
    
    # Equity Metric
    'equity': {
        'utility_variance': 1.0,             # Weight for variance in agent utilities
        'minimize': True,                    # Lower variance = more equitable
        'description': 'Minimize inequality in agent utilities'
    },
}

##########################################################################################
# DEVICE AND COMPUTATION

computation_config = {
    'use_cuda': True,
    'cuda_device': 0,
    'dtype': 'float32',
    'seed': 2024,
}

##########################################################################################
# MECHANISM INITIALIZATION

def initialize_mechanism_state():
    """
    Initialize mechanism parameters to neutral/default values
    
    Returns:
        dict with initial mechanism values
    """
    return {
        'subsidies': np.zeros(num_commodities),
        'taxes': np.zeros(num_commodities),
        'price_signals': np.ones(num_commodities),
        'inv_taxes': np.zeros(num_commodities),
    }

##########################################################################################
# VALIDATION UTILITIES

def validate_bilevel_config():
    """
    Validate configuration consistency and constraints
    """
    errors = []
    warnings = []
    
    # Check mechanism objectives sum to 1.0
    obj_sum = sum(mechanism_objectives.values())
    if not np.isclose(obj_sum, 1.0):
        warnings.append(f"Mechanism objectives sum to {obj_sum}, not 1.0")
    
    # Check agent counts
    if num_agents < 2:
        errors.append("num_agents must be >= 2")
    
    if num_commodities < 1:
        errors.append("num_commodities must be >= 1")
    
    # Check spot prices
    if len(spot_price_initial) != num_commodities:
        errors.append(f"spot_price_initial length {len(spot_price_initial)} != num_commodities {num_commodities}")
    
    # Check bounds
    if mechanism_config['max_subsidy'] < 0 or mechanism_config['max_subsidy'] > 1:
        errors.append("max_subsidy must be in [0, 1]")
    
    if mechanism_config['max_tax'] < 0 or mechanism_config['max_tax'] > 1:
        errors.append("max_tax must be in [0, 1]")
    
    # Check learning rates
    if bilevel_training_config['outer_learning_rate'] <= 0:
        errors.append("outer_learning_rate must be positive")
    
    if bilevel_training_config['inner_learning_rate'] <= 0:
        errors.append("inner_learning_rate must be positive")
    
    # Check optimization steps
    if bilevel_training_config['inner_optimization_steps'] < 1:
        errors.append("inner_optimization_steps must be >= 1")
    
    return errors, warnings

##########################################################################################
# PRINT CONFIGURATION

def print_bilevel_config():
    """Pretty print bilevel configuration"""
    print("\n" + "="*80)
    print("BILEVEL OPTIMIZATION CONFIGURATION")
    print("="*80)
    
    print("\n[STRUCTURE]")
    print(f"  Agents: {num_agents}")
    print(f"  Commodities: {num_commodities}")
    print(f"  Roles: {roles}")
    
    print("\n[MECHANISM OBJECTIVES]")
    for obj, weight in mechanism_objectives.items():
        print(f"  {obj:30s}: {weight:.2f}")
    
    print("\n[MECHANISM BOUNDS]")
    for key, val in mechanism_config.items():
        if 'granularity' not in key:
            print(f"  {key:30s}: {val}")
    
    print("\n[LOWER LEVEL DYNAMICS]")
    for key, val in transformation_config.items():
        print(f"  {key:30s}: {val}")
    for key, val in inventory_config.items():
        print(f"  {key:30s}: {val}")
    
    print("\n[BILEVEL TRAINING]")
    for key, val in bilevel_training_config.items():
        print(f"  {key:30s}: {val}")
    
    print("\n" + "="*80 + "\n")

##########################################################################################
