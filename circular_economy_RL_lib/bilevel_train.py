"""
Bilevel Optimization Training Script
=====================================

Entry point for training the bilevel optimization framework.
Upper level: Market authority (mechanism designer)
Lower level: Individual agents responding to mechanisms

Usage: python bilevel_train.py
"""

##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

##########################################################################################
# Imports

import torch
import logging
import multiprocessing
from utils import create_logger, get_result_folder
from bilevel_mechanism import (
    MechanismDesigner,
    MultiAgentBilevelPolicies,
    BilevelOptimizer
)
from config import config
import numpy as np
from torch.utils.tensorboard import SummaryWriter

##########################################################################################
# Parameters

logger_params = {
    'log_file': {
        'desc': 'bilevel_train',
        'filename': 'bilevel_run_log'
    }
}

bilevel_params = {
    'num_agents': config.get('num_agents', 3),
    'num_commodities': config.get('num_commodities', 12),
    'mechanism_hidden_dim': 256,
    'agent_hidden_dim': 128,
    'state_dim': 5 * config.get('num_commodities', 12),  # aggregated market state
    'outer_lr': 1e-4,
    'inner_lr': 1e-3,
    'inner_steps': 5,
    'num_epochs': config.get('num_epochs', 100),
    'log_freq': 10,
    'device': 'cuda' if (USE_CUDA and torch.cuda.is_available()) else 'cpu'
}

##########################################################################################
# Main Training Class

class BilevelTrainer:
    """
    Orchestrates bilevel optimization training
    """
    
    def __init__(self, params: dict):
        self.params = params
        self.device = torch.device(params['device'])
        self.logger = logging.getLogger('BilevelTrainer')
        
        # Initialize components
        self.mechanism_net = MechanismDesigner(
            num_commodities=params['num_commodities'],
            hidden_dim=params['mechanism_hidden_dim'],
            device=params['device']
        )
        
        self.agent_policies = MultiAgentBilevelPolicies(
            num_agents=params['num_agents'],
            num_commodities=params['num_commodities'],
            state_dim=params['state_dim'],
            hidden_dim=params['agent_hidden_dim'],
            device=params['device']
        )
        
        self.optimizer = BilevelOptimizer(
            mechanism_net=self.mechanism_net,
            agent_policies=self.agent_policies,
            num_agents=params['num_agents'],
            num_commodities=params['num_commodities'],
            outer_lr=params['outer_lr'],
            inner_lr=params['inner_lr'],
            inner_steps=params['inner_steps'],
            device=params['device']
        )
        
        # Logging and checkpointing
        log_folder = get_result_folder() + '/bilevel_log'
        self.checkpoint_folder = get_result_folder() + '/bilevel_checkpoints'
        os.makedirs(log_folder, exist_ok=True)
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        
        self.writer = SummaryWriter(log_folder)
        self.logger.info(f"Bilevel Training initialized on device: {self.device}")
        self.logger.info(f"Checkpoint folder: {self.checkpoint_folder}")
    
    def train(self):
        """
        Main training loop
        """
        num_epochs = self.params['num_epochs']
        log_freq = self.params['log_freq']
        
        self.logger.info("="*70)
        self.logger.info("Starting Bilevel Optimization Training")
        self.logger.info("="*70)
        self.logger.info(f"Upper Level: Market Authority (Mechanism Designer)")
        self.logger.info(f"Lower Level: Individual Agents (Sellers, Buyers, Transformers)")
        self.logger.info(f"Total Epochs: {num_epochs}")
        self.logger.info("="*70)
        
        for epoch in range(num_epochs):
            # Get aggregated market state
            market_state = self.optimizer.get_market_state()
            
            # Run one bilevel optimization step
            results = self.optimizer.outer_optimization_step(market_state)
            
            # Log to tensorboard
            self.writer.add_scalar('upper_loss', results['upper_loss'], epoch)
            self.writer.add_scalar('lower_loss', results['lower_loss'], epoch)
            self.writer.add_scalar('env_metric', results['env_metric'], epoch)
            self.writer.add_scalar('econ_metric', results['econ_metric'], epoch)
            self.writer.add_scalar('equity_metric', results['equity_metric'], epoch)
            self.writer.add_scalar('mechanism/subsidies_mean', results['subsidies_mean'], epoch)
            self.writer.add_scalar('mechanism/taxes_mean', results['taxes_mean'], epoch)
            
            # Periodic logging
            if (epoch + 1) % log_freq == 0:
                self.logger.info(f"\n{'='*70}")
                self.logger.info(f"Epoch [{epoch+1:3d}/{num_epochs}]")
                self.logger.info(f"{'='*70}")
                self.logger.info(f"Upper Level Loss:    {results['upper_loss']:12.6f}")
                self.logger.info(f"Lower Level Loss:    {results['lower_loss']:12.6f}")
                self.logger.info(f"Environment Metric:  {results['env_metric']:12.6f}")
                self.logger.info(f"Economic Metric:     {results['econ_metric']:12.6f}")
                self.logger.info(f"Equity Metric:       {results['equity_metric']:12.6f}")
                self.logger.info(f"Subsidies (mean):    {results['subsidies_mean']:12.6f}")
                self.logger.info(f"Taxes (mean):        {results['taxes_mean']:12.6f}")
                
                # Save checkpoint
                self._save_checkpoint(epoch)
        
        self.logger.info("\n" + "="*70)
        self.logger.info("*** Bilevel Training Complete ***")
        self.logger.info("="*70)
        self.writer.close()
    
    def _save_checkpoint(self, epoch: int):
        """
        Save model checkpoints
        """
        checkpoint = {
            'epoch': epoch,
            'mechanism_net_state': self.mechanism_net.state_dict(),
            'agent_policies_state': self.agent_policies.state_dict(),
            'params': self.params
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_folder, 
            f'bilevel_checkpoint_epoch_{epoch+1}.pt'
        )
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")


##########################################################################################
# Utility Functions

def _print_config():
    """Print configuration"""
    logger = logging.getLogger('root')
    logger.info("="*70)
    logger.info("BILEVEL OPTIMIZATION CONFIGURATION")
    logger.info("="*70)
    logger.info(f"DEBUG_MODE: {DEBUG_MODE}")
    logger.info(f"USE_CUDA: {USE_CUDA}, CUDA_DEVICE_NUM: {CUDA_DEVICE_NUM}")
    logger.info(f"Number of CPUs: {multiprocessing.cpu_count()}")
    logger.info("="*70)
    logger.info("BILEVEL PARAMETERS:")
    for key, value in bilevel_params.items():
        logger.info(f"  {key:25s}: {value}")
    logger.info("="*70)
    logger.info("SYSTEM PARAMETERS (from config):")
    for key, value in config.items():
        logger.info(f"  {key:25s}: {value}")
    logger.info("="*70)


##########################################################################################
# Main

def main():
    """Main entry point"""
    create_logger(**logger_params)
    _print_config()
    
    # Initialize trainer
    trainer = BilevelTrainer(bilevel_params)
    
    # Run training
    trainer.train()


##########################################################################################

if __name__ == "__main__":
    main()
