# circular_economy_RL_lib/trainer.py

import torch
import torch.optim as optim
import numpy as np
import os
from simulator import Manufacturing_Simulator
from agent import AgentPool
from market_authority import MarketAuthority
from config import config, SELLER, BUYER, TRANSFORM, stages
from utils import get_result_folder, seedEverything
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self):
        for key, value in config.items():
            setattr(self, key, value)

        if self.seed is not None:
            seedEverything(self.seed)

        self.env = Manufacturing_Simulator()
        
        # Upper Level (Leader Policy)
        self.market_authority = MarketAuthority(self.num_commodities)
        self.upper_optimizer = optim.Adam(self.market_authority.parameters(), lr=self.upper_lr)
        
        # Lower Level (Decentralized Followers)
        # We track two separate pools corresponding to different reward profiles
        self.agent_pool_pure = AgentPool(self.num_agents, self.num_commodities, self.history_length, prefix="pure")
        self.agent_pool_penalized = AgentPool(self.num_agents, self.num_commodities, self.history_length, prefix="penalized")
        
        # Logging
        log_folder = get_result_folder() + '/bilevel_log'
        self.writer = SummaryWriter(log_folder)

    def rollout_episode(self, pool, mode="pure", mechanism_np=None):
        """
        Run a single full rollout episode under the fixed mechanism parameters.
        Returns trajectories and aggregated objectives.
        """
        self.env.set_market_mechanism(mechanism_np)
        obs_s = self.env.reset()
        
        batch_obs = [[] for _ in range(len(stages))]
        batch_acts = [[] for _ in range(len(stages))]
        batch_log_probs = [[] for _ in range(len(stages))]
        batch_rews = [[] for _ in range(len(stages))]

        total_lower_rewards = 0.0
        
        for ep_t in range(self.episode_length):
            # 1. Seller Stage
            batch_obs[SELLER].append(obs_s)
            action_s, log_prob_s = pool.get_actions(obs_s, SELLER)
            batch_acts[SELLER].append(action_s)
            batch_log_probs[SELLER].append(log_prob_s)
            
            obs_b = self.env.step_sell(obs_s, action_s)

            # 2. Buyer Stage
            batch_obs[BUYER].append(obs_b)
            action_b, log_prob_b = pool.get_actions(obs_b, BUYER)
            batch_acts[BUYER].append(action_b)
            batch_log_probs[BUYER].append(log_prob_b)
            
            obs_t, rew_b, rew_s = self.env.step_buy(obs_b, action_b)
            batch_rews[SELLER].append(rew_s)
            batch_rews[BUYER].append(rew_b)

            # 3. Transform Stage
            batch_obs[TRANSFORM].append(obs_t)
            action_t, log_prob_t = pool.get_actions(obs_t, TRANSFORM)
            batch_acts[TRANSFORM].append(action_t)
            batch_log_probs[TRANSFORM].append(log_prob_t)
            
            obs_s, rew_t, done = self.env.step_trans(obs_t, action_t)
            batch_rews[TRANSFORM].append(rew_t)

            # Accumulate reward profiles step-wise
            pure_step_rwd = (rew_s + rew_b + rew_t).mean()
            total_lower_rewards += pure_step_rwd

        # Apply Gaur et al. 2025 Penalty step-wise across steps to stabilize policy optimization
        if mode == "penalized":
            for stage in stages:
                for step in range(self.episode_length):
                    # Cost represents social penalties
                    penalty = self.penalty_sigma * (
                        self.upper_weight_wastewater * self.env.wastewater[0, 0, step + self.history_length]
                        + self.upper_weight_inventory * self.env.waste_inv[:, :, step + self.history_length].sum()
                    )
                    batch_rews[stage][step] = batch_rews[stage][step] - penalty

        # Determine societal upper-level cost for Outer Update
        upper_level_metrics = self.env.get_upper_level_reward()
        upper_level_cost = (
            -1.0 * (self.upper_weight_profit * upper_level_metrics["profit"]
            - self.upper_weight_wastewater * upper_level_metrics["wastewater"]
            - self.upper_weight_inventory * upper_level_metrics["waste_inventory"]
            - self.upper_weight_market_balance * upper_level_metrics["imbalance"])
        )

        return batch_obs, batch_acts, batch_log_probs, batch_rews, total_lower_rewards, upper_level_cost

    def compute_rtgs(self, batch_rews):
        batch_rtgs = [[] for _ in stages]
        for stage in stages:
            ep_rtgs = []
            discounted_reward = np.zeros_like(batch_rews[stage][0])
            for rew in reversed(batch_rews[stage]):
                discounted_reward = rew + discounted_reward * self.gamma
                ep_rtgs.insert(0, discounted_reward)
            batch_rtgs[stage] = torch.tensor(ep_rtgs, dtype=torch.float).reshape(-1, self.num_agents)
        return batch_rtgs

    def run_inner_loop(self, pool, mode="pure", mechanism_np=None):
        """
        Maximizes eitherpure return J or penalized return h_sigma over K updates.
        This provides the approximate optimal response policy (lambda* or lambda*_sigma).
        """
        for _ in range(self.inner_optimization_steps):
            obs, acts, log_p, rews, _, _ = self.rollout_episode(pool, mode, mechanism_np)
            rtgs = self.compute_rtgs(rews)
            
            # Format vectors to feed policy optimizers
            formatted_obs = [torch.tensor(obs[s], dtype=torch.float) for s in stages]
            formatted_acts = [torch.tensor(acts[s], dtype=torch.float) for s in stages]
            formatted_log_p = [torch.tensor(log_p[s], dtype=torch.float) for s in stages]
            
            for ag in range(self.num_agents):
                for stage in stages:
                    pool.learn(formatted_obs, formatted_acts, formatted_log_p, rtgs, stage, ag, self.n_updates_per_iteration)

    def learn(self):
        for epoch in range(self.num_epochs):
            # State evaluation
            market_state = self.env.get_market_state()
            
            # Step 1: Draw stochastic mechanism from Market Authority
            mechanism_np, log_prob = self.market_authority(market_state)
            
            # Step 2: Inner Loop Phase A - Estimate lambda*(phi) under Pure Rewards
            self.run_inner_loop(self.agent_pool_pure, mode="pure", mechanism_np=mechanism_np)
            
            # Step 3: Inner Loop Phase B - Estimate lambda*_sigma(phi) under Penalized Rewards
            self.run_inner_loop(self.agent_pool_penalized, mode="penalized", mechanism_np=mechanism_np)
            
            # Step 4: Sample converged trajectory data to evaluate the Hessian-free proxy gradient
            _, _, _, _, total_lower_rewards_pure, _ = self.rollout_episode(self.agent_pool_pure, mode="pure", mechanism_np=mechanism_np)
            _, _, _, _, total_lower_rewards_pen, upper_level_cost = self.rollout_episode(self.agent_pool_penalized, mode="penalized", mechanism_np=mechanism_np)
            
            # Step 5: Construct First-Order Proxy Hypergradient Update (Gaur et al. 2025)
            # grad(log_prob * multiplier) gives the exact REINFORCE policy gradient equivalent
            self.upper_optimizer.zero_grad()
            
            # Re-evaluate log_prob dynamically in current computational graph
            _, dynamic_log_prob = self.market_authority(market_state)
            
            # proxy_loss = G + 1/sigma * (J*(phi) - J(phi, lambda*_sigma))
            multiplier = upper_level_cost + (1.0 / self.penalty_sigma) * (total_lower_rewards_pure - total_lower_rewards_pen)
            surrogate_objective = dynamic_log_prob * multiplier
            
            surrogate_objective.backward()
            self.upper_optimizer.step()

            # Logging metrics
            self.writer.add_scalar('BRL/upper_level_cost', upper_level_cost, epoch)
            self.writer.add_scalar('BRL/lower_rewards_pure', total_lower_rewards_pure, epoch)
            self.writer.add_scalar('BRL/lower_rewards_penalized', total_lower_rewards_pen, epoch)
            self.writer.add_scalar('BRL/spot_mult_mean', mechanism_np["spot_mult"], epoch)
            self.writer.add_scalar('BRL/waste_penalty_mean', mechanism_np["waste_penalty"], epoch)
            
            print(f"Epoch {epoch+1:03d}/{self.num_epochs} "
                  f"| Upper Cost (G): {upper_level_cost:.4f} "
                  f"| Pure Return (J*): {total_lower_rewards_pure:.4f} "
                  f"| Penalized Return: {total_lower_rewards_pen:.4f}")
            
            # Save periodic checkpoints
            if (epoch + 1) % self.save_freq == 0:
                for ag in range(self.num_agents):
                    self.agent_pool_pure.save_model(TRANSFORM, ag, f"ppo_pure_ag{ag}")
                    self.agent_pool_penalized.save_model(TRANSFORM, ag, f"ppo_penalized_ag{ag}")

        self.writer.close()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.learn()
