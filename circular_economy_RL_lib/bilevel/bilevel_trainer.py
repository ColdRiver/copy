import copy
import torch
import torch.nn as nn
import torch.optim as optim

from bilevel_metrics import compute_upper_objective


class BilevelTrainer:

    def __init__(
        self,
        mechanism_net,
        market_policy,
        env,
        outer_lr=1e-4,
        inner_lr=3e-4,
        inner_steps=5,
        gamma=0.99,
        device="cuda"
    ):

        self.device = device

        self.mechanism_net = mechanism_net.to(device)
        self.market_policy = market_policy.to(device)

        self.env = env

        self.gamma = gamma
        self.inner_steps = inner_steps

        self.outer_optimizer = optim.Adam(
            self.mechanism_net.parameters(),
            lr=outer_lr
        )

        self.inner_optimizer = optim.Adam(
            self.market_policy.parameters(),
            lr=inner_lr
        )

    def rollout_lower_level(
        self,
        policy,
        mechanism_net
    ):

        state = self.env.reset()

        rewards = []

        done = False

        while not done:

            mech_state = self.env.get_mechanism_state()

            subsidy, tax = mechanism_net(mech_state)

            action = policy(state)

            next_state, reward, done, info = self.env.step(
                action,
                subsidy,
                tax
            )

            rewards.append(reward)

            state = next_state

        rewards = torch.stack(rewards)

        return rewards.mean()

    def solve_inner_problem(self):

        adapted_policy = copy.deepcopy(
            self.market_policy
        )

        optimizer = optim.Adam(
            adapted_policy.parameters(),
            lr=self.inner_optimizer.param_groups[0]["lr"]
        )

        for _ in range(self.inner_steps):

            lower_objective = self.rollout_lower_level(
                adapted_policy,
                self.mechanism_net
            )

            loss = -lower_objective

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        return adapted_policy

    def outer_objective(
        self,
        adapted_policy
    ):

        state = self.env.reset()

        env_metrics = []
        econ_metrics = []
        equity_metrics = []

        done = False

        while not done:

            mech_state = self.env.get_mechanism_state()

            subsidy, tax = self.mechanism_net(
                mech_state
            )

            action = adapted_policy(state)

            next_state, _, done, info = self.env.step(
                action,
                subsidy,
                tax
            )

            env_metrics.append(
                info["env_metric"]
            )

            econ_metrics.append(
                info["econ_metric"]
            )

            equity_metrics.append(
                info["equity_metric"]
            )

            state = next_state

        env_metric = torch.stack(
            env_metrics
        ).mean()

        econ_metric = torch.stack(
            econ_metrics
        ).mean()

        equity_metric = torch.stack(
            equity_metrics
        ).mean()

        return compute_upper_objective(
            env_metric,
            econ_metric,
            equity_metric
        )

    def bilevel_step(self):

        adapted_policy = self.solve_inner_problem()

        upper_loss = self.outer_objective(
            adapted_policy
        )

        self.outer_optimizer.zero_grad()

        upper_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.mechanism_net.parameters(),
            1.0
        )

        self.outer_optimizer.step()

        return {
            "upper_loss": upper_loss.item()
        }
