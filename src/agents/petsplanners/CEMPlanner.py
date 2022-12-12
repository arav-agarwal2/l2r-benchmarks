"""CEM Planner for PETS. Note that we've had some errors with this in the past, due to not tuning the parameters properly."""
from abc import ABC
from ast import Not
import numpy as np
import scipy
import gym

import torch
from src.agents.petsplanners.base import BasePlanner
from src.config.yamlize import yamlize
from src.constants import DEVICE


@yamlize
class CEMPlanner(BasePlanner):
    """CEM ( Gaussian Evolutionary Method ) Based Planner. Untuned."""

    def __init__(
        self,
        action_dim: int = 2,
        action_min: float = -1,
        action_max: float = 1,
        n_planner: int = 500,
        horizon: int = 12,
        iter_update_steps: int = 3,
        k_best: int = 5,
        epsilon: float = 1e-3,
        update_alpha: float = 0.1,
        lb: float = -0.5,
        ub: float = 0.5,
    ):
        """Initialize CEM Planner

        Args:
            action_dim (int, optional): Number of dimensions in action space. Defaults to 2.
            action_min (float, optional): Minimum action value. Defaults to -1.
            action_max (float, optional): Maximum action value. Defaults to 1.
            n_planner (int, optional): Number of trajectories to plan. Defaults to 500.
            horizon (int, optional): How many steps in the future to plan. Defaults to 12.
            iter_update_steps (int, optional): How many times to iterate on the planning side. Defaults to 3.
            k_best (int, optional): Number of best trajectories to keep. Defaults to 5.
            epsilon (float, optional): Minimum variance of planner. Defaults to 1e-3.
            update_alpha (float, optional): How much to update planner based on results. Defaults to 0.1.
            lb (float, optional): Lower bound. Defaults to -0.5.
            ub (float, optional): Upper bound. Defaults to 0.5.
        """
        super().__init__(n_planner, horizon)
        self.iter_update_steps = iter_update_steps
        self.k_best = k_best
        self.epsilon = epsilon
        self.lb = lb
        self.ub = ub
        self.action_low = action_min
        self.action_high = action_max
        self.action_space_size = action_dim
        self.update_alpha = update_alpha

    def get_action(
        self, state, dynamics_model: torch.nn.Module
    ) -> np.array:  # pragma: no cover
        """Generate action given state and dynamics model

        Args:
            state (torch.Tensor): State to plan from
            dynamics_model (torch.nn.Module): Dynamics model ( state, action -> nextstate, rewards )
            noise (bool, optional): Whether to add noise to action. Defaults to False.

        Returns:
            np.array: Planned action
        """
        initial_state = state.repeat((self.n_planner, 1)).to(DEVICE)

        mu = np.zeros(self.horizon * self.action_space_size)
        var = 5 * np.ones(self.horizon * self.action_space_size)
        X = scipy.stats.truncnorm(
            self.lb, self.ub, loc=np.zeros_like(mu), scale=np.ones_like(mu)
        )
        i = 0
        while (i < self.iter_update_steps) and (np.max(var) > self.epsilon):
            states = initial_state
            state_list = [initial_state.detach().cpu().numpy()]
            returns = np.zeros((self.n_planner, 1))
            # variables
            lb_dist = mu - self.lb
            ub_dist = self.ub - mu
            constrained_var = np.minimum(
                np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var
            )

            actions = (
                X.rvs(size=[self.n_planner, self.horizon * self.action_space_size])
                * np.sqrt(constrained_var)
                + mu
            )
            actions = np.clip(actions, self.action_low, self.action_high)
            actions_t = (
                torch.from_numpy(
                    actions.reshape(
                        self.n_planner, self.horizon, self.action_space_size
                    )
                )
                .float()
                .to(DEVICE)
            )
            for t in range(self.horizon):
                with torch.no_grad():
                    print(t, states.mean().item())
                    states, rewards = dynamics_model.predict(states, actions_t[:, t, :])
                    state_list.append(states.detach().cpu().numpy())
                returns += rewards.cpu().numpy()

            k_best_rewards, k_best_actions = self._select_k_best(returns, actions)
            mu, var = self._update_gaussians(mu, var, k_best_actions)
            i += 1

        best_action_sequence = mu.reshape(self.horizon, -1)
        best_action = np.copy(best_action_sequence[0])
        assert best_action.shape == (self.action_space_size,)
        return best_action

    def _select_k_best(self, rewards, action_hist):
        """Select k best actions given previous info

        Args:
            rewards (np.array): Rewards array
            action_hist (np.array): Actions array

        Returns:
            tuple: Tuple of k best rewards, and true actions.
        """
        assert rewards.shape == (self.n_planner, 1)
        idxs = np.argsort(rewards, axis=0)

        elite_actions = action_hist[idxs][-self.k_best :, :].squeeze(
            1
        )  # sorted (elite, horizon x action_space)
        k_best_rewards = rewards[idxs][-self.k_best :, :].squeeze(-1)

        assert k_best_rewards.shape == (self.k_best, 1)
        assert elite_actions.shape == (
            self.k_best,
            self.horizon * self.action_space_size,
        )
        return k_best_rewards, elite_actions

    def _update_gaussians(self, old_mu, old_var, best_actions):
        """Update Gaussians given information

        Args:
            old_mu (np.array): Last mu
            old_var (np.array): Last var
            best_actions (np.array): Best actions

        Returns:
            tuple: (new mu, new vars)
        """
        assert best_actions.shape == (
            self.k_best,
            self.horizon * self.action_space_size,
        )

        new_mu = best_actions.mean(0)
        new_var = best_actions.var(0)

        mu = self.update_alpha * old_mu + (1.0 - self.update_alpha) * new_mu
        var = self.update_alpha * old_var + (1.0 - self.update_alpha) * new_var
        assert mu.shape == (self.horizon * self.action_space_size,)
        assert var.shape == (self.horizon * self.action_space_size,)
        return mu, var
