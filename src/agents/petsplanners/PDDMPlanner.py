"""PDDM Planner. Largely untested."""
from abc import ABC
from ast import Not
import numpy as np
import gym


import torch
from src.agents.petsplanners.base import BasePlanner
from src.constants import DEVICE
from src.config.yamlize import yamlize


@yamlize
class PDDMPlanner(BasePlanner):
    """Likely an iteration of https://github.com/google-research/pddm."""

    def __init__(
        self,
        action_dim: int = 2,
        action_min: float = -1,
        action_max: float = 1,
        n_planner: int = 500,
        horizon: int = 12,
        gamma: float = 1.0,
        beta: float = 0.5,
    ):
        """Initialize PDDM Planner

        Args:
            action_dim (int, optional): Size of action space. Defaults to 2.
            action_min (float, optional): Minimum action value. Defaults to -1.
            action_max (float, optional): Maximum action value. Defaults to 1.
            n_planner (int, optional): Number of trajectories to plan. Defaults to 500.
            horizon (int, optional): Length to plan into the future. Defaults to 12.
            gamma (float, optional): Gamma. Defaults to 1.0.
            beta (float, optional): Beta. Defaults to 0.5.
        """
        super().__init__(n_planner, horizon)
        self.gamma = gamma
        self.beta = beta
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max

    def get_action(
        self, state, dynamics_model: torch.nn.Module, noise: bool = True
    ) -> np.array:  # pragma: no cover
        """Generate action given state and dynamics model

        Args:
            state (torch.Tensor): State to plan from
            dynamics_model (torch.nn.Module): Dynamics model ( state, action -> nextstate, rewards )
            noise (bool, optional): Whether to add noise to action. Defaults to False.

        Returns:
            np.array: Planned action
        """
        initial_states = state.repeat((self.n_planner, 1)).to(DEVICE)

        actions, returns, state_list = self.get_pred_trajectories(
            initial_states, dynamics_model
        )
        optimal_action = self._update_mu(actions, returns)

        if noise:
            optimal_action += np.random.normal(0, 0.005, size=optimal_action.shape)
        return optimal_action

    def _update_mu(self, action_hist, returns):
        """Update mu given action history and returns

        Args:
            action_hist (np.array): Action history
            returns (np.array): Rewards

        Returns:
            np.array: Mu
        """
        assert action_hist.shape == (self.n_planner, self.horizon, self.action_dim)
        assert returns.shape == (self.n_planner, 1)

        c = np.exp(self.gamma * (returns) - np.max(returns))
        d = np.sum(c) + 1e-10
        assert c.shape == (self.n_planner, 1)
        assert d.shape == (), "Has shape {}".format(d.shape)
        c_expanded = c[:, :, None]
        assert c_expanded.shape == (self.n_planner, 1, 1)
        weighted_actions = c_expanded * action_hist
        self.mu = weighted_actions.sum(0) / d
        assert self.mu.shape == (self.horizon, self.action_dim)

        return self.mu[0]

    def _sample_actions(self, past_action):
        """Sample actions given information

        Args:
            past_action (np.arrau): Last action

        Returns:
            np.array: New actions
        """
        u = np.random.normal(
            loc=0, scale=1.0, size=(self.n_planner, self.horizon, self.action_dim)
        )
        actions = u.copy()
        for t in range(self.horizon):
            if t == 0:
                actions[:, t, :] = (
                    self.beta * (self.mu[t, :] + u[:, t, :])
                    + (1 - self.beta) * past_action
                )
            else:
                actions[:, t, :] = (
                    self.beta * (self.mu[t, :] + u[:, t, :])
                    + (1 - self.beta) * actions[:, t - 1, :]
                )
        assert actions.shape == (
            self.n_planner,
            self.horizon,
            self.action_dim,
        ), "Has shape {} but should have shape {}".format(
            actions.shape, (self.n_planner, self.horizon, self.action_space)
        )
        actions = np.clip(actions, self.action_min, self.action_max)
        return actions

    def _get_pred_trajectories(self, states, model):
        """Predicted trajectories given model and states

        Args:
            states (np.array): States array
            model (nn.Module): Dynamics model

        Returns:
            tuple: Tuple of actions, rewards, and state list
        """
        returns = np.zeros((self.n_planner, 1))
        state_list = [states.detach().cpu().numpy()]
        np.random.seed()
        past_action = self.mu[0].copy()
        actions = self._sample_actions(past_action)
        torch_actions = torch.from_numpy(actions).float().to(DEVICE)
        for t in range(self.horizon):
            with torch.no_grad():
                actions_t = torch_actions[:, t, :]
                assert actions_t.shape == (self.n_planner, self.action_dim)
                states, rewards = model.run_ensemble_prediction(states, actions_t)
                state_list.append(states.detach().cpu().numpy())
            returns += rewards.cpu().numpy()
        return actions, returns, state_list
