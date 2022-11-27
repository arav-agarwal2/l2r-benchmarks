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
class RandomPlanner(BasePlanner):
    """Random Shooting-Based Planner."""

    def __init__(
        self,
        action_dim: int = 2,
        action_min: float = -1,
        action_max: float = 1,
        n_planner: int = 500,
        horizon: int = 12,
    ):
        """Initialize Random Shooting Planner

        Args:
            action_dim (int, optional): Dimension of action space. Defaults to 2.
            action_min (float, optional): Minimum action. Defaults to -1.
            action_max (float, optional): Maximum action. Defaults to 1.
            n_planner (int, optional): Number of trajectories to test. Defaults to 500.
            horizon (int, optional): Planning horizon. Defaults to 12.
        """
        super().__init__(n_planner, horizon)
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max

    def get_action(
        self, state, dynamics_model: torch.nn.Module, noise: bool = False
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
        rollout_actions = (
            torch.from_numpy(
                np.random.uniform(
                    low=self.action_min,
                    high=self.action_max,
                    size=(self.n_planner, self.horizon, self.action_dim),
                )
            )
            .to(DEVICE)
            .float()
        )
        returns, all_states = self._compute_returns(
            initial_states, rollout_actions, dynamics_model
        )
        best_action_idx = returns.argmax()
        optimal_action = rollout_actions[:, 0, :][best_action_idx]

        if noise:
            optimal_action += torch.normal(
                torch.zeros(optimal_action.shape),
                torch.ones(optimal_action.shape) * 0.01,
            ).to(DEVICE)

        return optimal_action.cpu().numpy()

    def _compute_returns(self, states, actions, dynamics_model):
        """Compute rewards, states from initial states, actions

        Args:
            states (torch.Tensor): Initial states
            actions (torch.Tensor): Actions
            dynamics_model (nn.Module): Dynamics model

        Returns:
            tuple: Tuple of returns, states
        """
        returns = torch.zeros((self.n_planner, 1)).to(DEVICE)
        state_list = [states.detach().cpu().numpy()]
        for t in range(self.horizon):
            with torch.no_grad():
                states, rewards = dynamics_model.predict(states, actions[:, t, :])
                state_list.append(states.detach().cpu().numpy())
            returns += rewards

        return returns, state_list
