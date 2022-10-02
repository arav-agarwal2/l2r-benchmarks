from abc import ABC
from ast import Not
import numpy as np
import gym
import torch
from src.planners.BasePlanner import BasePlanner
from src.constants import DEVICE

class RandomPlanner(BasePlanner):

    def __init__(self, action_space=BasePlanner.default_action_space, n_planner: int = 500, horizon: int = 12):
        super().__init__(action_space, n_planner, horizon)


    def get_action(self, state, dynamics_model: torch.nn.Module, noise: bool = False) -> np.array:  # pragma: no cover
        """
        # Outputs action given the current state
        obs: A Numpy Array compatible with the PETSAgent and like classes.
        returns:
            action: np.array (2,)
            action should be in the form of [\delta, a], where \delta is the normalized steering angle, and a is the normalized acceleration.
        """
        initial_states = state.repeat((self.n_planner, 1)).to(DEVICE)
        rollout_actions =  torch.from_numpy(np.random.uniform(low=self.action_space.low,
                                    high=self.action_space.high,
                                    size=(self.n_planner, self.horizon, self.action_space.shape[0]))).to(DEVICE).float()
        returns, all_states = self._compute_returns(initial_states, rollout_actions, dynamics_model)
        best_action_idx = returns.argmax()
        optimal_action = rollout_actions[:, 0, :][best_action_idx]
        
        
        if noise:
            optimal_action += torch.normal(torch.zeros(optimal_action.shape),
                                           torch.ones(optimal_action.shape) * 0.01).to(DEVICE)


        return optimal_action.cpu().numpy()

    def _compute_returns(self, states, actions, dynamics_model):
        returns = torch.zeros((self.n_planner, 1)).to(DEVICE)
        state_list = [states.detach().cpu().numpy()]
        for t in range(self.horizon):
            with torch.no_grad():
                states, rewards = dynamics_model.predict(states, actions[:, t, :])
                state_list.append(states.detach().cpu().numpy())
            returns += rewards

        return returns, state_list
