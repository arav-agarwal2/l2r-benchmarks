from abc import ABC
from ast import Not
import numpy as np
import gym


import torch
from src.planners.BasePlanner import BasePlanner
from src.constants import DEVICE

class PDDMPlanner(BasePlanner):

    def __init__(self, action_space=BasePlanner.default_action_space, n_planner: int = 500, horizon: int = 12, gamma: float = 1.0, beta: float = 0.5):
        super().__init__( action_space, n_planner, horizon)
        self.gamma = gamma
        self.beta = beta

    def get_action(self, state, dynamics_model: torch.nn.Module, noise: bool = True) -> np.array:  # pragma: no cover
        """
        # Outputs action given the current state
        obs: A Numpy Array compatible with the PETSAgent and like classes.
        returns:
            action: np.array (2,)
            action should be in the form of [\delta, a], where \delta is the normalized steering angle, and a is the normalized acceleration.
        """
        initial_states = state.repeat((self.n_planner, 1)).to(DEVICE)
            
        actions, returns, state_list = self.get_pred_trajectories(initial_states, dynamics_model)
        optimal_action = self._update_mu(actions, returns)

        if noise:
            optimal_action += np.random.normal(0, 0.005, size=optimal_action.shape)
        return optimal_action


    def _update_mu(self, action_hist, returns):
        assert action_hist.shape == (self.n_planner, self.horizon, self.action_space)
        assert returns.shape == (self.n_planner, 1)

        c = np.exp(self.gamma * (returns) -np.max(returns))
        d = np.sum(c) + 1e-10
        assert c.shape == (self.n_planner, 1)
        assert d.shape == (), "Has shape {}".format(d.shape)
        c_expanded = c[:, :, None]
        assert c_expanded.shape == (self.n_planner, 1, 1)
        weighted_actions = c_expanded * action_hist
        self.mu = weighted_actions.sum(0) / d
        assert self.mu.shape == (self.horizon, self.action_space)       
        
        return self.mu[0]
    
    def _sample_actions(self, past_action):
        u = np.random.normal(loc=0, scale=1.0, size=(self.n_planner, self.horizon, self.action_space))
        actions = u.copy()
        for t in range(self.horizon):
            if t == 0:
                actions[:, t, :] = self.beta * (self.mu[t, :] + u[:, t, :]) + (1 - self.beta) * past_action
            else:
                actions[:, t, :] = self.beta * (self.mu[t, :] + u[:, t, :]) + (1 - self.beta) * actions[:, t-1, :]
        assert actions.shape == (self.n_planner, self.horizon, self.action_space), "Has shape {} but should have shape {}".format(actions.shape, (self.n_planner, self.horizon, self.action_space))
        actions = np.clip(actions, self.action_low, self.action_high)
        return actions
    
    def _get_pred_trajectories(self, states, model): 
        returns = np.zeros((self.n_planner, 1))
        state_list = [states.detach().cpu().numpy()]
        np.random.seed()
        past_action = self.mu[0].copy()
        actions = self._sample_actions(past_action)
        torch_actions = torch.from_numpy(actions).float().to(DEVICE)
        for t in range(self.horizon):
            with torch.no_grad():
                actions_t = torch_actions[:, t, :]
                assert actions_t.shape == (self.n_planner, self.action_space)
                states, rewards = model.run_ensemble_prediction(states, actions_t)
                state_list.append(states.detach().cpu().numpy())
            returns += rewards.cpu().numpy()
        return actions, returns, state_list
