from abc import ABC
from ast import Not
import numpy as np
import gym


class BasePlanner(ABC):
    def __init__(self, n_planner: int = 500, horizon: int = 12):
        self.n_planner = n_planner
        self.horizon = horizon

    def get_action(self, state) -> np.array:  # pragma: no cover
        """Generate action given state and dynamics model

        Args:
            state (torch.Tensor): State to plan from
            dynamics_model (torch.nn.Module): Dynamics model ( state, action -> nextstate, rewards )
            noise (bool, optional): Whether to add noise to action. Defaults to False.

        Returns:
            np.array: Planned action
        """

        raise NotImplementedError
