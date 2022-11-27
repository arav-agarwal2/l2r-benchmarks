from abc import ABC
from ast import Not
import numpy as np
import gym


class BasePlanner(ABC):

    def __init__(self, n_planner: int = 500, horizon: int = 12):
        self.n_planner = n_planner
        self.horizon = horizon

    def get_action(self, state) -> np.array:  # pragma: no cover
        """
        # Outputs action given the current state
        obs: A Numpy Array compatible with the PETSAgent and like classes.
        returns:
            action: np.array (2,)
            action should be in the form of [\delta, a], where \delta is the normalized steering angle, and a is the normalized acceleration.
        """
        raise NotImplementedError


