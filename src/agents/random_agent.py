"""Simple Random Agent."""
import json
import time
import numpy as np
from src.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    """Randomly pick actions in the space."""

    def select_action(self, obs) -> np.array:
        """Selection action through random sampling.

        Args:
            obs (np.array): Observation (unused)

        Returns:
            np.array: Action
        """
        return self.action_space.sample()
