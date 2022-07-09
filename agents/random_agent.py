import json
import time
import numpy as np
from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    """Randomly pick actions in the space. """
    
    def __init__(self):
        super().__init__()

    def select_action(self, obs) -> np.array:
        return self.action_space.sample()

    def register_reset(self, obs) -> np.array:
        pass

    def load_model(self, path):
        pass

    def save_model(self, path):
        pass
