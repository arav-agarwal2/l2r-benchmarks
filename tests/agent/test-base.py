import numpy as np
import gym
from src.agents.base import BaseAgent


def test_random_agent():
    ra = BaseAgent()
    # Check that default action space is default
    assert ra.action_space == gym.spaces.Box(-1, 1, (2,))
    # Check that assigning an action space works
    ra = BaseAgent(gym.spaces.Discrete(2))
    assert ra.action_space == gym.spaces.Discrete(2)
