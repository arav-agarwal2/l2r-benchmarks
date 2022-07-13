import numpy as np
import gym
from src.agents.random_agent import RandomAgent

def test_random_agent():
    ra = RandomAgent()
    # Just check that it randomly samples well for now
    assert np.linalg.norm(ra.select_action(np.zeros(0))) <= np.sqrt(2)

    # Check that assigning another space works
    ra = RandomAgent(action_space=gym.spaces.Discrete(2))
    action = ra.select_action(np.zeros(0))
    assert action in [0,1]

    # TODO: Add some nicer test.