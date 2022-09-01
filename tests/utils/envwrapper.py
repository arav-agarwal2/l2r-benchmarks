from src.utils.envwrapper import EnvContainer
import gym
import numpy as np


def test_container():
    env = gym.make("MountainCar-v0")
    envwrapped = EnvContainer(env, "MountainCar")
    a, b, c, d, e = envwrapped.reset()
    a, b, c, d, e, f = envwrapped.step(1)
