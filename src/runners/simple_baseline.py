import json
import time
import numpy as np
from runners.base import BaseRunner


class SimpleRunner(BaseRunner):
    def __init__(self, env, agent):
        super().__init__()

    def run(self):
        for _ in range(300):
            done = False
            obs, _ = self.env.reset()

            while not done:
                action = self.agent.select_action(obs)
                obs, reward, done, info = self.env.step(action)

    def evaluation(self):
        self.run()
