import numpy as np
import torch
import itertools
from src.constants import DEVICE


class EnvContainer:
    """Container for L2R Environment."""

    def __init__(self, encoder=None, env=None):
        self.encoder = encoder
        self.env = env

    def _process_obs(self, obs: dict):
        obs_camera = obs["images"]["CameraFrontRGB"]
        obs_encoded = self.encoder.encode(obs_camera).to(DEVICE)
        speed = (
            torch.tensor(np.linalg.norm(obs["pose"][3:6], ord=2))
            .to(DEVICE)
            .reshape((-1, 1))
            .float()
        )
        return torch.cat((obs_encoded, speed), 1).to(DEVICE)

    def step(self, action, env=None):
        if env:
            self.env = env
        obs, reward, done, info = self.env.step(action)
        return self._process_obs(obs), reward, done, info

    def reset(self, random_pos=False, env=None):
        if env:
            self.env = env
        obs = self.env.reset(random_pos=random_pos)
        return self._process_obs(obs)

    def reset_episode(self, t):
        camera, feat, state, _, _ = self.reset(random_pos=True)
        ep_ret, ep_len, experience = 0, 0, []
        t_start = t + 1
        camera, feat, _, _, _, _ = self.step([0, 1])
        return camera, ep_len, ep_ret, experience, feat, state, t_start
