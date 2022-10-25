import numpy as np
import torch
import itertools
from src.constants import DEVICE


class EnvContainer:
    """Container for L2R Environment."""

    def __init__(self, encoder=None, env=None):
        self.encoder = encoder
        self.env = env
        self.is_async = False

    def _process_obs(self, obs: dict):
        obs_camera = obs["images"]["CameraFrontRGB"]
        obs_encoded = self.encoder.encode(obs_camera).to(DEVICE)
        speed = (
            torch.tensor(np.linalg.norm(obs["pose"][3:6], ord=2))
            .to(DEVICE)
            .reshape((-1, 1))
            .float()
        )
        return torch.cat((obs_encoded, speed), 1).to(DEVICE).detach().numpy()

    def step(self, action, env=None):
        import logging
        logging.warn((action, type(action), action.shape))
        action = action.reshape((2,))
        if env:
            self.env = env
        obs, reward, terminated, info = self.env.step(action)
        info["TimeLimit.truncated"] = False # Tianshou's making me add this; need to check TODO
        logging.warn(info)
        return self._process_obs(obs), reward, np.array(terminated), info

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

    def __len__(self):
        return 1

    def __getattr__(self, name):
        try:
            return getattr(self.env,name)
        except Exception as e:
            raise e