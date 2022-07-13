import numpy as np
import torch
import itertools
from src.constants import DEVICE

class EnvContainer():
    """Container for Environment and Encoder. """

    def __init__(self, env, env_type, encoder=None):
        self.env = env
        self.encoder = encoder
        self.env_type = env_type

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs[1], self.encode(obs), obs[0], reward, done, info

    def reset(self, random_pos=False):
        camera = 0
        if self.env_type == 'roborace':
            while (np.mean(camera) == 0) | (np.mean(camera) == 255):
                obs = self.env.reset(random_pos=random_pos)
                (state, camera), _ = obs
        else:
            obs = self.env.reset()
            return obs, obs, 0, False, ""
        return camera, self.encode((state, camera)), state, False, ""

    def encode(self, o):
        state, img = o

        if self.encoder is None:
            return o

        if self.cfg["use_encoder_type"] == "vae":
            img_embed = self.encoder.encode_raw(np.array(img), DEVICE)[0][0]
            speed = (
                torch.tensor((state[4] ** 2 + state[3] ** 2 + state[5] ** 2) ** 0.5)
                .float()
                .reshape(1, -1)
                .to(DEVICE)
            )
            out = torch.cat([img_embed.unsqueeze(0), speed], dim=-1).squeeze(0)
            self.using_speed = 1
        else:
            raise NotImplementedError

        assert not torch.sum(torch.isnan(out)), "found a nan value"
        out[torch.isnan(out)] = 0

        return out
    
    def reset_episode(self, t):
        camera, feat, state, _, _ = self.reset(random_pos=True)
        ep_ret, ep_len, experience = 0, 0, []
        t_start = t + 1
        camera, feat, _, _, _, _ = self.step([0, 1])
        return camera, ep_len, ep_ret, experience, feat, state, t_start

