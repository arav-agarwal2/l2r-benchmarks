import numpy as np
import torch
from constants import DEVICE

class EnvContainer():
    """Container for Environment and Encoder. """

    def __init__(self, env, encoder):
        self.env = env
        self.encoder = encoder

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs[1], self.encode(obs), obs[0], reward, done, info

    def reset(self, random_pos=False):
        camera = 0
        while (np.mean(camera) == 0) | (np.mean(camera) == 255):
            obs = self.env.reset(random_pos=random_pos)
            (state, camera), _ = obs
        return camera, self.encode((state, camera)), state, False, ""

    def encode(self, o):
        state, img = o

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
