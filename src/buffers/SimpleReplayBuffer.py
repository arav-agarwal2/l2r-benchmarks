"""Default Replay Buffer."""
import collections
import torch
import numpy as np
from typing import Tuple

from src.config.yamlize import yamlize
from src.constants import DEVICE


@yamlize
class SimpleReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int, batch_size: int):
        self.max_size = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.weights = None

    def __len__(self):
        return len(self.buffer)

    def store(self, buffer_dict):
        """Store data from buffer_dict

        Args:
            buffer_dict (_type_): Buffer dict
        """

        def convert(arraylike):
            """Convert from tensor to nparray

            Args:
                arraylike (Torch.Tensor): Tensor to convert

            Returns:
                np.array: Converted numpyarray
            """
            obs = arraylike
            if isinstance(obs, torch.Tensor):
                if obs.requires_grad:
                    obs = obs.detach()
                obs = obs.cpu().numpy()
            return obs

        self.obs_buf[self.ptr] = convert(buffer_dict["obs"])
        self.obs2_buf[self.ptr] = convert(buffer_dict["next_obs"])
        self.act_buf[self.ptr] = buffer_dict["act"].action  # .detach().cpu().numpy()
        self.rew_buf[self.ptr] = buffer_dict["rew"]
        self.done_buf[self.ptr] = buffer_dict["done"]
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        """Sample batch from self.

        Returns:
            dict: Dictionary of batched information.
        """

        idxs = np.random.choice(
            self.size, size=min(self.batch_size, self.size), replace=False
        )
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        self.weights = torch.tensor(
            np.zeros_like(idxs), dtype=torch.float32, device=DEVICE
        )
        return {
            k: torch.tensor(v, dtype=torch.float32, device=DEVICE)
            for k, v in batch.items()
        }

    def finish_path(self, action_obj=None):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        pass