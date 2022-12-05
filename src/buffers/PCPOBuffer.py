"""Default Replay Buffer."""
import torch
import numpy as np
from typing import Tuple
from src.config.yamlize import yamlize
from src.constants import DEVICE
import scipy

@yamlize
class SimpleReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int, batch_size: int):
        """Initialize simple replay buffer

        Args:
            obs_dim (int): Observation dimension
            act_dim (int): Action dimension
            size (int): Buffer size
            batch_size (int): Batch size
        """

        self.obs_buf = np.zeros(
            (size, obs_dim), dtype=np.float32
        )  # +1:spd #core.combined_shape(size, obs_dim)
        self.act_buf = np.zeros(
            (size, act_dim), dtype=np.float32
        )  # core.combined_shape(size, act_dim)
        self.discounted_ret_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.target_val_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.cost_val_buf = np.zeros(size, dtype=np.float32)
        self.cost_adv_buf = np.zeros(size, dtype=np.float32)
        self.target_cost_val_buf = np.zeros(size, dtype=np.float32)
        
        self.ptr, self.size, self.max_size = 0, 0, size
        self.batch_size = batch_size
        self.weights = None

    def store(self, buffer_dict):
        """Store data from buffer_dict

        Args:
            buffer_dict (_type_): Buffer dict
        """
        # pdb.set_trace()

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
        self.act_buf[self.ptr] = buffer_dict["act"].action
        self.rew_buf[self.ptr] = buffer_dict["rew"]
        self.val_buf[self.ptr] = buffer_dict["act"].value
        self.logp_buf[self.ptr] = buffer_dict["act"].logp
        self.cost_buf[self.ptr] = buffer_dict["cost"]
        self.cost_val_buf[self.ptr] = buffer_dict["act"].cost_value

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        """Sample batch from self.

        Returns:
            dict: Dictionary of batched information.
        """

        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        batch  = dict(
            obs=self.obs_buf, act=self.act_buf, target_v=self.target_val_buf,
            adv=self.adv_buf, log_p=self.logp_buf,
            discounted_ret=self.discounted_ret_buf,
            cost_adv=self.cost_adv_buf, target_c=self.target_cost_val_buf,
        )
        
        return {
            k: torch.tensor(v, dtype=torch.float32, device=DEVICE)
            for k, v in batch.items()
        }
    
    def discount_cumsum(x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    
    def calculate_adv_and_value_targets(self, vals, rews, lam=None):
        """ Compute the estimated advantage"""

        # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
        lam = self.lam if lam is None else lam
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = self.discount_cumsum(deltas, self.gamma * lam)
        value_net_targets = adv + vals[:-1]
        return adv, value_net_targets

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

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], action_obj.value)
        vals = np.append(self.val_buf[path_slice], action_obj.value)
        costs = np.append(self.cost_buf[path_slice], action_obj.cost)
        cost_vs = np.append(self.cost_val_buf[path_slice], action_obj.cost)

        discounted_ret = self.discount_cumsum(rews, self.gamma)[:-1]
        self.discounted_ret_buf[path_slice] = discounted_ret

        adv, v_targets = self.calculate_adv_and_value_targets(vals, rews)
        self.adv_buf[path_slice] = adv
        self.target_val_buf[path_slice] = v_targets

        # calculate costs
        c_adv, c_targets = self.calculate_adv_and_value_targets(cost_vs, costs,lam=self.lam_c)
        self.cost_adv_buf[path_slice] = c_adv
        self.target_cost_val_buf[path_slice] = c_targets

        self.path_start_idx = self.ptr