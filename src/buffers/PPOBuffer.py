from src.config.yamlize import yamlize
import torch
import numpy as np
import scipy
from scipy import signal
import pdb
from src.utils.utils import ActionSample

DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def stats_scalar(x):
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = np.sum(x), len(x)
    mean = global_sum / global_n

    global_sum_sq = np.sum((x - mean)**2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    return mean, std

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

@yamlize
class PPOBuffer:
    '''
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    '''

    def __init__(self, obs_dim: int, act_dim: int, size: int, batch_size: int, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros( (size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros( (size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.batch_size = batch_size

    def store(self, buffer_dict):
        '''
        Append one timestep of agent-environment interaction to the buffer.
        '''
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = buffer_dict["obs"].detach().cpu()
        self.act_buf[self.ptr] = buffer_dict["act"].action
        self.rew_buf[self.ptr] = buffer_dict["rew"]
        self.val_buf[self.ptr] = buffer_dict["act"].value
        self.logp_buf[self.ptr] = buffer_dict["act"].logp
        self.ptr += 1

    def finish_path(self, action_obj=None):
        '''
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
        '''

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], action_obj.value)
        vals = np.append(self.val_buf[path_slice], action_obj.value)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        if np.isnan(self.adv_buf).any():
            pdb.set_trace()
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def sample_batch(self):
        '''
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        '''
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        if np.isnan(self.adv_buf).any():
            pdb.set_trace()
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = stats_scalar(self.adv_buf)
        if np.isnan(adv_std):
            pdb.set_trace()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

