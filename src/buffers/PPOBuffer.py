"""Buffer for PPO."""
from src.config.yamlize import yamlize
import torch
import numpy as np
import scipy
from scipy import signal
from src.utils.utils import ActionSample

DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def discount_cumsum(x, discount):
    """Wraper arround discounted cumulative summation.

    Args:
        x (np.array): Array to sum over
        discount (float): discount parameter.

    Returns:
        float: output
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


@yamlize
class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        size: int,
        batch_size: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        eps: float = 1e-3,
    ):
        """Initialize PPOBuffer

        Args:
            obs_dim (int): Observation Dimension
            act_dim (int): Action Dimension
            size (int): Size of Replay Buffer
            batch_size (int): Batch Size
            gamma (float, optional): Gamma. Defaults to 0.99.
            lam (float, optional): Lambda. Defaults to 0.95.
            eps (_type_, optional): Epsilon. Defaults to 1e-3.
        """
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.size = 0
        self.batch_size = batch_size
        self.eps = eps

    def store(self, buffer_dict):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        self.obs_buf[self.ptr] = buffer_dict["obs"].detach().cpu()
        self.act_buf[self.ptr] = buffer_dict["act"].action
        self.rew_buf[self.ptr] = buffer_dict["rew"]
        self.val_buf[self.ptr] = buffer_dict["act"].value
        self.logp_buf[self.ptr] = buffer_dict["act"].logp
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

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
        if self.path_start_idx > self.ptr:
            path_slice1 = slice(self.path_start_idx, self.size)
            path_slice2 = slice(0, self.ptr)
            rews = np.append(
                self.rew_buf[path_slice1] + self.val_buf[path_slice2], action_obj.value
            )
            vals = np.append(
                self.val_buf[path_slice1] + self.val_buf[path_slice2], action_obj.value
            )
        else:
            path_slice = slice(self.path_start_idx, self.ptr)
            rews = np.append(self.rew_buf[path_slice], action_obj.value)
            vals = np.append(self.val_buf[path_slice], action_obj.value)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def sample_batch(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # assert self.ptr == self.max_size    # buffer has to be full before you can get
        # self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick

        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + self.eps)

        idxs = np.random.choice(
            self.size, size=min(self.batch_size, self.size), replace=False
        )
        data = dict(
            obs=self.obs_buf[idxs],
            act=self.act_buf[idxs],
            ret=self.rew_buf[idxs],
            adv=self.adv_buf[idxs],
            logp=self.logp_buf[idxs],
        )

        self.weights = torch.tensor(
            np.zeros_like(idxs), dtype=torch.float32, device=DEVICE
        )

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
