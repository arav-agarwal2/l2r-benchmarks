"""PPOAgent Definition. """
import itertools
from copy import deepcopy

import torch
import numpy as np
from gym.spaces import Box
from torch.optim import Adam

from src.agents.base import BaseAgent
from src.config.yamlize import yamlize, create_configurable, NameToSourcePath
from src.utils.utils import ActionSample

from src.constants import DEVICE


@yamlize
class PPOAgent(BaseAgent):
    """Proximal Policy Optimization Agent"""

    def __init__(
        self,
        steps_to_sample_randomly: int,
        lr: float,
        clip_ratio: float,
        load_checkpoint_from: str = "",
        train_pi_iters: int = 80,
        train_v_iters: int = 80,
        target_kl: float = 0.01,
        actor_critic_cfg_path: str = "",
    ):
        """Initialize Proximal Policy Optimization Agent

        Args:
            steps_to_sample_randomly (int): Number of steps to sample randomly
            lr (float): Learning rate
            clip_ratio (float): Clip ratio
            load_checkpoint_from (str, optional): Where to load checkpoint from. Using default does not load any checkpoint. Defaults to ''.
            train_pi_iters (int, optional): Number of update iterations for policy per call to `update`. Defaults to 80.
            train_v_iters (int, optional): Number of update iterations for value per call to `update`. Defaults to 80.
            target_kl (float, optional): Target Kubler-Leibleck Divergence. Defaults to 0.01.
            actor_critic_cfg_path (str, optional): Path to AC cfg. Defaults to ''.
        """
        super(PPOAgent, self).__init__()
        self.steps_to_sample_randomly = steps_to_sample_randomly
        self.load_checkpoint_from = load_checkpoint_from
        self.lr = lr
        self.clip_ratio = clip_ratio

        self.t = 0
        self.deterministic = False  # TODO: Fix.
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters

        self.record = {"transition_actor": ""}

        self.action_space = Box(-1, 1, (2,))
        self.act_dim = self.action_space.shape[0]
        self.obs_dim = 32

        self.actor_critic = create_configurable(
            actor_critic_cfg_path, NameToSourcePath.network
        )
        self.actor_critic.to(DEVICE)

        self.target_kl = target_kl

        if self.load_checkpoint_from != "":
            self.load_model(self.load_checkpoint_from)

        self.actor_critic_target = deepcopy(self.actor_critic)

        self.v_params = itertools.chain(self.actor_critic.v.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.actor_critic.policy.parameters(), lr=self.lr)
        self.v_optimizer = Adam(self.v_params, lr=self.lr)
        self.pi_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pi_optimizer, 1, gamma=0.5
        )

    def select_action(self, obs) -> np.array:
        """Select action given observation array.

        Args:
            obs (np.array): Observation array

        Returns:
            np.array: Action array
        """
        action_obj = ActionSample()
        if self.t > self.steps_to_sample_randomly:
            a, logp = self.actor_critic.pi(obs.to(DEVICE), self.deterministic)
            action_obj.action = a.squeeze().detach().cpu().numpy()
            action_obj.value = (
                self.actor_critic.v(obs.to(DEVICE)).detach().cpu().numpy()
            )
            action_obj.logp = logp.squeeze().detach().cpu().numpy()
            self.record["transition_actor"] = "learner"
        else:
            a = self.action_space.sample()
            # logp = np.ones((self.action_space.shape[0], ))/self.action_space.shape[0]
            logp = np.ones((1,))
            # TODO: add default value after getting value shape
            v = np.ones((1,))
            action_obj.action = a
            action_obj.value = v
            action_obj.logp = logp
            self.record["transition_actor"] = "random"
        self.t = self.t + 1
        return action_obj

    def register_reset(self, obs):
        """Handle reset of episode."""
        pass

    def _compute_loss_pi(self, data):
        """Compute policy loss.

        Args:
            data (dict): dictionary of data to calculate loss from.

        Returns:
            loss_pi, pi_info: loss information.
        """
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        # Policy loss
        pi, logp = self.actor_critic.pi(obs.to(DEVICE))
        # logp = logp.cpu().numpy()
        # pi = pi.cpu()
        logp_old = logp_old.to(DEVICE)
        adv = adv.to(DEVICE)
        ratio = torch.exp(logp - logp_old)
        if ratio.isnan().any().item():
            print("ratio is nan ---")
            print(ratio)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        # print("entropy", pi.entropy)
        # ent = pi.entropy.mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        # pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        pi_info = dict(kl=approx_kl, cf=clipfrac)
        return loss_pi, pi_info

    def _compute_loss_v(self, data):
        """Compute value loss.

        Args:
            data (dict): dictionary of data to calculate from.

        Returns:
            loss_q: loss information.
        """
        ## Check this.
        obs, ret = data["obs"], data["ret"]
        ret = ret.to(DEVICE)
        return ((self.actor_critic.v(obs.to(DEVICE)) - ret) ** 2).mean()

    def update(self, data):
        """Update parameters given batch of data.

        Args:
            data (dict): Dict of batched data to update params from.
        """

        pi_l_old, pi_info_old = self._compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self._compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self._compute_loss_pi(data)
            kl = pi_info["kl"]
            if kl > 1.5 * self.target_kl:
                # print(next(self.actor_critic.pi.mu_net.parameters()))
                # self.file_logger('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            self.pi_optimizer.step()
        # print(next(self.actor_critic.pi.mu_net.parameters()))
        # logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_v_iters):
            self.v_optimizer.zero_grad()
            loss_v = self._compute_loss_v(data)
            loss_v.backward()
            self.v_optimizer.step()

    def load_model(self, path):
        """Load model from path

        Args:
            path (str): Load path using str
        """
        self.actor_critic.load_state_dict(torch.load(path))

    def save_model(self, path):
        """Save model to path

        Args:
            path (str): Save path using str
        """
        torch.save(self.actor_critic.state_dict(), path)
