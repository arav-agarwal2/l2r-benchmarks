import itertools
from multiprocessing.sharedctypes import Value
import queue, threading
from copy import deepcopy
from src.loggers.FileLogger import FileLogger

import torch
import numpy as np
from gym.spaces import Box
from torch.optim import Adam

from src.agents.base import BaseAgent
from src.config.yamlize import yamlize
from src.deprecated.network import ActorCritic, CriticType
from src.encoders.VAE import VAE
from src.utils.utils import RecordExperience

from src.constants import DEVICE

from src.config.parser import read_config
from src.config.schema import agent_schema

from src.utils.envwrapper import EnvContainer


@yamlize
class PPOAgent(BaseAgent):
    def __init__(
        self, 
        steps_to_sample_randomly: int,
        record_dir: str,
        track_name: str,
        experiment_name: str,
        gamma: float,
        alpha: float,
        polyak: float,
        make_random_actions: bool,
        checkpoint: str,
        load_checkpoint: bool,
        model_save_path: str,
        lr: float,
        clip_ratio: float
    ):
        super(PPOAgent, self).__init__()
        self.steps_to_sample_randomly = steps_to_sample_randomly
        self.record_dir = record_dir
        self.track_name = track_name
        self.experiment_name = experiment_name
        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak
        self.make_random_actions = make_random_actions
        self.checkpoint = checkpoint
        self.load_checkpoint = load_checkpoint
        self.model_save_path = model_save_path
        self.lr = lr
        self.clip_ratio = clip_ratio

        self.save_episodes = True
        self.episode_num = 0
        self.best_ret = 0
        self.t = 0
        self.deterministic = False
        self.atol = 1e-3
        self.store_from_safe = False
        self.pi_scheduler = None
        self.t_start = 0
        self.best_pct = 0

        self.metadata = {}
        self.record = {"transition_actor": ""}

        self.action_space = Box(-1, 1, (2,))
        self.act_dim = self.action_space.shape[0]
        self.obs_dim = 32

        self.actor_critic = ActorCritic(
            self.obs_dim,
            self.action_space,
            None,
            latent_dims=self.obs_dim,
            device=DEVICE,
            critic_type=CriticType.Value
        )

        if self.checkpoint and self.load_checkpoint:
            self.load_model(self.checkpoint)

        self.actor_critic_target = deepcopy(self.actor_critic)

        self.v_params = itertools.chain(
            self.actor_critic.v.parameters()
        )

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(
            self.actor_critic.policy.parameters(), lr=self.lr
        )
        self.v_optimizer = Adam(self.v_params, lr=self.lr)
        self.pi_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pi_optimizer, 1, gamma=0.5
        )


    def select_action(self, obs) -> np.array: 
        if self.t > self.steps_to_sample_randomly:
            a = self.actor_critic.act(obs.to(DEVICE), self.deterministic)
            a = a  # numpy array...
            self.record["transition_actor"] = "learner"
        else:
            a = self.action_space.sample()
            self.record["transition_actor"] = "random"
        self.t = self.t + 1
        return a

    def register_reset(self, obs) -> np.array:  
        self.deterministic = True
        self.t = 1e6

    def compute_loss_pi(self,data):

        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.actor_critic.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self,data):
        ## Check this.
        obs, ret = data['obs'], data['rew']
        file_logger = FileLogger(
            self.model_save_path, "log_dir_test"
        )
        file_logger.log(f"Data: {data}")
        return ((self.actor_critic.v(obs) - ret)**2).mean()

    def update(self, data): 
        
        # First run one gradient descent step for Q1 and Q2
        self.v_optimizer.zero_grad()
        loss_v, v_info = self.compute_loss_v(data)
        loss_v.backward()
        self.v_optimizer.step()

         # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.v_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next step.
        for p in self.v_params:
            p.requires_grad = True

    def load_model(self, path):
        self.actor_critic.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.actor_critic.state_dict(), path)