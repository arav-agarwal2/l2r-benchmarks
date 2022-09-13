import itertools
from multiprocessing.sharedctypes import Value
import queue, threading
from copy import deepcopy

import torch
import numpy as np
from gym.spaces import Box
from torch.optim import Adam

from src.agents.base import BaseAgent
from src.config.yamlize import yamlize
from src.deprecated.network import ActorCritic
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

        self.actor_critic = ActorCritic(
            self.obs_dim,
            self.action_space,
            None,
            latent_dims=self.obs_dim,
            device=DEVICE,
        )



    def select_action(self, obs) -> np.array: 
        
        raise NotImplementedError

    def register_reset(self, obs) -> np.array:  
        raise NotImplementedError

    def update(self, data): 
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError