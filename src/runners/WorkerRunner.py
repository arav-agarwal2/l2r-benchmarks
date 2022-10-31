import json
import time
from matplotlib.font_manager import json_dump
import numpy as np
import wandb
from src.loggers.WanDBLogger import WanDBLogger
from src.runners.base import BaseRunner
from src.utils.envwrapper import EnvContainer
from src.utils.utils import ActionSample
from src.agents.SACAgent import SACAgent
from src.loggers.TensorboardLogger import TensorboardLogger
from src.loggers.FileLogger import FileLogger

from src.config.parser import read_config
from src.config.schema import agent_schema
from src.config.schema import experiment_schema
from src.config.schema import replay_buffer_schema
from src.config.yamlize import create_configurable, NameToSourcePath, yamlize
from src.config.schema import encoder_schema
from src.constants import DEVICE

from torch.optim import Adam
import torch
import itertools
import jsonpickle


@yamlize
class WorkerRunner(BaseRunner):
    """
    Runner designed for the Worker. All it does is collect data under two scenarios:
      - train, where we include some element of noise
      - test, where we include no such noise.
    """
    def __init__(
        self,
        agent_config_path: str,
        buffer_config_path: str,
        max_episode_length: int
    ):
        super().__init__()
        # Moved initialization of env to run to allow for yamlization of this class.
        # This would allow a common runner for all model-free approaches

        # Initialize runner parameters
        self.agent_config_path = agent_config_path
        self.buffer_config_path = buffer_config_path
        self.max_episode_length = max_episode_length


        ## AGENT Declaration
        self.agent = create_configurable(self.agent_config_path, NameToSourcePath.agent)







    def run(self, env, agent_params):
        """Grab data for system that's needed, and send a buffer accordingly. Note: does a single 'episode'
           which might not be more than a segment in l2r's case.

        Args:
            env (_type_): _description_
            agent (_type_): some agent
        """
        self.agent.load_model(agent_params)
        t = 0
        done = False
        env.reset()

        ep_ret = 0
        self.replay_buffer = create_configurable(
                self.buffer_config_path, NameToSourcePath.buffer
            )
        while not done:
            t += 1
            self.agent.deterministic = False
            action_obj = self.agent.select_action(obs_encoded)
            if self.env_wrapped:
                next_state_encoded, reward, done, info = self.env_wrapped.step(
                    action_obj.action
                )
            else:
                next_state_encoded, reward, done, info = env.step(action_obj.action)

            ep_ret += reward
            self.replay_buffer.store(
                {
                    "obs": obs_encoded,
                    "act": action_obj,
                    "rew": reward,
                    "next_obs": next_state_encoded,
                    "done": done,
                }
            )
            if done or t == self.max_episode_length:
                self.replay_buffer.finish_path(action_obj)

            obs_encoded = next_state_encoded
        from copy import deepcopy
        return deepcopy(self.replay_buffer), ep_ret


    def eval(self, env):
        print("Evaluation:")
        val_ep_rets = []

        # Not implemented for logging multiple test episodes
        # assert self.cfg["num_test_episodes"] == 1
        env.reset()

        eval_done = False
        eval_ep_len = 0

        while (not eval_done) & (eval_ep_len <= self.max_episode_length):
            # Take deterministic actions at test time
            self.agent.deterministic = True
            self.t = 1e6
            eval_action_obj = self.agent.select_action(
                eval_obs_encoded
            )
            if self.env_wrapped:
                (
                    eval_obs_encoded_new,
                    eval_reward,
                    eval_done,
                    eval_info,
                ) = self.env_wrapped.step(eval_action_obj.action)
            else:
                eval_obs_encoded_new, eval_reward, eval_done, eval_info = env.step(
                    eval_action_obj.action
                )

            # Check that the camera is turned on
            eval_ep_ret += eval_reward
            eval_ep_len += 1
            eval_n_val_steps += 1


            eval_obs_encoded = eval_obs_encoded_new
            t_eval += 1



        return eval_info, eval_ep_ret
