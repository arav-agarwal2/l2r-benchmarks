import logging
import subprocess
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from gym import Wrapper
import gym

from tianshou.data import ReplayBuffer
from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.utils.net.common import Net
from tianshou.env import DummyVectorEnv

from distrib_l2r.api import BufferMsg
from distrib_l2r.api import EvalResultsMsg
from distrib_l2r.api import InitMsg
from distrib_l2r.utils import send_data

from l2r import build_env

from src.config.yamlize import create_configurable, NameToSourcePath, yamlize
from src.constants import DEVICE
from src.utils.envwrapper import EnvContainer
import numpy as np


class AsnycWorker:
    """An asynchronous worker"""

    def __init__(
        self,
        learner_address: Tuple[str, int],
        buffer_size: int = 5000,
        env_wrapper: Optional[Wrapper] = None,
        **kwargs,
    ) -> None:

        self.learner_address = learner_address
        self.buffer_size = buffer_size
        self.mean_reward = 0.0

        self.env = build_env(controller_kwargs={"quiet": True},
           env_kwargs=
                   {
                       "multimodal": True,
                       "eval_mode": True,
                       "n_eval_laps": 5,
                       "max_timesteps": 5000,
                       "obs_delay": 0.1,
                       "not_moving_timeout": 50000,
                       "reward_pol": "custom",
                       "provide_waypoints": False,
                       "active_sensors": [
                           "CameraFrontRGB"
                       ],
                       "vehicle_params":False,
                   },
           action_cfg=
                   {
                       "ip": "0.0.0.0",
                       "port": 7077,
                       "max_steer": 0.3,
                       "min_steer": -0.3,
                       "max_accel": 6.0,
                       "min_accel": -1,
                   },
            camera_cfg=[
                {
                    "name": "CameraFrontRGB",
                    "Addr": "tcp://0.0.0.0:8008",
                    "Width": 512,
                    "Height": 384,
                    "sim_addr": "tcp://0.0.0.0:8008",
                }
            ]
                   )

        self.encoder = create_configurable(
            "config_files/async_sac/encoder.yaml", NameToSourcePath.encoder
        )
        self.encoder.to(DEVICE)

        self.env.action_space = gym.spaces.Box(np.array([-1, -1]), np.array([1.0, 1.0]))
        self.env = EnvContainer(self.encoder, self.env)

        self.runner = create_configurable(
            "config_files/async_sac/worker.yaml", NameToSourcePath.runner
        )
        # print(self.env.action_space)

    def work(self) -> None:
        """Continously collect data"""

        is_train = True
        logging.warn("Trying to send data.")
        response = send_data(data=InitMsg(), addr=self.learner_address, reply=True)
        policy_id, policy = response.data["policy_id"], response.data["policy"]

        while True:
            buffer, result = self.collect_data(policy_weights=policy, is_train=is_train)
            logging.warn("Data collection finished! Sending.")

            if is_train:
                response = send_data(
                    data=BufferMsg(data=buffer), addr=self.learner_address, reply=True
                )
                logging.warn("Sent!")

            else:
                self.mean_reward = self.mean_reward * (0.2) + result["reward"] * 0.8
                logging.warn(f"reward: {self.mean_reward}")
                response = send_data(
                    data=EvalResultsMsg(data=result),
                    addr=self.learner_address,
                    reply=True,
                )
                logging.warn("Sent!")

            is_train = response.data["is_train"]
            policy_id, policy = response.data["policy_id"], response.data["policy"]

    def collect_data(
        self, policy_weights: dict, is_train: bool = True
    ) -> Tuple[ReplayBuffer, Any]:
        """Collect 1 episode of data in the environment"""
        
        buffer, result = self.runner.run(self.env, policy_weights, is_train)

        return buffer, result