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


# pip install git+https://github.com/learn-to-race/l2r.git@aicrowd-environment
from l2r import build_env
from l2r import RacingEnv

from src.config.yamlize import create_configurable, NameToSourcePath, yamlize
from src.constants import DEVICE
from src.utils.envwrapper import EnvContainer
import numpy as np

class AsnycWorker:
    """An asynchronous worker"""

    def __init__(
        self,
        policy: BasePolicy,
        learner_address: Tuple[str, int],
        buffer_size: int = 5000,
        env_wrapper: Optional[Wrapper] = None,
        **kwargs,
    ) -> None:

        self.learner_address = learner_address
        self.buffer_size = buffer_size
        self.policy = policy
        self.mean_reward = 0.0
        # start the simulator
        #subprocess.Popen(
        #    ["sudo", "-u", "ubuntu", "/workspace/LinuxNoEditor/ArrivalSim.sh"],
        #    stdout=subprocess.DEVNULL,
        #)

        # create the racing environment
        #env_config = EnvConfig
        #sim_config = SimulatorConfig
        #self.env = RacingEnv(env_config.__dict__, sim_config.__dict__)
        #self.env.make()

        #TODO: Make arg.
        subprocess.Popen(
            ["sudo", "-u", "ubuntu", "/workspace/LinuxNoEditor/ArrivalSim.sh"],
            stdout=subprocess.DEVNULL,
        )

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
                        "max_steer": 1.0,
                        "min_steer": -1.0,
                        "max_accel": 1.0,
                        "min_accel": -1,
                    })

        self.encoder = create_configurable(
            'config_files/example_sac/encoder.yaml', NameToSourcePath.encoder
        )
        self.encoder.to(DEVICE)

        #self.env.action_space = gym.spaces.Box(np.array([-0.3, -1]), np.array([0.3, 6]))
        self.env = self.DummyVectorEnv([EnvContainer(self.encoder, self.env)])
        
        #print(self.env.action_space)
    def work(self) -> None:
        """Continously collect data"""

        is_train = True
        logging.warn("Trying to send data.")
        response = send_data(data=InitMsg(), addr=self.learner_address, reply=True)
        policy_id, policy = response.data["policy_id"], response.data["policy"]

        while True:
            buffer, result = self.collect_data(
                policy_weights=policy, policy_id=policy_id, is_train=is_train
            )

            if is_train:
                response = send_data(
                    data=BufferMsg(data=buffer), addr=self.learner_address, reply=True
                )

            else:
                self.mean_reward = self.mean_reward*(0.2) + result['rew']*0.8
                logging.warn(f"reward: {self.mean_reward}")
                response = send_data(
                    data=EvalResultsMsg(data=result),
                    addr=self.learner_address,
                    reply=True,
                )

            is_train = response.data["is_train"]
            policy_id, policy = response.data["policy_id"], response.data["policy"]

    def collect_data(
        self, policy_weights: BasePolicy, policy_id: int, is_train: bool = True
    ) -> Tuple[ReplayBuffer, Any]:
        """Collect 1 episode of data in the environment"""
        buffer = ReplayBuffer(size=self.buffer_size)
        self.policy.load_state_dict(policy_weights)
        collector = Collector(
            policy=self.policy, env=self.env, buffer=buffer, exploration_noise=is_train
        )
        result = collector.collect(n_episode=1)
        result["policy_id"] = policy_id
        return buffer, result
