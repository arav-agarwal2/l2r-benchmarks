import logging
import subprocess
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from gym import Wrapper
from l2r.envs.env import RacingEnv
import gym

from tianshou.data import ReplayBuffer
from tianshou.data import Collector
from tianshou.policy import BasePolicy

from distrib_l2r.api import BufferMsg
from distrib_l2r.api import EvalResultsMsg
from distrib_l2r.api import InitMsg
from distrib_l2r.utils import send_data


class EnvConfig(object):
    multimodal = True
    eval_mode = True
    n_eval_laps = 1
    max_timesteps = 5000
    obs_delay = 0.1
    not_moving_timeout = 100
    reward_pol = "custom"
    provide_waypoints = False
    reward_kwargs = {
        "oob_penalty": 5.0,
        "min_oob_penalty": 25.0,
        "max_oob_penalty": 125.0,
    }
    controller_kwargs = {
        "sim_version": "ArrivalSim-linux-0.7.1.188691",
        "quiet": False,
        "user": "ubuntu",
        "start_container": False,
        "sim_path": "/home/LinuxNoEditor",
    }
    action_if_kwargs = {
        "max_accel": 6,
        "min_accel": -16,
        "max_steer": .3,
        "min_steer": -.3,
        "ip": "0.0.0.0",
        "port": 7077,
    }
    pose_if_kwargs = {
        "ip": "0.0.0.0",
        "port": 7078,
    }
    camera_if_kwargs = {
        "ip": "0.0.0.0",
        "port": 8008,
    }
    segm_if_kwargs = {
        "ip": 'tcp://127.0.0.1',
        "port": 8009
    }
    birdseye_if_kwargs = {
        "ip": 'tcp://127.0.0.1',
        "port": 8010
    }
    birdseye_segm_if_kwargs = {
        "ip": 'tcp://127.0.0.1',
        "port": 8011
    }
    logger_kwargs = {
        "default": True,
    }
    cameras = {
        "CameraFrontRGB": {
            "Addr": "tcp://0.0.0.0:8008",
            "Format": "ColorBGR8",
            "FOVAngle": 90,
            "Width": 512,
            "Height": 384,
            "bAutoAdvertise": True,
        }
    }


class SimulatorConfig(object):
    racetrack = "Thruxton"
    active_sensors = [
        "CameraFrontRGB",
        "ImuOxtsSensor",
    ]
    driver_params = {
        "DriverAPIClass": "VApiUdp",
        "DriverAPI_UDP_SendAddress": "0.0.0.0",
    }
    camera_params = {
        "Format": "ColorBGR8",
        "FOVAngle": 90,
        "Width": 512,
        "Height": 384,
        "bAutoAdvertise": True,
    }
    vehicle_params = False





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
        self.env = gym.make('CartPole-v0')


        if env_wrapper:
            self.env = env_wrapper(self.env, **kwargs)

    def work(self) -> None:
        """Continously collect data"""

        is_train = True
        logging.info("Trying to send data.")
        response = send_data(data=InitMsg(), addr=self.learner_address, reply=True)
        policy_id, policy = response.data["policy_id"], response.data["policy"]

        while True:
            buffer, result = self.collect_data(
                policy=policy, policy_id=policy_id, is_train=is_train
            )

            if is_train:
                response = send_data(
                    data=BufferMsg(data=buffer), addr=self.learner_address, reply=True
                )

            else:
                response = send_data(
                    data=EvalResultsMsg(data=result),
                    addr=self.learner_address,
                    reply=True,
                )

            is_train = response.data["is_train"]
            policy_id, policy = response.data["policy_id"], response.data["policy"]

    def collect_data(
        self, policy: BasePolicy, policy_id: int, is_train: bool = True
    ) -> Tuple[ReplayBuffer, Any]:
        """Collect 1 episode of data in the environment"""
        logging.info(f"[is_train={is_train}] Collecting data")
        buffer = ReplayBuffer(buf_size=self.buffer_size)
        collector = Collector(
            policy=self.policy, env=self.env, buffer=buffer, exploration_noise=is_train
        )
        result = collector.collect(n_episode=1)
        logging.info(f"reward: {result['rew']}")
        result["policy_id"] = policy_id
        return buffer, result
