import logging
import subprocess
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from gym import Wrapper
from l2r import build_env
from tianshou.data import ReplayBuffer
from tianshou.data import Collector
from tianshou.policy import BasePolicy

from distrib_l2r.api import BufferMsg
from distrib_l2r.api import EvalResultsMsg
from distrib_l2r.api import InitMsg
from distrib_l2r.utils import send_data


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
        subprocess.Popen(
            ["sudo", "-u", "ubuntu", "/home/LinuxNoEditor/ArrivalSim.sh"],
            stdout=subprocess.DEVNULL,
        )

        # create the racing environment
        self.env = build_env()

        if env_wrapper:
            self.env = env_wrapper(self.env, **kwargs)

    def work(self) -> None:
        """Continously collect data"""

        is_train = True
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
    ) -> Tuple[ReplayBuffer, Dict[Any]]:
        """Collect 1 episode of data in the environment"""
        logging.info(f"[is_train={is_train}] Collecting data")
        buffer = ReplayBuffer(buf_size=self.buffer_size)
        collector = Collector(
            policy=self.policy, env=self.env, buffer=buffer, exploration_noise=is_train
        )
        result = collector.collect(n_episode=1)
        result["policy_id"] = policy_id
        return buffer, result
