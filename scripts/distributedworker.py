import socket
from distrib_l2r.asynchron.worker import AsnycWorker
from src.config.yamlize import create_configurable

# from src.utils.envwrapper_aicrowd import EnvContainer
from tianshou.policy import SACPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
import torch
from torch import nn
import numpy as np
import time


learner_ip = socket.gethostbyname("learner-service")
learner_address = (learner_ip, 4444)


if __name__ == "__main__":
    worker = AsnycWorker(learner_address=learner_address)
    print("Worker inited!!!")
    worker.work()
