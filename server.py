from distrib_l2r.asynchron.learner import AsyncLearningNode
from src.config.yamlize import NameToSourcePath, create_configurable
#from tianshou.policy import SACPolicy
#from tianshou.utils.net.common import Net
#from tianshou.utils.net.continuous import ActorProb, Critic
import torch
from torch import nn
import threading
import numpy as np
import time
import sys

state_shape = (33,)
action_shape = (2,)

if __name__ == "__main__":
    learner = AsyncLearningNode(
        agent=create_configurable(
            "config_files/async_sac_mountaincar/agent.yaml", NameToSourcePath.agent
        ),
        api_key=sys.argv[1],
    )
    print("Initialized!!.")
    server_thread = threading.Thread(target=learner.serve_forever)
    server_thread.start()
    print("Learning?")
    learner.learn()
