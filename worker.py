import socket
from distrib_l2r.asynchron.worker import AsnycWorker
#from src.utils.envwrapper_aicrowd import EnvContainer
from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import Net
import torch
from torch import nn
import numpy as np
import time


learner_ip = socket.gethostbyname('learner-service')
learner_address = (learner_ip, 4444)

state_shape = (4,)
action_shape = (2,)


net = Net(state_shape, action_shape, hidden_sizes=[128,128,128,128])
optim = torch.optim.Adam(net.parameters(), lr=1e-3)


if __name__ == '__main__':
    worker = AsnycWorker(policy=DQNPolicy(net, optim),learner_address=learner_address)
    print("Worker inited!!!")
    worker.work()
