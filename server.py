from distrib_l2r.asynchron.learner import AsyncLearningNode
from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import Net
import torch
from torch import nn
import threading
import numpy as np
import time

state_shape = (4,)
action_shape = (2,)


net = Net(state_shape, action_shape, hidden_sizes=[128,128,128,128], device=torch.device('cuda'))
net.to('cuda')
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

if __name__ == '__main__':
    learner = AsyncLearningNode(policy=DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320), )
    print("Initialized!!.")
    server_thread = threading.Thread(target=learner.serve_forever)
    server_thread.start()
    print("Learning?")
    while True:
        learner.learn()