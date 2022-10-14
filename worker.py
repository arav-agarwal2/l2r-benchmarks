import socket
from distrib_l2r.asynchron.worker import AsnycWorker
#from src.utils.envwrapper_aicrowd import EnvContainer
from tianshou.policy import DQNPolicy
import torch
from torch import nn
import numpy as np
import time


learner_ip = socket.gethostbyname('learner-service')
learner_address = (learner_ip, 4444)

state_shape = (4,)
action_shape = (2,)

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape))
        ])
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)


if __name__ == '__main__':
    worker = AsnycWorker(policy=DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320),learner_address=learner_address)
    print("Worker inited!!!")
    worker.work()
