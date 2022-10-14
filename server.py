from distrib_l2r.asynchron.learner import AsyncLearningNode
from tianshou.policy import DQNPolicy
import torch
from torch import nn
import numpy as np
import time

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
    learner = AsyncLearningNode(policy=DQNPolicy(model=net, optim=optim))
    print("Initialized!!.")
    learner.serve_forever(0.01)