import socket
from distrib_l2r.asynchron.worker import AsnycWorker
#from src.utils.envwrapper_aicrowd import EnvContainer
from tianshou.policy import SACPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
import torch
from torch import nn
import numpy as np
import time


learner_ip = socket.gethostbyname('learner-service')
learner_address = (learner_ip, 4444)


state_shape = (33,)
action_shape = (2,)

hidden_sizes=[128,128,128,128]
device='cpu'


net = Net(state_shape, action_shape, hidden_sizes=[128,128,128,128], device=torch.device('cuda'))
net.to('cpu')
net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
actor = ActorProb(
    net_a,
    action_shape,
    max_action=1.0,
    device=device,
    unbounded=True
).to(device)
actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)

net_c1 = Net(
    state_shape,
    action_shape,
    hidden_sizes=hidden_sizes,
    concat=True,
    device=device
)
critic1 = Critic(net_c1, device=device).to(device)
critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)

net_c2 = Net(
    state_shape,
    action_shape,
    hidden_sizes=hidden_sizes,
    concat=True,
    device=device
)
critic2 = Critic(net_c2, device=device).to(device)
critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

if __name__ == '__main__':
    worker = AsnycWorker(policy=SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim),learner_address=learner_address)
    print("Worker inited!!!")
    worker.work()
