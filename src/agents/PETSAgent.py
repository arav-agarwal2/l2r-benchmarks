import itertools
from multiprocessing.sharedctypes import Value
import queue, threading
from copy import deepcopy

import torch
import numpy as np
from gym.spaces import Box
from torch.optim import Adam

from src.agents.base import BaseAgent
from src.config.yamlize import create_configurable, yamlize
from src.planners.CEMPlanner import CEMPlanner
from src.utils.utils import ActionSample

from src.constants import DEVICE

from src.utils.envwrapper import EnvContainer

import torch.nn as nn
import torch.nn.functional as F

from src.constants import DEVICE

@yamlize
class PETSAgent(BaseAgent):
    """Adopted from https://github.com/BY571/PETS-MPC. Currently not using CEM, but random AS."""

    def __init__(self, network_config_path: str, n_ensembles: int = 7, lr: float = 1e-2, model_save_path: str = '/mnt/blah', load_checkpoint: bool = False, deterministic: bool = False):
        super().__init__()
        self.model = DynamicsNetwork.instantiate_from_config(network_config_path)
        self.model.to(DEVICE)
        self.model_save_path = model_save_path
        self.load_checkpoint = load_checkpoint
        self.deterministic = deterministic
        self.n_ensembles = n_ensembles
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.planner = CEMPlanner()
        

    def select_action(self, obs, noise=False) -> np.array:
        action_obj = ActionSample()
        state = obs.detach().float()
        action_obj.action = self.planner.get_action(state,self.model)
        return action_obj


    def register_reset(self, obs) -> np.array:
        pass

    def update(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        o2 = torch.tensor(torch.concatenate((o2 - o,r.reshape((-1,1))),axis=-1))
        o2 = o2.reshape((1,*o2.shape))
        o = torch.tensor(torch.concatenate((o,a),axis=-1))
        o = o.reshape((1,*o.shape)).repeat((self.n_ensembles,1,1))
        mu, log_var = self.model(o)
        assert mu.shape[1:] == o2.shape[1:]
        # Remove validate atm.
        inv_var = (-log_var).exp()
        loss = ((mu - o2)**2 * inv_var).mean(-1).mean(-1).sum() + log_var.mean(-1).mean(-1).sum()
        loss += 0.01 * torch.sum(self.model.max_logvar) - 0.01 * torch.sum(self.model.min_logvar)
        loss.backward()
        self.optimizer.step()
    

    def load_model(self, path):
        return self

    def save_model(self, path):
        pass
    

#TODO: Make ensemble_size and num_ensembles agree.
@yamlize
class DynamicsNetwork(nn.Module):
    def __init__(self, state_size: int = 32, action_size: int = 2, ensemble_size:int = 7, hidden_layer:int = 3, hidden_size: int = 200) -> None:
        super().__init__()
        self.ensemble_size = ensemble_size
        self.input_layer = Ensemble_FC_Layer(state_size + action_size, hidden_size, ensemble_size)
        hidden_layers = []
        hidden_layers.append(nn.SiLU())
        for _ in range(hidden_layer):
            hidden_layers.append(Ensemble_FC_Layer(hidden_size, hidden_size, ensemble_size))
            hidden_layers.append(nn.SiLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
 
        self.mu = Ensemble_FC_Layer(hidden_size, state_size + 1, ensemble_size)
        self.log_var = Ensemble_FC_Layer(hidden_size, state_size + 1, ensemble_size)
        

        self.min_logvar = (-torch.ones((1, state_size + 1)).float() * 10).to(DEVICE)
        self.max_logvar = (torch.ones((1, state_size + 1)).float() / 2).to(DEVICE)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
    
        mu = self.mu(x)

        log_var = self.max_logvar - F.softplus(self.max_logvar - self.log_var(x))
        log_var = self.min_logvar + F.softplus(log_var - self.min_logvar)

        return mu, log_var
    

    def predict(self, states, actions, deterministic=False):
        inputs = torch.cat((states, actions), axis=-1).cpu()
        inputs = torch.from_numpy(inputs.numpy()).float().to(DEVICE)
        inputs = inputs[None, :, :].repeat(self.ensemble_size, 1, 1)
        with torch.no_grad():
            mus, var = self(inputs)
            var = torch.exp(var)

        # [ensembles, batch, prediction_shape]
        assert mus.shape == (self.ensemble_size, states.shape[0], states.shape[1] + 1)
        assert var.shape == (self.ensemble_size, states.shape[0], states.shape[1] + 1)
        
        mus[:, :, :-1] += states.to(DEVICE)
        mus = mus.mean(0)
        std = torch.sqrt(var).mean(0)

        if not deterministic:
            predictions = torch.normal(mean=mus, std=std)
        else:
            predictions = mus

        assert predictions.shape == (states.shape[0], states.shape[1] + 1)

        next_states = predictions[:, :-1]
        rewards = predictions[:, -1].unsqueeze(-1)
        return next_states, rewards


class Ensemble_FC_Layer(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size, bias=True):
        super(Ensemble_FC_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x) -> torch.Tensor:
        w_times_x = torch.bmm(x, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

