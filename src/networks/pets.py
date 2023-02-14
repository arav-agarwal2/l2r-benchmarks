import itertools
from multiprocessing.sharedctypes import Value
import queue, threading
from copy import deepcopy

import torch
import numpy as np
from gym.spaces import Box
from torch.optim import Adam

from src.agents.base import BaseAgent
from src.config.yamlize import (
    create_configurable,
    yamlize,
    create_configurable_from_dict,
    ConfigurableDict,
    NameToSourcePath,
)
from src.utils.utils import ActionSample

from src.constants import DEVICE

import torch.nn as nn
import torch.nn.functional as F

from src.constants import DEVICE


@yamlize
class DynamicsNetwork(nn.Module):
    """PETS Dynamics Network. See paper for details, as this is a bit complex."""

    def __init__(
        self,
        state_size: int = 32,
        action_size: int = 2,
        ensemble_size: int = 7,
        hidden_layer: int = 3,
        hidden_size: int = 200,
    ) -> None:
        """Dynamics network init.

        Args:
            state_size (int, optional): State size. Defaults to 32.
            action_size (int, optional): Action dimension. Defaults to 2.
            ensemble_size (int, optional): Number of probabilistic ensembles. Must agree with n_ensembles in petsagent. Defaults to 7.
            hidden_layer (int, optional): Hidden layer count. Defaults to 3.
            hidden_size (int, optional): Hidden layer dimension. Defaults to 200.
        """
        super().__init__()
        self.ensemble_size = ensemble_size
        self.input_layer = Ensemble_FC_Layer(
            state_size + action_size, hidden_size, ensemble_size
        )
        hidden_layers = []
        hidden_layers.append(nn.SiLU())
        for _ in range(hidden_layer):
            hidden_layers.append(
                Ensemble_FC_Layer(hidden_size, hidden_size, ensemble_size)
            )
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
        inputs = torch.cat((states, actions), axis=-1).float().to(DEVICE)
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
        print(std, mus)

        if not deterministic:
            predictions = torch.normal(mean=mus, std=std)
        else:
            predictions = mus

        assert predictions.shape == (states.shape[0], states.shape[1] + 1)

        next_states = predictions[:, :-1]
        rewards = predictions[:, -1].unsqueeze(-1)
        return next_states, rewards


class Ensemble_FC_Layer(nn.Module):
    """Convenience layer for PETS dynamics."""

    def __init__(self, in_features, out_features, ensemble_size, bias=True):
        super(Ensemble_FC_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(
            torch.Tensor(ensemble_size, in_features, out_features)
        )
        torch.nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(ensemble_size, out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x) -> torch.Tensor:
        w_times_x = torch.bmm(x, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b
