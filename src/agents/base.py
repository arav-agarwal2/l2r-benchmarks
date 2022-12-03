"""Base agent definition. May be out of date."""
from abc import ABC
from ast import Not
import numpy as np
import gym


class BaseAgent(ABC):
    """Base Agent Definition."""

    default_action_space = gym.spaces.Box(-1, 1, (2,))

    def __init__(self, action_space=default_action_space):
        """Initialize Agent Space

        Args:
            action_space (gym.spaces.Box, optional): Default action space. Defaults to default_action_space.
        """
        self.action_space = action_space

    def select_action(self, obs) -> np.array:  # pragma: no cover
        """Select action based on obs

        Args:
            obs (np.array): Observation. See wrapper / env for details

        Raises:
            NotImplementedError: Need to implement in subclass

        Returns:
            np.array: Action
        """
        raise NotImplementedError

    def register_reset(self, obs) -> np.array:  # pragma: no cover
        """Handle reset of episode.

        Args:
            obs (np.array): Observation

        Returns:
            np.array: Action
        """
        return self.select_action(obs)

    def update(self, data):  # pragma: no cover
        """Model update given data

        Args:
            data (dict): Data.

        Raises:
            NotImplementedError: Need to overload
        """
        raise NotImplementedError

    def load_model(self, path):  # pragma: no cover
        """Load model checkpoint from path

        Args:
            path (str): Path to checkpoint

        Raises:
            NotImplementedError: Need to overload
        """
        raise NotImplementedError

    def save_model(self, path):  # pragma: no cover
        """Save model checkpoint to path

        Args:
            path (str): Path to checkpoint

        Raises:
            NotImplementedError: Need to overload.
        """
        raise NotImplementedError
