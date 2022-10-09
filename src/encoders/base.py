from abc import ABC, abstractmethod
from ast import Not
import numpy as np
import gym


class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, image):
        # image: np.array (H, W, C)
        # returns torch.Tensor(*) where * depends on the encoder
        pass
