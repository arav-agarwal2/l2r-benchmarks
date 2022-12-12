from abc import ABC, abstractmethod
from ast import Not
import numpy as np
import gym


class BaseDataFetcher(ABC):
    @abstractmethod
    def get_dataloaders(self, train_path, val_path):
        pass
