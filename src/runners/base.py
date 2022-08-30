from abc import ABC

class BaseRunner(ABC):
    """ABC for BaseRunner."""

    def __init__(self, env):
        """Initialize Base Runner

        Args:
            env (gym.env): Sample gym environment
            agent (BaseAgent): Agent Instance
            config (dict): Config, generated from strictyaml parsing
        """
        self.env = env

    def training(self): # pragma: no cover
        """
        Training Loop
        """
        raise NotImplementedError

    def evaluation(self): # pragma: no cover
        """
        Eval Loop
        """
        raise NotImplementedError