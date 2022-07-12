from abc import ABC

class BaseRunner(ABC):
    """ABC for BaseRunner."""

    def __init__(self, env, agent, config):
        """Initialize Base Runner

        Args:
            env (gym.env): Sample gym environment
            agent (BaseAgent): Agent Instance
            config (dict): Config, generated from strictyaml parsing
        """
        self.env = env
        self.agent = agent
        self.config = config

    def training(self):
        """
        Training Loop
        """
        raise NotImplementedError

    def evaluation(self):
        """
        Eval Loop
        """
        raise NotImplementedError