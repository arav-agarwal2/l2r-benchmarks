"""Base Runner. Inherit from here, and respect the protocol."""
from abc import ABC


class BaseRunner(ABC):
    """ABC for BaseRunner."""

    def __init__(self):
        """Initialize Base Runner"""

    def training(self, env):  # pragma: no cover
        """
        Training Loop
        """
        raise NotImplementedError

    def evaluation(self, env):  # pragma: no cover
        """
        Eval Loop
        """
        raise NotImplementedError
