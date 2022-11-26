"""Utility functions used throughout the source. Should generally not add here unless it helps significantly."""
from dataclasses import dataclass
import os, sys
import logging, re
import numpy as np
from datetime import datetime


@dataclass
class ActionSample:
    """Generic object to store action-related params. Might be useful to remove."""

    action = None
    value = None
    logp = None
