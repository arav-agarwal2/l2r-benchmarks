import logging
import subprocess
from l2r import build_env
from l2r import RacingEnv
from src.runners.sac import SACRunner

"""
This script uses the subprocess module to run the simulator.
"""

if __name__ == "__main__":
    # Build environment
    env = build_env()
    runner = SACRunner(env, None, None)
    # Race!
    runner.run()
