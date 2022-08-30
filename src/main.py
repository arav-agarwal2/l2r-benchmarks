import logging
import subprocess
from l2r import build_env
from l2r import RacingEnv
# from buffers.replay_buffer import ReplayBuffer
# from runners.sac import SACRunner

"""
This script uses the subprocess module to run the simulator.
"""

if __name__ == "__main__":
    # Build environment
    env = build_env()
    
#     buffer = ReplayBuffer()
#     runner = SACRunner(env, None, None, buffer)
#     # Race!
#     runner.run()
