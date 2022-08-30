import logging
import subprocess
from l2r import build_env
from l2r import RacingEnv
from src.buffers.replay_buffer import ReplayBuffer
from src.runners.sac import SACRunner

"""
This script uses the subprocess module to run the simulator.
"""

def race_n_episodes(env: RacingEnv, runner, num_episodes: int = 5):
    """Complete an episode in the environment"""

    for ep in range(num_episodes):
        logging.info(f"Episode {ep+1} of {num_episodes}")

        obs = env.reset()
        runner.run()

if __name__ == "__main__":
    # Build environment
    env = build_env()
    
    buffer = ReplayBuffer()
    runner = SACRunner(env, None, None, buffer)
    # Race!
    runner.run()
