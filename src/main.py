import logging
import subprocess
from l2r import build_env
from l2r import RacingEnv
from buffers.replay_buffer import ReplayBuffer
from runners.sac import SACRunner

"""
This script uses the subprocess module to run the simulator.
"""


def race_n_episodes(env: RacingEnv, runner: BaseRunner, num_episodes: int = 5):
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
    race_n_episodes(env=env, runner=runner, num_episodes=1)
