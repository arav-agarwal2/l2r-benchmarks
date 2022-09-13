import logging
import subprocess
from l2r import build_env
from l2r import RacingEnv
from src.runners.sac import SACRunner
import sys

if __name__ == "__main__":
    # Build environment
    env = build_env()
    if len(sys.argv) > 3:
        runner = SACRunner(env, sys.argv[3])
    else:
        runner = SACRunner(env)

    with open(
        f"{runner.agent.model_save_path}/{runner.exp_config['experiment_name']}/git_config",
        "w+",
    ) as f:
        f.write(" ".join(sys.argv[1:3]))
    # Race!
    runner.run()
