from hashlib import md5
import subprocess
#from l2r import build_env
#from l2r import RacingEnv
from src.config.yamlize import NameToSourcePath, create_configurable
import sys
import logging
import gym

if __name__ == "__main__":
    # Build environment
    #env = build_env(controller_kwargs={"quiet": True})
    env = gym.make("BipedalWalker-v3")
    runner = create_configurable(
        "config_files/example_sac/runner.yaml", NameToSourcePath.runner
    )

    with open(
        f"{runner.agent.model_save_path}/{runner.exp_config['experiment_name']}/git_config",
        "w+",
    ) as f:
        f.write(" ".join(sys.argv[1:3]))
    # Race!
    try:
        runner.run(env, sys.argv[3])
    except IndexError as e:
        logging.warning(e)
        runner.run(env, "")
