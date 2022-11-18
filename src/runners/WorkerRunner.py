from src.runners.base import BaseRunner

from src.config.yamlize import create_configurable, NameToSourcePath, yamlize
from src.constants import DEVICE

from torch.optim import Adam
import numpy as np

@yamlize
class WorkerRunner(BaseRunner):
    """
    Runner designed for the Worker. All it does is collect data under two scenarios:
      - train, where we include some element of noise
      - test, where we include no such noise.
    """

    def __init__(
        self, agent_config_path: str, buffer_config_path: str, max_episode_length: int
    ):
        super().__init__()
        # Moved initialization of env to run to allow for yamlization of this class.
        # This would allow a common runner for all model-free approaches

        # Initialize runner parameters
        self.agent_config_path = agent_config_path
        self.buffer_config_path = buffer_config_path
        self.max_episode_length = max_episode_length

        ## AGENT Declaration
        self.agent = create_configurable(self.agent_config_path, NameToSourcePath.agent)

    def run(self, env, agent_params, is_train):
        """Grab data for system that's needed, and send a buffer accordingly. Note: does a single 'episode'
           which might not be more than a segment in l2r's case.

        Args:
            env (_type_): _description_
            agent (_type_): some agent
            is_train: Whether to collect data in train mode or eval mode
        """
        self.agent.load_model(agent_params)
                   
        self.agent.deterministic = not is_train
        t = 0
        done = False
        state_encoded = env.reset()

        ep_ret = 0
        self.replay_buffer = create_configurable(
                self.buffer_config_path, NameToSourcePath.buffer
            )

        while not done:
            t += 1
            #print(f't:{t}')
            action_obj = self.agent.select_action(state_encoded)
            next_state_encoded, reward, done, info = env.step(action_obj.action)
            # print(f'info{info}')
            ep_ret += reward
            self.replay_buffer.store(
                {
                    "obs": state_encoded,
                    "act": action_obj,
                    "rew": reward,
                    "next_obs": next_state_encoded,
                    "done": done,
                }
            )
            if done or t == self.max_episode_length:
                self.replay_buffer.finish_path(action_obj)

            state_encoded = next_state_encoded
        from copy import deepcopy

        info["metrics"]["reward"] = ep_ret
        print(info["metrics"])
        return deepcopy(self.replay_buffer), info["metrics"]
