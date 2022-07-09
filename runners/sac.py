import json
import time
import numpy as np
from runners.base import BaseRunner
from base.envwrapper import EnvContainer


class SACRunner(BaseRunner):
    def __init__(self, env, agent, encoder):
        super().__init__()
        self.env = EnvContainer(self.env, encoder)

    def run(self):
        for _ in range(300):
            done = False
            obs, _ = self.env.reset()

            while not done:
                action = self.agent.select_action(obs)
                obs, reward, done, info = self.env.step(action)
    
    def eval(self):
        print("Evaluation:")
        val_ep_rets = []

        # Not implemented for logging multiple test episodes
        # assert self.cfg["num_test_episodes"] == 1

        for j in range(self.cfg["num_test_episodes"]):
            camera, features, state, _, _ = self.env.reset()
            d, ep_ret, ep_len, n_val_steps, self.metadata = False, 0, 0, 0, {}
            camera, features, state2, r, d, info = self.env.step([0, 1])
            experience, t = [], 0

            while (not d) & (ep_len <= self.cfg["max_ep_len"]):
                # Take deterministic actions at test time
                self.agent.deterministic = True
                self.t = 1e6
                a = self.agent.select_action(features, encode=False)
                camera2, features2, state2, r, d, info = self.env.step(a)

                # Check that the camera is turned on
                assert (np.mean(camera2) > 0) & (np.mean(camera2) < 255)

                ep_ret += r
                ep_len += 1
                n_val_steps += 1

                # Prevent the agent from being stuck
                if np.allclose(state2[15:16], state[15:16], atol=self.agent.atol, rtol=0):
                    # self.file_logger("Sampling random action to get unstuck")
                    a = self.agent.action_space.sample()
                    # Step the env
                    camera2, features2, state2, r, d, info = self.env.step(a)
                    ep_len += 1

                if self.cfg["record_experience"]:
                    recording = self.agent.add_experience(
                        action=a,
                        camera=camera,
                        next_camera=camera2,
                        done=d,
                        env=self.env,
                        feature=features,
                        next_feature=features2,
                        info=info,
                        state=state,
                        next_state=state2,
                        step=t,
                    )
                    experience.append(recording)

                features = features2
                camera = camera2
                state = state2
                t += 1

            self.agent.file_logger(f"[eval episode] {info}")

            val_ep_rets.append(ep_ret)
            self.agent.metadata["info"] = info
            self.agent.log_val_metrics_to_tensorboard(info, ep_ret, ep_len, n_val_steps)

            # Quickly dump recently-completed episode's experience to the multithread queue,
            # as long as the episode resulted in "success"
            if self.agent.cfg["record_experience"]:  # and self.metadata['info']['success']:
                self.agent.file_logger("writing experience")
                self.agent.save_queue.put(experience)

        self.agent.checkpoint_model(ep_ret, self.cfg['max_ep_len'])
        self.agent.update_best_pct_complete(info)

        return val_ep_rets
