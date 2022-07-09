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

    def training(self):
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(
            self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters()
        )

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(
            self.actor_critic.policy.parameters(), lr=self.cfg["lr"]
        )
        self.q_optimizer = Adam(self.q_params, lr=self.cfg["lr"])
        self.pi_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pi_optimizer, 1, gamma=0.5
        )

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_critic_target.parameters():
            p.requires_grad = False

        # Count variables (protip: try to get a feel for how different size networks behave!)
        # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])

        # Prepare for interaction with environment
        # start_time = time.time()
        best_ret, ep_ret, ep_len = 0, 0, 0

        self._reset(env, random_pos=True)
        camera, feat, state, r, d, info = self._step(env, [0, 1])

        experience = []
        speed_dim = 1 if self.using_speed else 0
        assert (
            len(feat)
            == self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] + speed_dim
        ), "'o' has unexpected dimension or is a tuple"

        t_start = self.t_start
        # Main loop: collect experience in env and update/log each epoch
        for t in range(self.t_start, self.cfg["total_steps"]):
            a = self.select_action(feat, encode=False)

            # Step the env
            camera2, feat2, state2, r, d, info = self._step(env, a)

            # Check that the camera is turned on
            assert (np.mean(camera2) > 0) & (np.mean(camera2) < 255)

            # Prevents the agent from getting stuck by sampling random actions
            # self.atol for SafeRandom and SPAR are set to -1 so that this condition does not activate
            if np.allclose(state2[15:16], state[15:16], atol=self.atol, rtol=0):
                # self.file_logger("Sampling random action to get unstuck")
                a = env.action_space.sample()

                # Step the env
                camera2, feat2, state2, r, d, info = self._step(env, a)
                ep_len += 1

            state = state2
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.cfg["max_ep_len"] else d

            # Store experience to replay buffer
            if (not np.allclose(state2[15:16], state[15:16], atol=3e-1, rtol=0)) | (
                r != 0
            ):
                self.replay_buffer.store(feat, a, r, feat2, d)
            else:
                # print('Skip')
                skip = True

            if self.cfg["record_experience"]:
                recording = self.add_experience(
                    action=a,
                    camera=camera,
                    next_camera=camera2,
                    done=d,
                    env=env,
                    feature=feat,
                    next_feature=feat2,
                    info=info,
                    reward=r,
                    state=state,
                    next_state=state2,
                    step=t,
                )
                experience.append(recording)

                # quickly pass data to save thread
                # if len(experience) == self.save_batch_size:
                #    self.save_queue.put(experience)
                #    experience = []

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            feat = feat2
            state = state2  # in case we, later, wish to store the state in the replay as well
            camera = camera2  # in case we, later, wish to store the state in the replay as well

            # Update handling
            if (t >= self.cfg["update_after"]) & (t % self.cfg["update_every"] == 0):
                for j in range(self.cfg["update_every"]):
                    batch = self.replay_buffer.sample_batch(self.cfg["batch_size"])
                    self.update(data=batch)

            if (t + 1) % self.cfg["eval_every"] == 0:
                # eval on test environment
                val_returns = self.eval(t // self.cfg["eval_every"], env)

                # Reset
                (
                    camera,
                    ep_len,
                    ep_ret,
                    experience,
                    feat,
                    state,
                    t_start,
                ) = self.reset_episode(env, t)

            # End of trajectory handling
            if d or (ep_len == self.cfg["max_ep_len"]):
                self.metadata["info"] = info
                self.episode_num += 1
                msg = f"[Ep {self.episode_num }] {self.metadata}"
                self.file_logger(msg)
                self.log_train_metrics_to_tensorboard(ep_ret, t, t_start)

                # Quickly dump recently-completed episode's experience to the multithread queue,
                # as long as the episode resulted in "success"
                if self.cfg[
                    "record_experience"
                ]:  # and self.metadata['info']['success']:
                    self.file_logger("Writing experience")
                    self.save_queue.put(experience)

                # Reset
                (
                    camera,
                    ep_len,
                    ep_ret,
                    experience,
                    feat,
                    state,
                    t_start,
                ) = self.reset_episode(env, t)

