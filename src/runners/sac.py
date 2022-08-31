import json
import time
import numpy as np
from src.runners.base import BaseRunner
from src.utils.envwrapper import EnvContainer
from src.agents.sac_agent import SACAgent
from src.loggers.TensorboardLogger import TensorboardLogger
from src.loggers.FileLogger import FileLogger
from src.encoders.VAE import VAE

from src.config.parser import read_config
from src.config.schema import agent_schema
from src.config.schema import experiment_schema
from src.config.schema import replay_buffer_schema
from src.config.schema import encoder_schema
from src.constants import DEVICE

from torch.optim import Adam
import torch
import itertools
from src.buffers.replay_buffer import ReplayBuffer

class SACRunner(BaseRunner):
    def __init__(self, env):
        super().__init__(env)
        self.agent_config = read_config("src/config_files/example_sac/agent.yaml",agent_schema)
        self.exp_config = read_config("src/config_files/example_sac/experiment.yaml",experiment_schema)
        self.buffer_config = read_config("src/config_files/example_sac/buffer.yaml",replay_buffer_schema)
        self.encoder_config = read_config("src/config_files/example_sac/encoder.yaml",encoder_schema)

        ## ENV Setup
        self.env = env

        ## BUFFER Declaration
        self.action_space = self.env.action_space
        #raise ValueError(self.action_space)
        self.replay_buffer = ReplayBuffer(obs_dim=33, act_dim=self.action_space.shape[0], size=self.buffer_config["replay_size"])

        ## LOGGER Declaration
        self.tb_logger_obj = TensorboardLogger(self.agent_config["model_save_path"], self.exp_config["experiment_name"])
        self.file_logger = FileLogger(self.agent_config["model_save_path"], self.exp_config["experiment_name"])
        self.file_logger.log("Using random seed: {}".format(0))

        ## ENCODER Declaration
        self.encoder = VAE()
        self.encoder.to(DEVICE)

        ## AGENT Declaration
        self.agent = SACAgent(self.agent_config)
        self.best_ret = 0

    def run(self):
        for _ in range(1):
            done = False
            obs = self.env.reset()['images']['CameraFrontRGB']
            obs = self.encoder.encode(obs)

            while not done:
                action = self.agent.select_action(obs)
                #raise ValueError(action)
                obs, reward, terminated, info = self.env.step(action)
                raise ValueError(action)
                obs = obs['images']['CameraFrontRGB']
                obs = self.encoder.encode(obs)
    
    def eval(self):
        print("Evaluation:")
        val_ep_rets = []

        # Not implemented for logging multiple test episodes
        # assert self.cfg["num_test_episodes"] == 1

        for j in range(self.cfg["num_test_episodes"]):
            camera, features, state, _, _ = self.env.reset()
            camera = self.encoder.encode(camera)
            d, ep_ret, ep_len, n_val_steps, self.metadata = False, 0, 0, 0, {}
            camera, features, state2, r, d, info = self.env.step([0, 1])
            camera = self.encoder.encode(camera)
            experience, t = [], 0

            while (not d) & (ep_len <= self.cfg["max_ep_len"]):
                # Take deterministic actions at test time
                self.agent.deterministic = True
                self.t = 1e6
                a = self.agent.select_action(features, encode=False)
                camera2, features2, state2, r, d, info = self.env.step(a)

                # Check that the camera is turned on
                assert (np.mean(camera2) > 0) & (np.mean(camera2) < 255)
                camera2 = self.encoder.encode(camera2)

                ep_ret += r
                ep_len += 1
                n_val_steps += 1

                # Prevent the agent from being stuck
                if np.allclose(state2[15:16], state[15:16], atol=self.agent.atol, rtol=0):
                    # self.file_logger.log("Sampling random action to get unstuck")
                    a = self.agent.action_space.sample()
                    # Step the env
                    camera2, features2, state2, r, d, info = self.env.step(a)
                    camera2 = self.encoder.encode(camera)
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

            self.file_logger.log(f"[eval episode] {info}")

            val_ep_rets.append(ep_ret)
            self.agent.metadata["info"] = info
            self.tb_logger_obj.log_val_metrics(info, ep_ret, ep_len, n_val_steps, self.metadata)

            # Quickly dump recently-completed episode's experience to the multithread queue,
            # as long as the episode resulted in "success"
            if self.agent.cfg["record_experience"]:  # and self.metadata['info']['success']:
                self.file_logger.log("writing experience")
                self.agent.save_queue.put(experience)

        self.checkpoint_model(ep_ret, self.cfg['max_ep_len'])
        self.agent.update_best_pct_complete(info)

        return val_ep_rets
    
    ## Sid needs to take care of the actor_critic objects
    def checkpoint_model(self, ep_ret, n_eps):
        if ep_ret > self.best_ret:  # and ep_ret > 100):
            path_name = f"{self.cfg['model_save_path']}/best_{self.cfg['experiment_name']}_episode_{n_eps}.statedict"
            self.file_logger.log(
                f"New best episode reward of {round(ep_ret, 1)}! Saving: {path_name}"
            )
            self.best_ret = ep_ret
            torch.save(self.actor_critic.state_dict(), path_name)
            path_name = f"{self.cfg['model_save_path']}/best_{self.cfg['experiment_name']}_episode_{n_eps}.statedict"
            try:
                # Try to save Safety Actor-Critic, if present
                torch.save(self.safety_actor_critic.state_dict(), path_name)
            except:
                pass

        elif self.cfg['save_freq'] > 0 and (n_eps + 1 % self.cfg["save_freq"] == 0):
            path_name = f"{self.cfg['model_save_path']}/{self.cfg['experiment_name']}_episode_{n_eps}.statedict"
            self.file_logger.log(
                f"Periodic save (save_freq of {self.cfg['save_freq']}) to {path_name}"
            )
            torch.save(self.actor_critic.state_dict(), path_name)
            path_name = f"{self.cfg['model_save_path']}/{self.cfg['experiment_name']}_episode_{n_eps}.statedict"
            try:
                # Try to save Safety Actor-Critic, if present
                torch.save(self.safety_actor_critic.state_dict(), path_name)
            except:
                pass

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

        self.env.reset(random_pos=True)
        camera, feat, state, r, d, info = self.env.step([0, 1])
        camera = self.encoder.encode(camera)
        experience = []
        speed_dim = 1 if self.using_speed else 0
        assert (
            len(feat)
            == self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] + speed_dim
        ), "'o' has unexpected dimension or is a tuple"

        t_start = self.t_start
        # Main loop: collect experience in env and update/log each epoch
        for t in range(self.t_start, self.cfg["total_steps"]):
            a = self.agent.select_action(feat, encode=False)

            # Step the env
            camera2, feat2, state2, r, d, info = self.env.step(a)
            camera2 = self.encoder.encode(camera2)
            # Check that the camera is turned on
            assert (np.mean(camera2) > 0) & (np.mean(camera2) < 255)

            # Prevents the agent from getting stuck by sampling random actions
            # self.atol for SafeRandom and SPAR are set to -1 so that this condition does not activate
            if np.allclose(state2[15:16], state[15:16], atol=self.atol, rtol=0):
                # self.file_logger.log("Sampling random action to get unstuck")
                a = self.agent.action_space.sample()

                # Step the env
                camera2, feat2, state2, r, d, info = self.env.step(a)
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
                    env=self.env,
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
                val_returns = self.eval()

                # Reset
                (
                    camera,
                    ep_len,
                    ep_ret,
                    experience,
                    feat,
                    state,
                    t_start,
                ) = self.env.reset_episode(t)

            # End of trajectory handling
            if d or (ep_len == self.cfg["max_ep_len"]):
                self.metadata["info"] = info
                self.episode_num += 1
                msg = f"[Ep {self.episode_num }] {self.metadata}"
                self.file_logger.log(msg)
                self.tb_logger_obj.log_train_metrics(ep_ret, t, t_start, self.episode_num, self.metadata)

                # Quickly dump recently-completed episode's experience to the multithread queue,
                # as long as the episode resulted in "success"
                if self.cfg[
                    "record_experience"
                ]:  # and self.metadata['info']['success']:
                    self.file_logger.log("Writing experience")
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
                ) = self.env.reset_episode(t)