import json
import time
from matplotlib.font_manager import json_dump
import numpy as np
import wandb
from src.loggers.WanDBLogger import WanDBLogger
from src.runners.base import BaseRunner
from src.utils.envwrapper import EnvContainer
from src.utils.utils import ActionSample
from src.agents.SACAgent import SACAgent
from src.loggers.TensorboardLogger import TensorboardLogger
from src.loggers.FileLogger import FileLogger

from src.config.parser import read_config
from src.config.schema import agent_schema
from src.config.schema import experiment_schema
from src.config.schema import replay_buffer_schema
from src.config.yamlize import create_configurable, NameToSourcePath, yamlize
from src.config.schema import encoder_schema
from src.constants import DEVICE

from torch.optim import Adam
import torch
import itertools
import jsonpickle


@yamlize
class ModelFreeRunner(BaseRunner):
    def __init__(
        self,
        exp_config_path: str,
        agent_config_path: str,
        buffer_config_path: str,
        encoder_config_path: str,
        model_save_dir: str,
        experience_save_dir: str,
        num_test_episodes: int,
        num_run_episodes: int,
        save_every_nth_episode: int,
        total_environment_steps: int,
        update_model_after: int,
        update_model_every: int,
        eval_every: int,
        max_episode_length: int,
        use_container: bool = False,
    ):
        super().__init__()
        # Moved initialzation of env to run to allow for yamlization of this class.
        # This would allow a common runner for all model-free approaches

        # Initialize runner parameters
        self.exp_config_path = exp_config_path
        self.agent_config_path = agent_config_path
        self.buffer_config_path = buffer_config_path
        self.encoder_config_path = encoder_config_path
        self.model_save_dir = model_save_dir
        self.experience_save_dir = experience_save_dir
        self.num_test_episodes = num_test_episodes
        self.num_run_episodes = num_run_episodes
        self.save_every_nth_episode = save_every_nth_episode
        self.total_environment_steps = total_environment_steps
        self.update_model_after = update_model_after
        self.update_model_every = update_model_every
        self.eval_every = eval_every
        self.max_episode_length = max_episode_length

        # Loading Experiment configuration
        self.exp_config = read_config(self.exp_config_path, experiment_schema)

        ## AGENT Declaration
        self.agent = create_configurable(self.agent_config_path, NameToSourcePath.agent)

        ## LOGGER Declaration
        self.tb_logger_obj = TensorboardLogger(
            self.agent.model_save_path, self.exp_config["experiment_name"]
        )
        self.file_logger = FileLogger(
            self.agent.model_save_path, self.exp_config["experiment_name"]
        )
        self.file_logger.log_obj.info("Using random seed: {}".format(0))

        ## ENCODER Declaration
        self.encoder = create_configurable(
            self.encoder_config_path, NameToSourcePath.encoder
        )
        self.encoder.to(DEVICE)

        ## BUFFER Declaration
        if not self.agent.load_checkpoint:
            self.replay_buffer = create_configurable(
                self.buffer_config_path, NameToSourcePath.buffer
            )
            self.best_ret = 0
            self.last_saved_episode = 0
            self.best_eval_ret = 0

        else:
            with open(self.exp_config["experiment_state_path"], "r") as openfile:
                json_object = openfile.readline()
            running_vars = jsonpickle.decode(json_object)
            self.file_logger.log(f"running_vars: {running_vars}, {type(running_vars)}")
            # self.replay_buffer = old_runner_obj.replay_buffer
            self.best_ret = running_vars["current_best_ret"]
            self.last_saved_episode = running_vars["last_saved_episode"]
            self.replay_buffer = running_vars["buffer"]
            self.best_eval_ret = running_vars["current_best_eval_ret"]

        if use_container:
            self.env_wrapped = EnvContainer(self.encoder)
        else:
            self.env_wrapped = None

        ## WANDB Declaration
        """self.wandb_logger = None
        if self.api_key:
            self.wandb_logger = WanDBLogger(
                api_key=self.api_key, project_name="test-project"
            )"""

    def run(self, env, api_key):
        self.wandb_logger = None
        if api_key:
            self.wandb_logger = WanDBLogger(
                api_key=api_key, project_name="test-project"
            )
        t = 0
        start_idx = self.last_saved_episode
        for ep_number in range(start_idx + 1, self.num_run_episodes + 1):

            done = False
            if self.env_wrapped:
                obs_encoded = self.env_wrapped.reset(True, env)
            else:
                obs_encoded, info = env.reset()
                #print(obs_encoded)
                obs_encoded = torch.Tensor(obs_encoded)
                obs_encoded.to(DEVICE)

            ep_ret = 0
            total_reward = 0
            info = None
            metric_total = []
            while not done:
                t += 1
                self.agent.deterministic = False
                action_obj = self.agent.select_action(obs_encoded)
                if self.env_wrapped:
                    obs_encoded_new, reward, done, info = self.env_wrapped.step(
                        action_obj.action
                    )
                else:
                    #obs_encoded_new, reward, done, info = env.step(action_obj.action)
                    obs_encoded_new, reward, done, terminated, info = env.step(action_obj.action)
                    obs_encoded_new = torch.Tensor(obs_encoded_new)
                    obs_encoded_new.to(DEVICE)
                    done = done or terminated
                ep_ret += reward
                # self.file_logger.log(f"reward: {reward}")
                self.replay_buffer.store(
                    {
                        "obs": obs_encoded,
                        "act": action_obj,
                        "rew": reward*0.1,
                        "next_obs": obs_encoded_new,
                        "done": done,
                    }
                )
                if done or t == self.max_episode_length:
                    self.replay_buffer.finish_path(action_obj)

                obs_encoded = obs_encoded_new


                if (t >= self.update_model_after) and (
                    t % self.update_model_every == 0
                ):
                    for _ in range(self.update_model_every):
                        batch = self.replay_buffer.sample_batch()
                        metrics = self.agent.update(data=batch)
                        metric_total.append(np.asarray(metrics))
            if ep_number % self.eval_every == 0:
                self.file_logger.log(f"Episode Number before eval: {ep_number}")
                eval_ret = self.eval(env)
                self.file_logger.log(f"Episode Number after eval: {ep_number}, {eval_ret}")
                if eval_ret > self.best_eval_ret:
                    self.best_eval_ret = eval_ret

            if self.wandb_logger:
                self.wandb_logger.log(
                    (
                        ep_ret
                    )
                )
            #            info["metrics"]["total_distance"],
            #            info["metrics"]["total_time"],
            #            info["metrics"]["num_infractions"],
            #            info["metrics"]["average_speed_kph"],
            #            info["metrics"]["average_displacement_error"],
            #            info["metrics"]["trajectory_efficiency"],
            #            info["metrics"]["trajectory_admissibility"],
            #            info["metrics"]["movement_smoothness"],
            #            info["metrics"]["timestep/sec"],
            #            info["metrics"]["laps_completed"],
            #        )
            #    )
            info = {'metrics':{}}
            info["metrics"]["reward"] = ep_ret
            #print(info["metrics"])
            print(self.agent.record["transition_actor"])
            #self.file_logger.log(f"Episode Number after WanDB call: {ep_number}")
            #self.file_logger.log(f"info: {info}")
            self.file_logger.log(
                f"Episode {ep_number}: Current return: {ep_ret}, Previous best return: {self.best_ret}"
            )
            #if len(metric_total) > 0:
            #    metric_total = np.stack(metric_total, axis=0).mean(axis=0)
            #    print(metric_total)
            self.checkpoint_model(ep_ret, ep_number)

    def eval(self, env):
        print("Evaluation:")
        val_ep_rets = []
        ep_ret = 0
        total_reward = 0
        info = None
        metric_total = []
        done = False
        t = 0
        obs_encoded, info = env.reset()
        #print(obs_encoded)
        obs_encoded = torch.Tensor(obs_encoded)
        obs_encoded = obs_encoded.to(DEVICE)
        while not done:
            t += 1
            self.agent.deterministic = True
            action_obj = ActionSample()
            action_obj.action = self.agent.actor.get_action(obs_encoded)[2].detach().cpu().numpy().flatten()
           
            #obs_encoded_new, reward, done, info = env.step(action_obj.action)
            obs_encoded_new, reward, done, terminated, info = env.step(action_obj.action)
            obs_encoded_new = torch.Tensor(obs_encoded_new)
            obs_encoded_new = obs_encoded_new.to(DEVICE)
            done = done or terminated
            ep_ret += reward

            obs_encoded = obs_encoded_new
        return ep_ret



    def checkpoint_model(self, ep_ret, ep_number):
        # Save every N episodes or when the current episode return is better than the best return
        # Following the logic of now deprecated checkpoint_model
        if ep_number % self.save_every_nth_episode == 0 or ep_ret > self.best_ret:
            self.best_ret = max(ep_ret, self.best_ret)
            save_path = f"{self.model_save_dir}/{self.exp_config['experiment_name']}/best_{self.exp_config['experiment_name']}_episode_{ep_number}.statedict"
            self.agent.save_model(save_path)
            self.file_logger.log(f"New model saved! Saving to: {save_path}")
            self.save_experiment_state(ep_number)

    def save_experiment_state(self, ep_number):
        running_variables = {
            "last_saved_episode": ep_number,
            "current_best_ret": self.best_ret,
            "buffer": self.replay_buffer,
            "current_best_eval_ret": self.best_eval_ret,
        }
        if not self.exp_config["experiment_state_path"].endswith(".json"):
            raise Exception("Folder or incorrect file type specified")

        if self.exp_config["experiment_state_path"]:
            # encoded = jsonpickle.encode(self)
            encoded = jsonpickle.encode(running_variables)
            with open(self.exp_config["experiment_state_path"], "w") as outfile:
                outfile.write(encoded)

        else:
            raise Exception("Path not specified or does not exist")

        pass

    """def training(self):
        # List of parameters for both Q-networks (save this for convenience)
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
                self.tb_logger_obj.log_train_metrics(
                    ep_ret, t, t_start, self.episode_num, self.metadata
                )

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
                ) = self.env.reset_episode(t)"""
