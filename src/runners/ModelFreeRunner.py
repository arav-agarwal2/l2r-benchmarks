import json
import time
from matplotlib.font_manager import json_dump
import numpy as np
import wandb
from src.loggers.WanDBLogger import WanDBLogger
from src.runners.base import BaseRunner
from src.utils.envwrapper import EnvContainer
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
        use_container: bool = True,
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
        self.exp_config = read_config(
            self.exp_config_path, experiment_schema
        )

        ## AGENT Declaration
        self.agent = create_configurable(
            self.agent_config_path, NameToSourcePath.agent
        )

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
        if(not self.agent.load_checkpoint):
            self.replay_buffer = create_configurable(
                self.buffer_config_path, NameToSourcePath.buffer
            )
            self.best_ret = 0
            self.last_saved_episode = 0
            self.best_eval_ret = 0
        
        else:
            with open(self.exp_config["experiment_state_path"], 'r') as openfile:           
                json_object = openfile.readline()
            running_vars = jsonpickle.decode(json_object)
            self.file_logger.log(f"running_vars: {running_vars}, {type(running_vars)}")
            #self.replay_buffer = old_runner_obj.replay_buffer
            self.best_ret = running_vars["current_best_ret"]
            self.last_saved_episode = running_vars["last_saved_episode"]
            self.replay_buffer = running_vars["buffer"]
            self.best_eval_ret = running_vars["current_best_eval_ret"]

        

        if use_container:
            self.env_wrapped = EnvContainer(self.encoder)
        else:
            self.env_wrapped = None

        ## WANDB Declaration
        '''self.wandb_logger = None
        if self.api_key:
            self.wandb_logger = WanDBLogger(
                api_key=self.api_key, project_name="test-project"
            )'''

    def run(self, env, api_key):
        self.wandb_logger = None
        if api_key:
            self.wandb_logger = WanDBLogger(
                api_key=api_key, project_name="test-project"
            )
        t = 0
        for ep_number in range(self.last_saved_episode + 1, self.num_run_episodes + self.last_saved_episode + 1):

            done = False
            if self.env_wrapped:
                obs_encoded = self.env_wrapped.reset(True,env)
            else:
                obs_encoded = env.reset()


            ep_ret = 0
            total_reward = 0
            info = None
            while not done:
                t += 1
                self.agent.deterministic = True
                action = self.agent.select_action(obs_encoded)
                if self.env_wrapped:
                    obs_encoded_new, reward, done, info = self.env_wrapped.step(action)
                else:
                    obs_encoded_new, reward, done, info = env.step(action)
                
                ep_ret += reward
                #self.file_logger.log(f"reward: {reward}")
                self.replay_buffer.store(
                    obs_encoded, action, reward, obs_encoded_new, done
                )
                #if (t + 1) % self.eval_every == 0 or t == self.max_episode_length:
                #    self.replay_buffer.finish_path() # TODO: Taking default value currently, select_action needs to return value to change this

                obs_encoded = obs_encoded_new
                if (t >= self.exp_config["update_after"]) & (
                    t % self.exp_config["update_every"] == 0
                ):
                    for _ in range(self.exp_config["update_every"]): 
                        batch = self.replay_buffer.sample_batch()
                        self.agent.update(data=batch)

                if(t % self.eval_every == 0):
                    eval_ret = self.eval(env)
                    if(eval_ret > self.best_eval_ret):
                        self.best_eval_ret = eval_ret


            if self.wandb_logger:
                self.wandb_logger.eval_log((ep_ret, 
                                            info["metrics"]["total_distance"], 
                                            info["metrics"]["total_time"], 
                                            info["metrics"]["num_infractions"],
                                            info["metrics"]["average_speed_kph"],
                                            info["metrics"]["average_displacement_error"],
                                            info["metrics"]["trajectory_efficiency"],
                                            info["metrics"]["trajectory_admissibility"],
                                            info["metrics"]["movement_smoothness"],
                                            info["metrics"]["timestep/sec"],
                                            info["metrics"]["laps_completed"],
                                            ))
            self.file_logger.log(f"info: {info}")
            self.file_logger.log(f"Episode {ep_number}: Current return: {ep_ret}, Previous best return: {self.best_ret}")
            self.checkpoint_model(ep_ret, ep_number)

    def eval(self, env):
        print("Evaluation:")
        val_ep_rets = []

        # Not implemented for logging multiple test episodes
        # assert self.cfg["num_test_episodes"] == 1

        for j in range(self.num_test_episodes):

            if self.env_wrapped:
                obs_encoded = self.env_wrapped.reset()
            else:
                obs_encoded = env.reset()
            

            done, ep_ret, ep_len, n_val_steps, self.metadata = False, 0, 0, 0, {}
            experience, t = [], 0

            while (not done) & (ep_len <= self.max_episode_length):
                # Take deterministic actions at test time
                self.agent.deterministic = True
                self.t = 1e6
                action = self.agent.select_action(obs_encoded, encode=False)
                if self.env_wrapped:
                    obs_encoded_new, reward, done, info = self.env_wrapped.step(action)
                else:
                    obs_encoded_new, reward, done, info = env.step(action)
            
                # Check that the camera is turned on
                ep_ret += reward
                ep_len += 1
                n_val_steps += 1

                #TODO Add the below comment's functionality to the eval loop. The below parts allows for restarts.
                """                 # Prevent the agent from being stuck 
                if np.allclose(
                    state2[15:16], state[15:16], atol=self.agent.atol, rtol=0
                ):
                    # self.file_logger.log("Sampling random action to get unstuck")
                    a = self.agent.action_space.sample()
                    # Step the env
                    camera2, features2, state2, r, d, info = self.env.step(a)
                    camera2 = self.encoder.encode(camera)
                    ep_len += 1
                """

                """
                if self.exp_config["record_experience"]:
                    recording = self.agent.add_experience(
                        action=a,
                        camera=camera,
                        next_camera=camera2,
                        done=d,
                        env=env,
                        feature=features,
                        next_feature=features2,
                        info=info,
                        state=state,
                        next_state=state2,
                        step=t,
                    )
                    experience.append(recording)
                """
                obs_encoded = obs_encoded_new
                t += 1

            self.file_logger.log(f"[eval episode] {info}")

            val_ep_rets.append(ep_ret)
            self.agent.metadata["info"] = info
            self.tb_logger_obj.log_val_metrics(
                info, ep_ret, ep_len, n_val_steps, self.metadata
            )
            if self.wandb_logger:
                self.wandb_logger.eval_log((ep_ret, 
                                            info["metrics"]["total_distance"], 
                                            info["metrics"]["total_time"], 
                                            info["metrics"]["num_infractions"],
                                            info["metrics"]["average_speed_kph"],
                                            info["metrics"]["average_displacement_error"],
                                            info["metrics"]["trajectory_efficiency"],
                                            info["metrics"]["trajectory_admissibility"],
                                            info["metrics"]["movement_smoothness"],
                                            info["metrics"]["timestep/sec"],
                                            info["metrics"]["laps_completed"],
                                            info["metrics"]["pct_complete"],
                                            ))
            
            """            # Quickly dump recently-completed episode's experience to the multithread queue,
            # as long as the episode resulted in "success"
            if self.exp_config[
                "record_experience"
            ]:  # and self.metadata['info']['success']:
                self.file_logger.log("writing experience")
                self.agent.save_queue.put(experience)

            """
            # TODO: add back - info no longer contains "pct_complete"

            # self.agent.update_best_pct_complete(info)
        return max(val_ep_rets)

    def checkpoint_model(self, ep_ret, ep_number):
        # Save every N episodes or when the current episode return is better than the best return
        # Following the logic of now deprecated checkpoint_model
        if (
            ep_number % self.save_every_nth_episode == 0
            or ep_ret > self.best_ret
        ):
            self.best_ret = max(ep_ret, self.best_ret)
            save_path = f"{self.model_save_dir}/{self.exp_config['experiment_name']}/best_{self.exp_config['experiment_name']}_episode_{ep_number}.statedict"
            self.agent.save_model(save_path)
            self.file_logger.log(f"New model saved! Saving to: {save_path}")
            self.last_saved_episode = ep_number
            self.save_experiment_state(ep_number)
    
    def save_experiment_state(self, epnumber):
        running_variables = {"last_saved_episode": self.last_saved_episode, "current_best_ret":self.best_ret, "current_episode":epnumber, "buffer": self.replay_buffer, "current_best_eval_ret": self.best_eval_ret}
        if(not self.exp_config["experiment_state_path"].endswith(".json")):
            raise Exception("Folder or incorrect file type specified")
        
        if(self.exp_config["experiment_state_path"]):
            #encoded = jsonpickle.encode(self)
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
