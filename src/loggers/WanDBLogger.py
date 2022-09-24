from src.loggers.base import BaseLogger
from src.config.yamlize import yamlize
import logging, re, sys
import wandb
from datetime import datetime


class WanDBLogger(BaseLogger):
    def __init__(self, api_key: str, project_name: str) -> None:
        # super().__init__(log_dir, experiment_name)
        wandb.login(key=api_key)
        wandb.init(project=project_name, entity="learn2race")

    def log(self, data):
        wandb.log({"reward": data[0]})
        wandb.log({"Distance": data[1]})
        wandb.log({"Time": data[2]})
        wandb.log({"num_infractions": data[3]})
        wandb.log({"average_speed_kph": data[4]})
        wandb.log({"average_displacement_error": data[5]})
        wandb.log({"trajectory_efficiency": data[6]})
        wandb.log({"trajectory_admissibility": data[7]})
        wandb.log({"movement_smoothness": data[8]})
        wandb.log({"timestep/sec": data[9]})
        wandb.log({"laps_completed": data[10]})

    def eval_log(self, data):
        wandb.log({"Eval reward": data[0]})
        wandb.log({"Eval Distance": data[1]})
        wandb.log({"Eval Time": data[2]})
        wandb.log({"Eval Num_infractions": data[3]})
        wandb.log({"Eval Average_speed_kph": data[4]})
        wandb.log({"Eval Average_displacement_error": data[5]})
        wandb.log({"Eval Trajectory_efficiency": data[6]})
        wandb.log({"Eval Trajectory_admissibility": data[7]})
        wandb.log({"Eval Movement_smoothness": data[8]})
        wandb.log({"Eval Timestep/sec": data[9]})
        wandb.log({"Eval Laps_completed": data[10]})
        wandb.log({"Eval Pct_complete": data[11]})
