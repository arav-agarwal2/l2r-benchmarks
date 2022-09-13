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
        wandb.log({"reward": data})