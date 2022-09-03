from src.loggers.base import BaseLogger
from src.config.yamlize import yamlize
import logging, re, sys
from datetime import datetime


class FileLogger(BaseLogger):
    def __init__(self, log_dir: str, experiment_name: str) -> None:
        super().__init__(log_dir, experiment_name)
        now = datetime.now()
        self.log_dir = log_dir
        # https://stackoverflow.com/questions/20240464/python-logging-file-is-not-working-when-using-logging-basicconfig
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(
                    f"{self.log_dir}/{experiment_name}/runlogs/{experiment_name}.log"
                ),
                logging.StreamHandler(sys.stdout),
            ],
            force=True,
        )
        self.log_obj = logging

    def log(self, data):
        self.log_obj.info(data)
