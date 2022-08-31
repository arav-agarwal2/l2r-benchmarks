from src.loggers.base import BaseLogger
import logging, re, sys
from datetime import datetime

class FileLogger(BaseLogger):
    def __init__(self, log_dir, exp_name) -> None:
        super().__init__(log_dir, exp_name)
        now = datetime.now()
        self.log_dir = log_dir
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"{self.log_dir}/runlogs/{exp_name}.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logging.info("HALP")
        raise ValueError(f"{self.log_dir}/runlogs/{exp_name}.log")
        self.log_obj = logging
    
    def log(self, data):
        self.log_obj.info(data)