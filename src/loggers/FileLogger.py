from src.loggers.base import BaseLogger
from src.config.yamlize import yamlize
import logging, re, sys
from datetime import datetime



@yamlize
class FileLogger(BaseLogger):
    def __init__(self, log_dir:str, exp_name:str) -> None:
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
            force=True
        )
        self.log_obj = logging
    
    def log(self, data):
        self.log_obj.info(data)