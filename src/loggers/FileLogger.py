from src.loggers.base import BaseLogger
import logging, re, sys
from datetime import datetime

class FileLogger(BaseLogger):
    def __init__(self, log_dir, exp_name) -> None:
        super().__init__(log_dir, exp_name)
        now = datetime.now()
        self.log_dir = log_dir
        # https://stackoverflow.com/questions/20240464/python-logging-file-is-not-working-when-using-logging-basicconfig
        from imp import reload 
        reload(logging)
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
