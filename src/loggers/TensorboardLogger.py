"""Wrapper around TensorBoard for logging. Unused for WandB, but should work."""
from datetime import datetime
import os
from src.loggers.base import BaseLogger
from src.config.yamlize import yamlize
from tensorboardX import SummaryWriter


class TensorboardLogger(BaseLogger):
    """TBLogger Instance"""

    def __init__(self, log_dir: str, experiment_name: str) -> None:
        """TensorBoard Logger

        Args:
            log_dir (str): Log directory
            experiment_name (str): Experiment Name
        """
        super().__init__(log_dir, experiment_name)
        current_time = datetime.now().strftime("%m%d%H%M%S")
        self.exp_name = experiment_name
        self.tb_log_dir = (
            f"{log_dir}/{experiment_name}/tblogs/{experiment_name}_{current_time}"
        )
        self.tb_logger = SummaryWriter(log_dir=self.tb_log_dir)

    def log(self, metric_data, ep_num):
        """Log metric. Silently fails if something goes wrong.

        Args:
            metric_name (dict): Dictionary of metric_name to metric_value
            ep_num (int): Metric location ( x-value )
        """
        try:
            for metric_name, metric_value in metric_data:
                self.tb_logger.add_scalar(metric_name, metric_value, ep_num)
        except:
            pass
