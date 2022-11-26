"""Definition of training constants, and other things which do not need configuration."""
import torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
