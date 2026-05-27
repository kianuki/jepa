import warnings
print(f"warnings")
import hydra
print(f"hydra")
import torch
print(f"torch")
from hydra.utils import instantiate
print(f"hydra utils")
from omegaconf import OmegaConf
print(f"omegaconf")

from src.datasets.data_utils import get_dataloaders
print(f"src datasets data utils")
from src.trainer import Trainer
print(f"src trainer")
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
print(f"src utils init utils")

print(f"cuda is available: {torch.cuda.is_available()}")
