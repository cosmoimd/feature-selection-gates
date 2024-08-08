#!/usr/bin/env python
# This code refers to the following paper:
# "Feature Selection Gates with Gradient Routing for Endoscopic Image Computing"
# Accepted for publication at MICCAI 2024.

# Please cite both the accepted version and the preprint version if you use this code,
# its methods, ideas, or any part of them.

# Accepted Publication:
# @inproceedings{roffo2024featureselection,
#    title={Feature Selection Gates with Gradient Routing for Endoscopic Image Computing},
#    author={Giorgio Roffo and Carlo Biffi and Pietro Salvagnini and Andrea Cherubini},
#    booktitle={MICCAI},
#    year={2024}
# }

# Preprint version:
# @misc{roffo2024hardattention,
#    title={Hard-Attention Gates with Gradient Routing for Endoscopic Image Computing},
#    author={Giorgio Roffo and Carlo Biffi and Pietro Salvagnini and Andrea Cherubini},
#    year={2024},
#    eprint={2407.04400},
#    archivePrefix={arXiv},
#    primaryClass={eess.IV}
# }

# Import built-in modules

# Import PyTorch modules
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Import plot modules

# Import 3rd party modules

# Import custom modules
# from datasets import dataset

# File information
__author__ = "Giorgio Roffo, Dr."
__copyright__ = "Copyright 2023, COSMO IMD"
__credits__ = ["Giorgio Roffo"]
__license__ = "Private"
__version__ = "1.0.0"
__maintainer__ = "groffo"
__email__ = "groffo@cosmoimd.com"
__status__ = "Production"  # can be "Prototype", "Development" or "Production"

class CosineScheduler:
    """
    This class serves as a wrapper around the PyTorch CosineAnnealingWarmRestarts learning rate scheduler.

    Args:
    optimizer (torch.optim.Optimizer): Optimizer.
    scheduler_params (dict): A dictionary containing the following keys:
        - 'T_0' (int): Number of iterations for the first restart.
        - 'T_mult' (int): A factor to increase T_0 after each restart.
        - 'eta_min' (float): Minimum learning rate. Default: 0.

    Returns:
    An instance of the class 'CosineScheduler'.
    """

    def __init__(self, optimizer, scheduler_params, epochs):
        t0 = epochs // scheduler_params['cycle_limit']
        self.scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                     T_0=t0,
                                                     T_mult=scheduler_params['T_mult'],
                                                     eta_min=scheduler_params['min_lr'])

    def step(self, epoch=None):
        """
        Updates the learning rate at the end of each epoch.

        Args:
        epoch (int): The current epoch number. Default: None.
        """
        self.scheduler.step(epoch)

    def get_lr(self):
        """
        Retrieves the current learning rate.

        Returns:
        (float): The current learning rate.
        """
        return self.scheduler.get_lr()

