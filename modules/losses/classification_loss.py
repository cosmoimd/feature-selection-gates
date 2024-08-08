#!/usr/bin/env python
# This scripts contains all the utility functions used in the toolbox.

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

import numpy as np
# Import built-in modules

# Import PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import plot modules
import matplotlib.pyplot as plt

# Import 3rd party modules

# Import custom modules
# from datasets import dataset

# File information
__author__ = "Giorgio Roffo"
__copyright__ = "Copyright 2023, COSMO IMD"
__credits__ = ["Giorgio Roffo"]
__license__ = "Private"
__version__ = "1.0.0"
__maintainer__ = "Giorgio Roffo"
__email__ = "groffo@cosmoimd.com"
__status__ = "Development"  # can be "Prototype", "Development" or "Production"


class CIFAR100Loss(nn.Module):
    def __init__(self,  alpha=0, beta=0, gamma=1, size_threshold_1=0, weight_1=0, size_threshold_2=0, weight_2=0):
        super(CIFAR100Loss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

    def forward(self, input, target):
        return self.cross_entropy_loss(input, target)
