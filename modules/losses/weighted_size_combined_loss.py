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


class WeightedCombinedLoss(nn.Module):
    def __init__(self, alpha, beta, gamma, size_threshold_1, weight_1, size_threshold_2, weight_2):
        """
        Implements a weighted combination of different loss functions (L1, L2, and Huber) with extra weights for small and large lesions.

        When selecting the weights:

        - Robustness to outliers: consider assigning higher weights to L1 and smooth L1 losses,
                as they are less sensitive to extreme values.

        - Sensitivity to errors: consider assigning a higher weight to L2 loss,
                as it penalizes large errors more severely than L1 loss.

        Example:
            weighted_combined_loss = WeightedCombinedLoss(alpha=1.0, beta=1.0, gamma=1.0,
                                                size_threshold_1=5.0, weight_1=2.0, size_threshold_2=9.0, weight_2=5.0)

        Args:
            alpha (float): Weight for L1 loss.
            beta (float): Weight for L2 loss.
            gamma (float): Weight for smooth L1 loss (Huber loss).
            size_threshold_1 (float): First size threshold for weighting large lesions.
            size_threshold_2 (float): Second size threshold for weighting large lesions.
            weight_1 (float): Weight for lesions larger than size_threshold_1.
            weight_2 (float): Weight for lesions larger than size_threshold_2.
        """
        super(WeightedCombinedLoss, self).__init__()
        print('+ Weighted Polyp Sizing Loss Function: PSizeLoss')
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.size_threshold_1 = size_threshold_1
        self.size_threshold_2 = size_threshold_2
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        self.l1_loss = nn.L1Loss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')

    def test_weighted_combined_loss(self):
        # Generate random input and target tensors with shape (8, 1)
        input = torch.randn(8, 1)
        target = torch.randn(8, 1)

        # Generate random noise tensor with the same shape as the target tensor
        random_noise = torch.randn_like(target)

        # Calculate the initial loss using the randomly generated input and target tensors
        loss_initial = self(input, target)
        # Print the initial loss value
        print("Loss with initial predictions:", loss_initial.item())

        # Loop through 30 evenly spaced factors between 0.3 and 0.01
        for factor in torch.linspace(0.3, 0.01, 30):
            # Create a new input tensor by adding scaled random noise to the target tensor
            input_close_to_target = target + random_noise * factor

            # Calculate the loss using the new input tensor (close to the target) and the target tensor
            loss_better_predictions = self(input_close_to_target, target)

            # Print the loss value for the current noise factor
            print(f"Loss with factor {factor:.3f}: {loss_better_predictions.item()}")

    def forward(self, input, target):
        """
        Compute the weighted loss.

        Args:
            input (torch.Tensor): Predicted output from the model of shape (batch_size, 1).
            target (torch.Tensor): Ground truth labels of shape (batch_size, 1).

        Returns:
            torch.Tensor: Weighted loss.
        """
        # Calculate the L1, L2 (MSE), and smooth L1 (Huber) losses
        l1 = self.l1_loss(input, target)
        l2 = self.mse_loss(input, target)
        smooth_l1 = self.smooth_l1_loss(input, target)

        # Combine the losses with the specified weights
        loss = self.alpha * l1 + self.beta * l2 + self.gamma * smooth_l1

        # Apply extra weights for lesions larger than the specified size thresholds
        large_lesion_1 = (target > self.size_threshold_1)
        large_lesion_2 = (target >= self.size_threshold_2)

        weights = torch.ones_like(loss)
        weights[large_lesion_1 & ~large_lesion_2] *= self.weight_1
        weights[large_lesion_2] *= self.weight_2

        weighted_loss = loss * weights

        return weighted_loss.mean()
