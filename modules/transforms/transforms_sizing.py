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
from typing import Dict, Tuple

# Import PyTorch modules
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

# Import plot modules

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


class PairedTCombo1:
    """
    A class used to represent a composition of rigid transformations
    (rotations, flipping, color jittering, random noise, and normalizations)

    Attributes
    ----------
    resize_to : int
        The size to resize images to.
    rotation_degree : int
        Maximum degree for rotation.
    color_jitter_params : dict
        Parameters for color jittering, includes brightness, contrast, saturation and hue.
    noise_factor : float
        Factor controlling the amount of random noise added to the image.

    Methods
    -------
    __call__(image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        Applies the transformation pipeline to the input image and mask.
    """
    def __init__(self, resize_to: int, rotation_degree: int, color_jitter_params: Dict[str, float], noise_factor: float, phase: str,
                 mean: list, std: list):
        self.resize_to = resize_to
        self.rotation_degree = rotation_degree
        self.color_jitter_params = color_jitter_params
        self.noise_factor = noise_factor
        self.phase = phase
        self.processing_mean = mean
        self.processing_std = std

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the transformation pipeline to the input image and mask.

        The transformation pipeline includes:
        - resizing the image and mask
        - randomly applying affine transformations
        - randomly flipping the image and mask horizontally or vertically
        - applying color jitter and noise on the image
        - normalizing the image tensor

        Parameters
        ----------
        image : torch.Tensor
            The image tensor to be transformed.
        mask : torch.Tensor
            The mask tensor to be transformed.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple of the transformed image and mask tensors.
        """
        # Resize the image and mask
        image = F.resize(image, [self.resize_to, self.resize_to])
        mask = F.resize(mask, [self.resize_to, self.resize_to])

        if self.phase in ['training', 'train']:
            # Randomly apply affine transformations
            angle = (2*torch.rand(1).item()-1) * self.rotation_degree
            image = F.affine(image, angle=angle, translate=[0, 0], scale=1, shear=0.0)
            mask = F.affine(mask, angle=angle, translate=[0, 0], scale=1, shear=0.0)

            # Randomly flip image
            if torch.rand(1) < 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

            if torch.rand(1) < 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)

            # Apply color jitter only on image
            transform_color_jitter = transforms.ColorJitter(**self.color_jitter_params)
            image = transform_color_jitter(image)

            # Apply random noise
            image = self.add_noise(image, self.noise_factor)

        # Normalize tensor
        image = F.normalize(image, mean=self.processing_mean,
                            std=self.processing_std)

        return image, mask

    @staticmethod
    def add_noise(image, noise_factor):
        noise = torch.randn(image.size()) * noise_factor
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0., 1.)
