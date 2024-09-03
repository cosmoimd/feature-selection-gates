#!/usr/bin/env python
# This code refers to the following paper:
# "Feature Selection Gates with Gradient Routing for Endoscopic Image Computing"
# Accepted for publication at MICCAI 2024.

# Please cite both the accepted version and the preprint version if you use this code,
# its methods, ideas, or any part of them.

# Accepted Publication:
# @inproceedings{roffo2024FSG,
#    title={Feature Selection Gates with Gradient Routing for Endoscopic Image Computing},
#    author={Giorgio Roffo and Carlo Biffi and Pietro Salvagnini and Andrea Cherubini},
#    booktitle={MICCAI 2024, the 27th International Conference on Medical Image Computing and Computer Assisted Intervention, Marrakech, Morocco, October 2024.},
#    year={2024}
#    organization={Springer}
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
from __future__ import print_function

import random

# Import 3rd party modules
import numpy as np
# Import PyTorch modules
import torch
from torch.utils.data import Sampler

# Import plot modules

# Import custom modules
# from datasets import dataset

# File information
# Author: Giorgio Roffo, PhD. Senior Research Scientist. Cosmo IMD, Lainate, MI, Italy.
__author__ = "Giorgio Roffo, Dr."
__copyright__ = "Copyright 2024, COSMO IMD"
__credits__ = ["Giorgio Roffo"]
__license__ = "Private"
__version__ = "1.0.1"
__maintainer__ = "groffo"
__email__ = "groffo@cosmoimd.com,giorgio.roffo@gmail.com"
__status__ = "Production"  # can be "Prototype", "Development" or "Production"


class InstanceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, num_samples_per_instance=None, rand_seed=None, freeze=False):
        '''
        Custom sampler for instance based sampling.
        The sampler selects a fixed number N of samples for each of the different instances (or classes).
        So if the dataset consists of I different instances, the number of samples that will be used
        in an epoch will be I * N. A shuffle is then applied to the selected samples.

        Args
        ----------
        data_source: PyTorch data loading utility Dataset class.
        num_samples_per_instance (int): number of random samples for each instance.
        rand_seed (int): set a random seed (Default: 0).
        '''
        self.data_source = data_source
        self._num_samples = None
        self.num_samples_per_instance = num_samples_per_instance
        self.rand_seed = rand_seed
        self.freeze = freeze

        if self.rand_seed is not None:
            random.seed(self.rand_seed)
        else:
            random.seed(0)

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            # Overestimate
            return len(self.data_source.list_of_unique_object_ids) * self.num_samples_per_instance

        return self._num_samples

    def __iter__(self):

        ind_arr = []
        for idx in np.arange(len(self.data_source.list_of_unique_object_ids)):
            available_frames = np.argwhere(
                self.data_source.list_of_unique_object_ids[idx] == self.data_source.list_object_ids)
            if len(available_frames) > 1:
                available_frames = available_frames.squeeze()
                if self.num_samples_per_instance <= len(available_frames):
                    if not self.freeze:
                        idx_choice = np.random.choice(available_frames.squeeze(), self.num_samples_per_instance,
                                                      replace=False)
                    else:
                        idx_choice = available_frames.squeeze()[:self.num_samples_per_instance]
                else:
                    idx_choice = available_frames.squeeze()

                ind_arr.extend(idx_choice.tolist())

        # shuffles the original list
        if not self.freeze:
            random.shuffle(ind_arr)
            random.shuffle(ind_arr)

        # Update num_samples
        self._num_samples = len(ind_arr)

        return iter(ind_arr)

    def __len__(self):
        return self.num_samples


class CIFAR100Sampler(Sampler):
    def __init__(self, data_source, num_samples_per_instance, rand_seed=None, freeze=False):
        self.data_source = data_source
        self.num_samples_per_instance = num_samples_per_instance
        self.rand_seed = rand_seed
        self.freeze = freeze

        if self.rand_seed is not None:
            random.seed(self.rand_seed)
        else:
            random.seed(0)

        if not isinstance(num_samples_per_instance, int) or num_samples_per_instance <= 0:
            raise ValueError(
                "num_samples_per_instance should be a positive integer value, but got num_samples_per_instance={}".format(
                    num_samples_per_instance))

        self.indices_per_class = [[] for _ in range(100)]  # CIFAR-100 has 100 classes
        for idx, label in enumerate(self.data_source.targets):
            self.indices_per_class[label].append(idx)

    def __iter__(self):
        selected_indices = []
        for class_indices in self.indices_per_class:
            if self.num_samples_per_instance <= len(class_indices):
                if not self.freeze:
                    idx_choice = np.random.choice(class_indices, self.num_samples_per_instance, replace=False)
                else:
                    idx_choice = class_indices[:self.num_samples_per_instance]
            else:
                idx_choice = np.array(class_indices)

            selected_indices.extend(idx_choice.tolist())

        if not self.freeze:
            random.shuffle(selected_indices)

        return iter(selected_indices)

    def __len__(self):
        return len(self.indices_per_class) * self.num_samples_per_instance
