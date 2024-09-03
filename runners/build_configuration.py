#!/usr/bin/env python
# This scripts contains all the utility functions used in the toolbox.

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
import importlib

# Import PyTorch modules
import torch

# Import custom modules
from modules.base_params import *
from modules.datasets import sampler
from modules.losses.classification_loss import CIFAR100Loss
from modules.losses.weighted_size_combined_loss import WeightedCombinedLoss

# Import plot modules
# Import 3rd party modules

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


# -----------------------------------------------------------------
# FSAG: Feature-Selection-Attention Gates
# Gradient Routing (GR) for Online-Feature-Selection (OFS)
# ----------------------------------------------------------------
def build_configuration(config, k_fold, num_folds=6):
    """
    This function reads a configuration file and prepares the necessary components for a machine learning training or
    evaluation run. It handles setting up the transforms, dataset, sampler, model, loss function, optimizer, and scheduler.
    It also allows configuration of these components via a provided config dictionary.

    Args:
        config (dict): The configuration parameters from a YAML config file or similar source. This should contain
                       keys and values defining the settings for the transforms, dataset, model, loss function,
                       optimizer, and scheduler.
        k_fold (int): The k-th fold for k-fold cross-validation. Used to split the dataset into training and testing subsets.
        num_folds (int): total number of folds, e.g., 6

    Returns:
        dataset_train (torch.utils.data.Dataset): The training dataset instance.
        dataset_test (torch.utils.data.Dataset): The testing dataset instance.
        train_sampler (InstanceSampler): The instance sampler for the training dataset.
        model (torch.nn.Module): The instantiated model.
        transforms (torchvision.transforms): The transformations to apply on the training dataset.
        resize_normalize_only (torchvision.transforms): The transformations to apply on the testing dataset.
        loss_fn (torch.nn.Module): The loss function for the model.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler for the optimizer.
    """
    print("> Building the configuration...")

    # Instantiate transforms
    print("> Setting up transforms...")
    transforms_module = importlib.import_module('modules.transforms.' + config['transforms'])
    transforms_class = getattr(transforms_module, config['transform_combo'])

    # Get transform parameters from the config
    resize_image = config.get('resize_to', 384)
    rotation_degree = config.get('rotation_degree', 0)
    color_jitter_params = config.get('color_jitter_params', {})
    noise_factor = config.get('noise_factor', 0)
    image_preprocessing_mu = config.get('dataset_params')['image_preprocessing']['mean']
    image_preprocessing_std = config.get('dataset_params')['image_preprocessing']['std']

    transforms = transforms_class(resize_image, rotation_degree, color_jitter_params, noise_factor,
                                  phase='training',
                                  mean=image_preprocessing_mu,
                                  std=image_preprocessing_std)
    resize_normalize_only = transforms_class(resize_image, rotation_degree, color_jitter_params, noise_factor,
                                             phase='testing',
                                             mean=image_preprocessing_mu,
                                             std=image_preprocessing_std)

    # Instantiate model
    print("> Initializing model...")
    model_module = importlib.import_module('modules.models.' + config['model'])
    model_class = getattr(model_module, config['model_class'])

    # Get model parameters from the config
    model_params = config.get('model_params', {})

    # Update defaults with provided parameters
    default_model_params.update(model_params)

    # Pass the dictionary keys and values as arguments to the function
    model = model_class(default_model_params)

    # Instantiate loss
    print("> Setting up loss function...")
    loss_module = importlib.import_module('modules.losses.' + config['loss'])
    loss_class = getattr(loss_module, config['loss_class'])

    # -----------------------------------------------------------------
    # FSAG: Feature-Selection-Attention Gates
    # Gradient Routing (GR) for Online-Feature-Selection (OFS)
    # ----------------------------------------------------------------
    # Get loss parameters from the config
    loss_params = config.get('loss_params', {})
    gr_params = config.get('gr_ofs_lr', 0.000001)
    enable_gr = config.get('FSAG', False)
    enable_lr_gr = config.get('FSAG_LR', False)
    input_types = config.get('model_params', {})

    # Update defaults with provided parameters
    default_loss_params.update(loss_params)

    # Pass the dictionary keys and values as arguments to the function
    loss_fn = loss_class(**default_loss_params)

    # ---------------------------------------------------------------
    # Gradient Routing (GR) loss for Online-Feature-Selection (OFS)
    # ---------------------------------------------------------------
    if model_params['num_classes'] == 1:
        gr_ofs_loss = WeightedCombinedLoss(0, 0, 1, 5.0, 5.0, 10.0, 10.0)
    else:
        gr_ofs_loss = CIFAR100Loss()

    # Instantiate optimizer
    print("> Setting up main optimizer...\n\n")
    print('  FSAG: Feature-Selection-Attention Gates')
    print('  Gradient Routing (GR) for Online-Feature-Selection (OFS)\n\n')

    optimizer_class = getattr(torch.optim, config['optimizer'])
    # Filter out GROFS (FSAG) parameters from the main optimizer
    non_grofs_params = [p for n, p in model.named_parameters() if not 'fsag' in n]
    optimizer = optimizer_class(non_grofs_params, **config['optimizer_params'])
    if enable_gr:
        print(f"+ Setting up GR-OFS optimizer LR: {gr_params}")

        # Define prefixes based on active input types
        active_prefixes = []
        if input_types['use_rgb']:
            active_prefixes.append('fsag_rgb_')
        if input_types['use_mask']:
            active_prefixes.append('fsag_mask_')
            active_prefixes.append('fsag_rgb_')
        if input_types['use_depth']:
            active_prefixes.append('fsag_depth_')

        # Collect all relevant FSAG parameters for the GROFS optimizer
        fsag_params = [p for n, p in model.named_parameters() if
                       any(n.startswith(prefix) for prefix in active_prefixes)]
        if len(fsag_params) == 0:
            fsag_params = [p for n, p in model.named_parameters() if
                           any(prefix in n for prefix in active_prefixes)]

        # CNNs: Ensure FSG parameters are included for Fully Connected Layers.
        # Extend fsag_params with the parameters corresponding to 'fsag_L1' and 'fsag_L2'.
        fsag_params.extend([p for n, p in model.named_parameters() if n in ['fsag_L1', 'fsag_L2']])

        GROFS_optimizer = torch.optim.Adam(fsag_params, lr=gr_params)
    else:
        GROFS_optimizer = None

    # Instantiate scheduler
    print("> Setting up scheduler...")
    scheduler_module = importlib.import_module('modules.schedulers.' + config['lr_scheduler'])
    scheduler_class = getattr(scheduler_module, config['lr_scheduler_class'])
    lr_scheduler = scheduler_class(optimizer, config['scheduler_params'], config['num_epochs'])
    if enable_lr_gr:
        lr_scheduler_gr = scheduler_class(GROFS_optimizer, config['scheduler_params'], config['num_epochs'])
    else:
        lr_scheduler_gr = None

    # Instantiate dataset
    print("> Loading dataset...")
    dataset_module = importlib.import_module('modules.datasets.dataset')
    dataset_class = getattr(dataset_module, config['dataset_params']['dataset_class'])

    # Get dataset parameters from the config
    dataset_params = config.get('dataset_params', {})

    # Update defaults with provided parameters
    default_dataset_params.update(dataset_params)

    # Pass the dictionary keys and values as arguments to the function
    print("  Training Set...")
    dataset_train = dataset_class(params=default_dataset_params, phase='training', test_id=k_fold, num_folds=num_folds,
                                  transform=transforms)

    print("  Validation Set...")
    dataset_valid = dataset_class(params=default_dataset_params, phase='validation', test_id=k_fold,
                                  num_folds=num_folds, transform=resize_normalize_only)

    print("  Testing Set...")
    dataset_test = dataset_class(params=default_dataset_params, phase='testing', test_id=k_fold, num_folds=num_folds,
                                 transform=resize_normalize_only)

    # Returns num_samples_per_instance random samples for each instance.
    print("> Creating sampler...")
    if 'n_frames_to_sample_valid' not in config['dataset_params']:
        config['dataset_params']['n_frames_to_sample_valid'] = 300  # Default value

    if model_params['num_classes'] == 1:
        train_sampler = sampler.InstanceSampler(data_source=dataset_train,
                                                num_samples_per_instance=config['dataset_params'][
                                                    "n_frames_to_sample_train"],
                                                rand_seed=0, freeze=False)

        valid_sampler = sampler.InstanceSampler(data_source=dataset_valid,
                                                num_samples_per_instance=config['dataset_params'][
                                                    "n_frames_to_sample_valid"],
                                                rand_seed=0, freeze=False)
    else:
        train_sampler = sampler.CIFAR100Sampler(data_source=dataset_train,
                                                num_samples_per_instance=config['dataset_params'][
                                                    "n_frames_to_sample_train"],
                                                rand_seed=0, freeze=False)

        valid_sampler = sampler.CIFAR100Sampler(data_source=dataset_valid,
                                                num_samples_per_instance=config['dataset_params'][
                                                    "n_frames_to_sample_valid"],
                                                rand_seed=0, freeze=False)

    print("> Configuration built successfully!")
    return dataset_train, dataset_test, dataset_valid, train_sampler, valid_sampler, model, transforms, resize_normalize_only, loss_fn, gr_ofs_loss, optimizer, GROFS_optimizer, lr_scheduler, lr_scheduler_gr
