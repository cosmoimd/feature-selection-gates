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
import argparse
import re
# Import built-in modules
import shutil

# Import 3rd party modules
import numpy as np
import pandas as pd
# Import PyTorch modules
import torch
from sklearn.metrics import balanced_accuracy_score
from tabulate import tabulate

import utils
from modules.analytics import visualizations
# Import custom modules
from utils import load_yml, find_checkpoint
from runners.build_configuration import build_configuration
from runners.trainer import Trainer
import matplotlib
matplotlib.use('Agg')

import os
import ssl

# Import plot modules

# File information
__author__ = "Giorgio Roffo, Dr."
__copyright__ = "Copyright 2023, COSMO IMD"
__credits__ = ["Giorgio Roffo"]
__license__ = "Private"
__version__ = "1.0.0"
__maintainer__ = "groffo"
__email__ = "groffo@cosmoimd.com"
__status__ = "Prototype"



#################################################
# Demo for configuring an experiment
#################################################
def run_polypsizing_experiments(pars, output_folder):
    """
    The main function for polyp size estimation and experiments.

    Args:
        pars (dict): A dictionary containing parameters for training, including
        the dataset, model, transforms, loss function, optimizer, learning rate scheduler,
        batch size, learning rate, and number of epochs.
    """
    # Determine device and inform user
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'> Training on device: {device}')
    inference = pars['inference_mode'] if 'inference_mode' in pars else False
    verbose_validation_mode = pars['verbose_validation_mode'] if 'verbose_validation_mode' in pars else False

    kfold = pars['dataset_params']['kfold']
    start_kfold = pars['dataset_params']['start_from_fold']

    # If you need to loop on the k-folds
    # for k in np.arange(start_kfold, kfold):
    #
    #     print(f'+ Working on K-FOLD [{k}]\n')
    #     directory_name = os.path.join(output_folder, f"kfold_{k}")
    #
    #     # Check if the directory exists
    #     if not os.path.exists(directory_name):
    #         # Create the directory
    #         os.makedirs(directory_name)

    ########################################################
    # Steps to prepare the experiments:
    ########################################################

    # Step 1: Build the configuration
    # This loads the dataset, model, transforms, loss function, and learning rate scheduler.

    # (train_set, test_set, valid_set, train_sampler, valid_sampler, model, transforms,
    #  resize_normalize_only, loss_fn, gr_ofs_loss, optimizer, GROFS_optimizer, lr_scheduler, lr_scheduler_gr) = build_configuration(
    #     pars, k_fold=k)


    # Step 2: Create DataLoaders - Training and Testing
    # This loads the data and provides batches of data for training.
    # tr_dataloader = torch.utils.data.DataLoader(train_set, batch_size=pars['dataset_params']['batch_size'],
    #                                             num_workers=pars["dataset_params"]["num_workers"],
    #                                             pin_memory=True,
    #                                             sampler=train_sampler)

    # This loads the data and provides batches of data for testing
    # vl_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=pars['dataset_params']['batch_size']*3,
    #                                             num_workers=pars["dataset_params"]["num_workers"],
    #                                             pin_memory=True,
    #                                             sampler=valid_sampler)

    # This loads the data and provides batches of data for testing
    # ts_dataloader = torch.utils.data.DataLoader(test_set, batch_size=pars['dataset_params']['batch_size']*4,
    #                                             num_workers=pars["dataset_params"]["num_workers"],
    #                                             pin_memory=True)

    # Step 3: Instantiate a trainer
    # This handles the training process, including the forward pass, computing the loss,
    # and updating the parameters with the optimizer.
    # trainer = Trainer(pars, model, loss_fn, gr_ofs_loss, optimizer, GROFS_optimizer, lr_scheduler, lr_scheduler_gr)


#############################################
# Main Script
#############################################
if __name__ == "__main__":

    print(f"{'-' * 45}")
    print("DEMO: main training script for polyp sizing.")
    print("This script will train the model based on the parameters provided.")
    print(f"{'-' * 45}")


    ssl._create_default_https_context = ssl._create_unverified_context

    # Argument parser for command-line script usage.
    parser = argparse.ArgumentParser(description="Training script for polyp sizing.")
    parser.add_argument("-parFile", type=str, help="Path to the parameter file.")
    args = parser.parse_args()

    # Check if a parameter file path was provided as an argument.
    if args.parFile:
        # Check if the provided path is a valid file.
        if os.path.isfile(args.parFile):
            # Load parameters from the provided YAML file.
            pars = load_yml(args.parFile)
        else:
            print("Provided parameter file does not exist. Please check the path.")
            exit()
    else:
        # If no parameter file was provided, use the default file.
        default_yml_path = "config/config_resnet18_RGB.yml"
        if os.path.isfile(default_yml_path):
            # Load parameters from the default YAML file.
            pars = load_yml(default_yml_path)
        else:
            print(f"Default parameter file {default_yml_path} does not exist. Please check the path.")
            exit()

    # Define the output directory
    output_folder = pars['dataset_params']['output_folder']
    inference = pars['inference_mode'] if 'inference_mode' in pars else False

    # Check if the directory already exists
    if os.path.exists(output_folder):
        # If the directory exists, alert the user
        print(f"The directory '{output_folder}' already exists.")

        if not inference:
            # Ask the user if they want to delete the existing directory and create a new one
            user_input = input("Do you want to delete this directory and create a new one? (yes/no  y/n): ")

            # If the user answers "yes", delete the existing directory and create a new one
            if user_input.lower() in ["yes", "y"]:
                shutil.rmtree(output_folder)  # This will delete the directory along with all its content
                os.makedirs(output_folder)
                print(f"The directory '{output_folder}' has been deleted and a new one has been created.")
            else:
                print("The existing directory will be used.")
    else:
        # If the directory doesn't exist, create it
        os.makedirs(output_folder)
        print(f"The directory '{output_folder}' has been created.")

    # If a parameter file was provided, copy it to the output directory
    if args.parFile:
        shutil.copy2(args.parFile, output_folder)

    #################################################
    # Experiment: K-Fold, Train/Test the networks
    #################################################
    run_polypsizing_experiments(pars, output_folder)
    #################################################
    
