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

"""
This script is designed to pre-process the RAW datasets.
It operates by reading the dataset and creating a CSV file that only includes frames that contain polyps (objects).
The script accepts a parameter file that specifies dataset details such as path,
output folder, and more. If no parameter file is provided, it will look for a default one.
"""


# Import built-in modules
import os
import shutil
import argparse

# Import custom modules
from utils import load_yml
from modules.datasets.dataset import IMDRealColonDatasetPreprocessor, SUNDatasetPreprocessor, IMDKfoldSplitter, \
    IMDDataStats

# Import PyTorch modules
# Import plot modules
# Import 3rd party modules

# File information
__author__ = "Giorgio Roffo, Dr."
__copyright__ = "Copyright 2024, COSMO IMD"
__credits__ = ["Giorgio Roffo"]
__license__ = "Private"
__version__ = "1.0.0"
__maintainer__ = "groffo"
__email__ = "groffo@cosmoimd.com"
__status__ = "Production"  # can be "Prototype", "Development" or "Production"


if __name__ == "__main__":
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

    # Get the output folder path from the configuration
    output_folder = pars['dataset_params']['dataset_csv_path']
    build_dataset = False # Flag set to True to build or Re-Build the dataset

    # Check if the output folder already exists
    if os.path.exists(output_folder):
        # If the folder exists, ask the user if they want to delete it and create a new one
        confirmation = input(
            f"The folder {output_folder} already exists. Do you want to remove it and create a new one? [y/n]: ")

        # If the user confirms, delete the existing folder and create a new one
        if confirmation.lower() == 'y':
            build_dataset = True  # to True to Re-Build the dataset
            # Delete the existing directory and all its contents
            shutil.rmtree(output_folder)
            # Create a new directory
            os.makedirs(output_folder)
            # Print a confirmation message
            print(f"Folder {output_folder} has been recreated.")
        else:
            build_dataset = False  # Using existing dataset
            # If the user does not confirm, print a message and continue using the existing folder
            print("Using existing folder.")
    else:
        build_dataset = True  # to True to Build the new dataset
        # If the folder does not exist, create it
        os.makedirs(output_folder)
        # Print a confirmation message
        print(f"Folder {output_folder} has been created.")

    # Flag set to True to build or Re-Build the dataset
    if build_dataset:
        # Iterate over all specified datasets in the parameters.
        for dataset in pars['dataset_params']['datasets']:

            # Extract dataset-specific parameters.
            # NAME,CSV_FILE,PATH_FRAMES_FOLDERS,FORMAT_FOLDERS,GT_NAME,OBJ_ID
            name, csv_path, frames_folders_path, format, gt_column, object_id_column = dataset

            # If the dataset is from COSMO, process it.
            if 'REAL' in name:

                # Instantiate dataset preprocessor with the path to frames and the output folder.
                dataset_processor = IMDRealColonDatasetPreprocessor(base_dataset_path=frames_folders_path,
                                                                    dataset_tuple=dataset,
                                                                    output_folder=output_folder,
                                                                    saving_step=pars['dataset_params']['saving_step'])
                # Preprocess the dataset.
                print('#' * 150)
                print('# IMDRealColonDatasetPreprocessor Initialization ')
                print(f'\n> The IMDRealColonDatasetPreprocessor will perform the following steps:\n'
                      f'  1. Traverse all directories and XML annotations in the base dataset path.\n'
                      f'  2. For each XML file, extract the object data and append it to a DataFrame.\n'
                      f'  3. When the DataFrame has {dataset_processor.saving_step} rows, it will be saved to a CSV file in the output folder.\n'
                      f'  4. This process will be repeated until all XML files have been processed.\n')
                print('#' * 150)
                print(f'> Base dataset path: {frames_folders_path}')
                print(f'> Output folder: {output_folder}')
                print(f'> Saving step (rows per file): {pars["dataset_params"]["saving_step"]}')
                print(f'> Column names: {dataset_processor.columns}')

                print(f'> Example of a resulting CSV file name: "polypsizing_cosmo_imd_part1.csv"\n'
                      f'> Each CSV file will have the following columns: {", ".join(dataset_processor.columns)}\n')
                print('#' * 150)
                print('> Starting the IMDRealColonDatasetPreprocessor')

                dataset_processor.process_dataset()

                print('\n> IMDRealColonDatasetPreprocessor completed successfully. All preprocessed data have been saved.')
                print(f'> Please check the output folder ({dataset_processor.output_folder}) for the CSV files.')
                print('#' * 150)

            # If the dataset is SUN, process it.
            elif 'SUN' in name:

                # Instantiate dataset preprocessor with the path to frames and the output folder.
                dataset_processor = SUNDatasetPreprocessor(base_dataset_path=frames_folders_path,
                                                                    dataset_tuple=dataset,
                                                                    output_folder=output_folder,
                                                                    saving_step=pars['dataset_params']['saving_step'])
                # Preprocess the dataset.
                print('#' * 150)
                print('# SUNDatasetPreprocessor Initialization ')
                print(f'\n> The SUNDatasetPreprocessor will perform the following steps:\n'
                      f'  1. Traverse all directories and XML annotations in the base dataset path.\n'
                      f'  2. For each XML file, extract the object data and append it to a DataFrame.\n'
                      f'  3. When the DataFrame has {dataset_processor.saving_step} rows, it will be saved to a CSV file in the output folder.\n'
                      f'  4. This process will be repeated until all XML files have been processed.\n')
                print('#' * 150)
                print(f'> Base dataset path: {frames_folders_path}')
                print(f'> Output folder: {output_folder}')
                print(f'> Saving step (rows per file): {pars["dataset_params"]["saving_step"]}')
                print(f'> Column names: {dataset_processor.columns}')

                print(f'> Example of a resulting CSV file name: "polypsizing_sun_part1.csv"\n'
                      f'> Each CSV file will have the following columns: {", ".join(dataset_processor.columns)}\n')
                print('#' * 150)
                print('> Starting the SUNDatasetPreprocessor')

                dataset_processor.process_dataset()

                print('\n> SUNDatasetPreprocessor completed successfully. All preprocessed data have been saved.')
                print(f'> Please check the output folder ({dataset_processor.output_folder}) for the CSV files.')
                print('#' * 150)
            else:
                print('The parameter file does not contain any settings for known datasets.')


    ######################################
    # Create the lists - K-Fold
    ######################################
    if build_dataset:
        print(f"Creating new splits. Generating {pars['dataset_params']['kfold']} new folds.")
        splitter = IMDKfoldSplitter(output_folder, num_splits=pars['dataset_params']['kfold'])
        splitter.create_splits(mode='per_object')

    stats_generator = IMDDataStats(dataset_folder=output_folder, mode='per_object')
    stats_generator.plot_distribution()
    stats_generator.plot_distribution_per_unique_id()
    stats_generator.show_stats_per_unique_id()

    stats_generator.show_stats()


