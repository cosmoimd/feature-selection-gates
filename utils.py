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

import json
import os
import re

# Import plot modules
# Import 3rd party modules
import numpy as np
import pandas as pd
# Import PyTorch modules
import torch
# Import built-in modules
import yaml

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
# Import custom modules
from modules.analytics.calculate_metrics import compute_metrics_testing

# File information
__author__ = "Giorgio Roffo, Dr."
__copyright__ = "Copyright 2024, COSMO IMD"
__credits__ = ["Giorgio Roffo"]
__license__ = "Private"
__version__ = "1.0.0"
__maintainer__ = "groffo"
__email__ = "groffo@cosmoimd.com"
__status__ = "Production"  # can be "Prototype", "Development" or "Production"

# Define the order of methods for plotting
method_order = [
    'R18 (RGB)', 'FSG: R18 (RGB)', 'R18 (DPT)', 'FSG: R18 (DPT)', 'R18 (LOC)', 'FSG: R18 (LOC)',
    'MultiStream-R18 [RGB+DPT]', 'FSG: MultiStream-R18 (RGB+DPT)', 'MultiStream-R18 [LOC+DPT]',
    'FSG: MultiStream-R18 (LOC+DPT)', 'ViT (RGB)', 'FSG: ViT (RGB)'
]

# Dictionary to map the raw method names to their new names
method_name_mapping = {
                        'baseline_resnet18_RGB_384': 'R18 (RGB)',
                        'single_input_resnet18_FSAG_RGB_384': 'FSG: R18 (RGB)',
                        'baseline_resnet18_DPT_384': 'R18 (DPT)',
                        'single_input_resnet18_FSAG_DPT_384': 'FSG: R18 (DPT)',
                        'baseline_resnet18_LOC_384': 'R18 (LOC)',
                        'single_input_resnet18_FSAG_LOC_384': 'FSG: R18 (LOC)',
                        'multi_stream_R18_RGB_DPT_384': 'MultiStream-R18 [RGB+DPT]',
                        'double_input_resnet18_FSAG_RGB_DPT_384': 'FSG: MultiStream-R18 (RGB+DPT)',
                        'multi_stream_R18_LOC_DPT_384': 'MultiStream-R18 [LOC+DPT]',
                        'double_input_resnet18_FSAG_LOC_DPT_384': 'FSG: MultiStream-R18 (LOC+DPT)',
                        'baseline_vit_tiny_patch16_384': 'ViT (RGB)',
                        'vit_FSAG_tiny_patch16_384': 'FSG: ViT (RGB)',
                    }

# Dictionary to map the method names to their line color, style, and width
name_color_mapping = {
                        'ViT (RGB)': ('blue', '-', 2),
                        'FSG: ViT (RGB)': ('blue', ':', 4),
                        'R18 (RGB)': ('red', '-', 2),
                        'R18 (LOC)': ('black', '-', 2),
                        'R18 (DPT)': ('orange', '-', 2),
                        'MultiStream-R18 [RGB+DPT]': ('green', '-', 2),
                        'MultiStream-R18 [LOC+DPT]': ('darkred', '-', 2),
                        'FSG: R18 (LOC)': ('black', ':', 4),
                        'FSG: R18 (DPT)': ('orange', ':', 4),
                        'FSG: R18 (RGB)': ('red', ':', 4),
                        'FSG: MultiStream-R18 (RGB+DPT)': ('green', ':', 4),
                        'FSG: MultiStream-R18 (LOC+DPT)': ('darkred', ':', 4)
                      }

# Dictionary to map the method names to their optimal threshold (apply to predictions) - For Testing only
# Thresholds are used to map the continuous predictions to discrete classes Dim, Small, Large.
name_opt_threshold_mapping = {
                        'ViT (RGB)': [4.75, 10.0],
                        'FSG: ViT (RGB)': [4.75, 10.0],
                        'R18 (RGB)': [4.75, 10.0],
                        'R18 (LOC)': [4.75, 10.0],
                        'R18 (DPT)': [4.75, 10.0],
                        'MultiStream-R18 [RGB+DPT]': [4.75, 10.0],
                        'MultiStream-R18 [LOC+DPT]': [4.75, 10.0],
                        'FSG: R18 (LOC)': [4.75, 10.0],
                        'FSG: R18 (DPT)': [4.75, 10.0],
                        'FSG: R18 (RGB)': [4.75, 10.0],
                        'FSG: MultiStream-R18 (RGB+DPT)': [4.75, 10.0],
                        'FSG: MultiStream-R18 (LOC+DPT)': [4.75, 10.0],
                      }

MIN_TRAIN_EPOCHS = 0  # Maximum Training Epochs
CLASSES = ["Dim", "Small", "Large"]  # Classes of polyp sizes

# Global pattern for checkpoints
pattern = r'epoch-(\d+)_valid_acc-(\d+\.\d+)_loss-(\d+\.\d+)_checkpoint\.pth'

def find_checkpoint(path, criterion='min_loss'):
    """
    Searches for checkpoint files within a given directory, identifying the file with the minimum loss value or
    maximum validation accuracy based on the specified criterion.

    Args:
    - path (str): The directory path where checkpoint files are located. These files should follow a naming
      convention like 'epoch-0_valid_acc-0.3332_loss-0.0451_checkpoint.pth', where the epoch number, validation
      accuracy, and loss values are variable.
    - criterion (str): The criterion for selecting the checkpoint file. It can be 'min_loss' or 'max_valid_acc'.

    Returns:
    - tuple: A tuple containing the selected value (as a float), the epoch number (as an integer), and the full
      filename of the checkpoint file. If no file matching the expected naming pattern is found, raises FileNotFoundError.
    """

    # Initialize variables for both criteria
    min_loss = float('inf')
    max_valid_acc = 0.0
    epoch_of_selected = -1
    selected_filename = None

    # Iterate over all files in the given path
    for filename in os.listdir(path):
        match = re.match(pattern, filename)
        if match:
            epoch, valid_acc, loss = int(match.group(1)), float(match.group(2)), float(match.group(3))
            if criterion == 'min_loss':
                if loss < min_loss and epoch > MIN_TRAIN_EPOCHS:
                    min_loss = loss
                    epoch_of_selected = epoch
                    selected_filename = filename
            elif criterion == 'max_valid_acc':
                if valid_acc > max_valid_acc and epoch > MIN_TRAIN_EPOCHS:
                    max_valid_acc = valid_acc
                    epoch_of_selected = epoch
                    selected_filename = filename

    if selected_filename is None:
        raise FileNotFoundError(f"No file matching the criteria was found in directory {path}")

    selected_value = min_loss if criterion == 'min_loss' else max_valid_acc
    return selected_value, epoch_of_selected, selected_filename

def load_epoch_csv(epoch_of_min_loss, dir_path, fold_num):
    """
    Loads a CSV file corresponding to a specific epoch number into a pandas DataFrame.

    The CSV file is expected to follow a naming convention like 'valid_inference_results_{epoch}_kfold_0.csv',
    where {epoch} is the epoch number.

    Args:
    - epoch_of_min_loss (int): The epoch number used to identify the specific CSV file.
    - dir_path (str): The directory path where the CSV files are located.
    - fold_num (int): number of the fold integer

    Returns:
    - DataFrame: A pandas DataFrame containing the data from the specified CSV file. If the file does not exist,
      returns None.

    Raises:
    - FileNotFoundError: If no file matching the constructed filename is found in the specified directory.
    """

    # Construct the filename based on the given epoch number
    filename = f'test_inference_results_epoch_min_loss_{epoch_of_min_loss}_kfold_{fold_num}.csv'
    full_path = os.path.join(dir_path, filename)

    # Check if the file exists before attempting to read
    if os.path.exists(full_path):
        # Read the CSV into a DataFrame and return it
        df = pd.read_csv(full_path)
        return df
    else:
        # Raise an error if the file does not exist
        raise FileNotFoundError(f"No file found for epoch {epoch_of_min_loss} in directory {dir_path}")


class Scaler:
    def __init__(self, feature_range=(-1, 1), polypsizing=True):
        self.polypsizing = polypsizing
        self.min, self.max = feature_range
        self.range = self.max - self.min
        self.min_data = 0.5
        self.max_data = 20.0
        self.data_range = self.max_data - self.min_data

    def scale(self, target):
        """
        Scales the target values to the feature range.

        Args:
            target (torch.Tensor): A tensor containing the target values.

        Returns:
            torch.Tensor: A tensor containing the scaled target values.
        """
        if self.polypsizing:
            scaled = ((target - self.min_data) / self.data_range) * self.range + self.min
            return scaled.type(torch.float32)
        else:
            return target

    def inverse_scale(self, target):
        """
        Scales the target values back to the original range.

        Args:
            target (torch.Tensor): A tensor containing the target values.

        Returns:
            torch.Tensor: A tensor containing the target values in the original range.
        """
        if self.polypsizing:
            unscaled = ((target - self.min) / self.range) * self.data_range + self.min_data
            unscaled = torch.where(unscaled < self.min_data, self.min_data * torch.ones_like(unscaled), unscaled)

            unscaled = torch.where(unscaled > self.max_data * 1.10, self.max_data * 1.10 * torch.ones_like(unscaled),
                                   unscaled)

            return unscaled.type(torch.float32)
        else:
            return target


def sort_as_mapping_order(df, name_mapping):
    """
    Sorts a DataFrame based on a specified mapping of method names, ensuring that
    the DataFrame rows are ordered according to the sequence of methods defined in the mapping.

    Args:
    df (pd.DataFrame): The input DataFrame containing a column with method names that need to be sorted.
                       This DataFrame must have a 'Method Name' column.
    name_mapping (dict): A dictionary mapping from raw method names (as they appear in the DataFrame)
                         to new names. The order of methods in this mapping dictates the order
                         in which rows should be sorted in the output DataFrame.

    Returns:
    pd.DataFrame: A DataFrame sorted according to the order specified in name_mapping. The 'Method Name'
                  column in the returned DataFrame will contain the new names as per the mapping, and the
                  rows will be ordered according to the sequence of these new names.

    The function first maps the raw method names to their new names as specified by the 'name_mapping' argument.
    It then creates a list of these new names ordered according to their appearance in the 'name_mapping' dictionary.
    This ordered list is used to define a categorical data type in pandas, which ensures that when the DataFrame
    is sorted, it respects this specific order. Finally, the DataFrame is sorted, the original 'Method Name' column
    is replaced with the new names, and the sorted DataFrame is returned.
    """
    # Apply the name mapping to the 'Method Name' column
    df['Mapped Name'] = df['Method Name'].map(name_mapping)

    # Create an ordered list of the mapped names based on the order in the name_mapping dictionary
    ordered_mapped_names = list(name_mapping.values())

    # Create a categorical type with the ordered list, ensuring that the sort_values method respects this order
    df['Mapped Name'] = pd.Categorical(df['Mapped Name'], categories=ordered_mapped_names, ordered=True)

    # Sort the dataframe based on the 'Mapped Name' column
    df_sorted = df.sort_values(by='Mapped Name')

    # Drop the original 'Method Name' column and rename 'Mapped Name' to 'Method Name'
    df_sorted.drop(columns=['Method Name'], inplace=True)
    df_sorted.rename(columns={'Mapped Name': 'Method Name'}, inplace=True)

    return df_sorted


def group_and_sort_by_metric(df, name_mapping):
    """
    Groups the DataFrame by the 'Metric' column and sorts each group according to the specified method name mapping.

    Args:
    df (pd.DataFrame): The input DataFrame containing a 'Metric' column and a 'Method Name' column.
                       The DataFrame may contain various metrics, each of which will be sorted individually.
    name_mapping (dict): A dictionary mapping from raw method names (as they appear in the DataFrame)
                         to new names. The sorting order within each metric group is determined by the sequence
                         of methods defined in this mapping.

    Returns:
    pd.DataFrame: A new DataFrame where each metric group has been sorted according to the method name mapping.
                  The DataFrame maintains the grouping by 'Metric', but within each group, the rows are sorted
                  according to the specified order of methods.

    The function first groups the DataFrame by the 'Metric' column. Then, for each group, it applies the
    'sort_as_mapping_order' function to sort the rows based on the method name mapping. Finally, the sorted groups
    are concatenated back into a single DataFrame, which is returned.
    """
    # Group the DataFrame by the 'Metric' column
    grouped = df.groupby('Metric')

    # Initialize an empty list to hold the sorted DataFrames
    sorted_groups = []

    # For each group, apply the sort_as_mapping_order function and add the result to the list
    for _, group in grouped:
        sorted_group = sort_as_mapping_order(group, name_mapping)
        sorted_groups.append(sorted_group)

    # Concatenate the sorted groups into a single DataFrame
    sorted_df = pd.concat(sorted_groups).reset_index(drop=True)

    return sorted_df

def map_to_class_gt(value):
    """
    Classifies a numerical value into one of three distinct categories ("Dim", "Small", or "Large")
    based on its magnitude relative to two fixed thresholds. The function is tailored to work with
    both predictions and ground truth values, offering a standardized method for categorization.

    Args:
        value (float): The numerical value to be classified. It should be a single floating-point
                       number, typically used to represent a prediction or an actual measurement.

    Returns:
        str: The category of the input value determined by its size:
             - "Dim": for values less than or equal to 5.0, indicating a lower range.
             - "Small": for values greater than 5.0 but less than 10.0, denoting a middle range.
             - "Large": for values greater than or equal to 10.0, signifying a higher range.
    """
    if value <= 5.0:
        return "Dim"
    elif 5.0 < value < 10.0:
        return "Small"
    else:
        return "Large"


def map_to_classes_binary_gt(value):
    """
    Maps a numerical input to one of two possible classes based on its magnitude.
    The classification is binary and determined by predefined thresholds, facilitating
    the distinction between smaller and larger values.

    Args:
        value (float): The numerical value to be classified. It should be a floating-point
                       number representing either a measurement, a ground truth value,
                       or a prediction output from a model.

    Returns:
        str: Returns "DS" if the input value is less than 10.0, indicating a class that
             encompasses both "Dim" and "Small" characteristics as per the original categorization.
             Returns "Large" for values greater than or equal to 10.0, aligning with the original
             classification's notion of largeness.
    """
    return "DS" if value < 10.0 else "Large"


def map_to_class_pred(value, threshold1=5.0, threshold2=10.0):
    """
    Classifies a numerical value into one of three categories ("Dim", "Small", or "Large")
    based on comparison with two predefined thresholds. The function is versatile and can
    handle both ground truth and prediction values, offering a clear categorization that
    facilitates further analysis or processing.

    Args:
        value (float): The value to be classified. It must be a single floating-point number,
                       typically representing a measurement, ground truth, or prediction.
        threshold1 (float, optional): The first threshold used for classification. Defaults to 5.0.
        threshold2 (float, optional): The second threshold used for classification. Defaults to 10.0.

    Returns:
        str: The classification of the input value as either "Dim" (≤ threshold1), "Small"
             (> threshold1 and < threshold2), or "Large" (≥ threshold2).
    """
    if value <= threshold1:
        return "Dim"
    elif threshold1 < value < threshold2:
        return "Small"
    else:
        return "Large"

def map_to_classes_binary_pred(value, threshold=10.0):
    """
    Classifies a numerical value into one of two categories based on a single predefined threshold.
    The function is designed to handle numerical data representing either a prediction or a ground truth,
    simplifying the categorization into a binary scheme for ease of analysis.

    Args:
        value (float): The value to be classified, expected to be a floating-point number indicative of
                       a measurable outcome, ground truth, or predictive result.
        threshold (float, optional): The threshold value used to delineate the two categories.
                                     Defaults to 10.0.

    Returns:
        str: The classification of the input value as "DS" for values below the threshold,
             encompassing both traditionally "Dim" and "Small" categories, or "Large" for
             values equal to or exceeding the threshold.
    """
    return "DS" if value < threshold else "Large"


def load_yml(file):
    """
    Loads parameters from a YAML file.

    Args:
    file (str): Path to the YAML file.

    Returns:
    dict: A dictionary containing the parameters.
    """
    with open(file, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


# Computes and stores the average and current value
class AverageMeter(object):
    """
        Utility class for updating and estimating the accuracy during training.
        This class is used in the training loop or testing/inference to keep track of some metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        # Check for NaN gradients
        if torch.isnan(p.grad.data).any() or torch.isinf(p.grad.data).any():
            continue

        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def create_and_save_subplots(input_1, masked_rgb, expanded_binary_mask, depth_map, save_dir, prog_num):
    """
    Create subplots for the given tensors, loop over the batch size, and save the images to the specified directory.
    Also, save the images RGB, depth separately in jpg format.
    """

    def clip_and_permute(tensor):
        return tensor.clamp(0, 1).permute(1, 2, 0).numpy()

    # Move tensors from GPU to CPU and detach from the computation graph
    input_1 = input_1.cpu().detach()
    masked_rgb = masked_rgb.cpu().detach()
    expanded_binary_mask = expanded_binary_mask.cpu().detach()

    # Check if depth_map is not None and process accordingly
    if depth_map is not None:
        depth_map = depth_map.cpu().detach()
    else:
        # Create a dummy depth map with zeros
        depth_map = torch.zeros_like(masked_rgb[:, 0:1, :, :])

    # Create the save directory
    os.makedirs(save_dir, exist_ok=True)

    # Loop over the batch size
    batch_size = input_1.shape[0]
    for i in range(batch_size):
        cur_input_1 = clip_and_permute(input_1[i])
        cur_masked_rgb = clip_and_permute(masked_rgb[i])
        cur_expanded_binary_mask = clip_and_permute(expanded_binary_mask[i])
        cur_depth_map = depth_map[i].squeeze().numpy()

        # Normalize the depth map
        if np.max(cur_depth_map) != np.min(cur_depth_map):
            cur_depth_map = (cur_depth_map - np.min(cur_depth_map)) / (np.max(cur_depth_map) - np.min(cur_depth_map))

        # Create and save subplots
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(cur_input_1)
        axs[0, 0].axis('off')
        axs[0, 0].set_title('RGB + Augs')

        axs[0, 1].imshow(cur_masked_rgb)
        axs[0, 1].axis('off')
        axs[0, 1].set_title('Location Map RGB')

        axs[1, 0].imshow(cur_expanded_binary_mask)
        axs[1, 0].axis('off')
        axs[1, 0].set_title('Expanded Loc. Mask')

        axs[1, 1].imshow(cur_depth_map, cmap='gray')
        axs[1, 1].axis('off')
        axs[1, 1].set_title('Depth Map')

        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.suptitle('Multi-Stream Inputs')
        save_path = os.path.join(save_dir, f"batch_{prog_num}_{i + 1}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        # Save individual images (RGB, depth) in JPG format
        plt.imsave(os.path.join(save_dir, f"batch_{prog_num}_{i + 1}_rgb.jpg"), cur_input_1)
        if np.max(cur_depth_map) != 0:  # Save depth image only if it's not all zeros
            plt.imsave(os.path.join(save_dir, f"batch_{prog_num}_{i + 1}_depth.jpg"), cur_depth_map, cmap='gray')

    print("Images and subplots saved successfully.")


def is_bbox_centered(bbox, center_ratio=0.7, mode="center"):
    """
    Check if the bounding box is within the center of the image.

    Args:
        bbox (tuple): A tuple (x_center, y_center, width, height) representing the bounding box.
                      All values are relative to the image size and normalized to [0, 1].
        center_ratio (float): The ratio of the image to consider as the center.
                              For example, 0.7 means the center 70% of the image.
        mode (str): The mode to use for checking if the bounding box is centered.
                    If "center", checks if the center of the bounding box is within the center of the image.
                    If "edges", checks if any edges of the bounding box are within the center of the image.

    Returns:
        bool: True if the bounding box is considered to be centered, False otherwise.

    Raises:
        ValueError: If mode is not "center" or "edges".
    """

    x_center, y_center, width, height = bbox  # unpack the bounding box
    radius = center_ratio / 2  # calculate the radius of the circle

    def in_circle(x, y):
        # checks if a point is inside the circle
        return np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) <= radius

    if mode == "center":  # if mode is "center"
        # check if the center of the bounding box is within the center of the image
        return in_circle(x_center, y_center)
    elif mode == "edges":  # if mode is "edges"
        # calculate the coordinates of the corners of the bounding box
        x_left = x_center - width / 2
        x_right = x_center + width / 2
        y_top = y_center - height / 2
        y_bottom = y_center + height / 2
        # check if all corners of the bounding box are within the center of the image
        return in_circle(x_left, y_top) and in_circle(x_right, y_top) and in_circle(x_left, y_bottom) and in_circle(
            x_right, y_bottom)
    else:  # if mode is neither "center" nor "edges"
        # raise an error
        raise ValueError(f"Invalid mode: {mode}. Valid modes are 'center' and 'edges'.")


def filter_dirs_by_kfold_presence(method_dirs, base_path, num_folds):
    """
    Filters a list of directories, keeping only those that contain a specific set of subfolders named 'kfold_i',
    where 'i' ranges from 0 to num_folds-1.

    Args:
    - method_dirs (list of str): A list of directory names to check for the presence of kfold subfolders.
    - base_path (str): The base directory path where the method directories are located.
    - num_folds (int): The number of kfold subfolders required for a directory to be considered valid.

    Returns:
    - list of str: A filtered list of directory names from the original list that contain all required kfold subfolders.

    Note:
    - This function does not verify the content of kfold subfolders, only their existence.
    - Directories that do not exist or do not contain all required kfold subfolders are excluded from the return list.
    """

    filtered_dirs = []  # Initialize an empty list to hold directories that pass the filter criteria

    for method_dir in method_dirs:
        # Construct the full path to the method directory
        full_path = os.path.join(base_path, method_dir)

        # Check if all kfold_i subfolders exist within the current method directory
        all_folds_exist = all(os.path.isdir(os.path.join(full_path, f'kfold_{i}')) for i in range(num_folds))

        # If the directory contains all required kfold_i subfolders, add it to the filtered list
        if all_folds_exist:
            filtered_dirs.append(method_dir)

    return filtered_dirs


def filter_method_directories(method_dirs, method_name_mapping):
    """
    Filters a list of directory names based on a mapping dictionary.

    Args:
        method_dirs (list): A list of directory names (strings) to be filtered.
        method_name_mapping (dict): A dictionary where keys are raw method names that
                                    correspond to some entries in method_dirs, and
                                    values are the new names for these methods.

    Returns:
        list: A filtered list of directory names that exist as keys in the
              method_name_mapping dictionary.
    """
    # Filter the method_dirs list by checking if each directory is a key in method_name_mapping
    filtered_dirs = [dir_name for dir_name in method_dirs if dir_name in method_name_mapping.keys()]

    return filtered_dirs

def concatenate_csv_best_valid_loss(base_path, num_folds, output_dir):
    """
     The function concatenates CSV files across the 'kfold_x' directories for each of
     the experiment/method directories present in the specified base path. An extra column named 'kfold_id'
     is added to the resultant dataframe to identify which kfold the data belongs to.

     The concatenated CSVs are saved in a sub-directory named output_dir within the base path.

     Args:
     - base_path (str): The root directory where the experiment/method directories are located.
     - num_folds (int): number of folds used in the experiments

     Returns:
     - str: Path to the output_dir directory where the concatenated CSV files are saved.
     - methods_performance_summary: A CSV DataFrame with a summary for each method and fold
     """
    # List all directories in base_path
    method_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # Filters method_dirs list of directories, keeping only those that exist as keys in the method_name_mapping dictionary.
    method_dirs = filter_method_directories(method_dirs, method_name_mapping)

    # Filters method_dirs list of directories, keeping only those that contain a specific set of subfolders named 'kfold_i',
    # where 'i' ranges from 0 to num_folds-1.
    method_dirs = filter_dirs_by_kfold_presence(method_dirs, base_path, num_folds)

    # Creating a sorted list based on the order of keys in the dictionary method_name_mapping
    method_dirs = sorted(method_dirs,
                         key=lambda x: list(method_name_mapping.keys()).index(x) if x in method_name_mapping else len(
                             method_name_mapping))

    print(f"+ Total number of methods available: {len(method_dirs) - 1} \n")  # subtracting 1 for output_dir directory

    # methods_performance_summary DataFrame is defined here
    methods_performance_summary = pd.DataFrame([])

    for method in method_dirs:
        if (method != output_dir and method != 'ignore' and "RESULTS" not in method):

            # Create an empty dataframe to hold concatenated data for this method
            allfolds_in_df = pd.DataFrame()

            # methods_performance_summary DataFrame is defined here
            methods_performance_summary = pd.DataFrame(columns=[
                'method_name', 'kfold_id', 'epoch_of_min_loss', 'Validation Loss',
                'Balanced Binary Accuracy', 'Balanced Accuracy', 'Sensitivity',
                'Specificity', 'balanced_performance', 'Precision', 'Recall', 'F1 Score'
            ])

            # Loop through all kfold directories
            for k in range(num_folds):
                ckpt_dir = os.path.join(base_path, method, f'version-{k}')

                if not os.path.exists(ckpt_dir):
                    print(f"Method '{method}' does not have '{ckpt_dir}' directory. Skipping...")
                    continue  # Skip the current method and proceed to the next

                print(f"Method '{method}'")

                # Searches for checkpoint files within ckpt_dir directory, identifying the file with the minimum loss value.
                min_loss, epoch_of_min_loss, ckpt_filename = find_checkpoint(ckpt_dir, criterion='min_loss')

                kfold_dir = os.path.join(base_path, method, f'kfold_{k}')
                if not os.path.exists(kfold_dir):
                    print(f"Method '{method}' does not have '{kfold_dir}' directory. Skipping...")
                    continue  # Skip the current method and proceed to the next

                # Loads a CSV file corresponding to a specific epoch number into a pandas DataFrame.
                test_df = load_epoch_csv(epoch_of_min_loss, kfold_dir, fold_num=k)

                # Retrieve the optimal thresholds for the method
                T1, T2 = name_opt_threshold_mapping[method_name_mapping[method]]

                test_df['gt_class'] = test_df['gt'].apply(map_to_class_gt)
                test_df['pred_class'] = test_df['pred'].apply(lambda x: map_to_class_pred(x, T1, T2))

                test_df['gt_bin_lass'] = test_df['gt'].apply(map_to_classes_binary_gt)
                test_df['pred_bin_class'] = test_df['pred'].apply(lambda x: map_to_classes_binary_pred(x, T2))

                # Add kfold ID to the dataframe
                test_df['kfold_id'] = k

                # Compute classification metrics
                _, _, _, _, _, _, balanced_bin_accuracy, bin_accuracy = compute_metrics_testing(test_df['gt_bin_lass'], test_df['pred_bin_class'])

                # Compute metrics
                sensitivity, specificity, balanced_performance, precision, recall, f1, balanced_acc, dsl_accuracy = compute_metrics_testing(
                    test_df['gt_class'], test_df['pred_class'])

                # Print out the metrics
                print(f"Classification Metrics - Best Epoch {epoch_of_min_loss} [{method.upper()}]:")
                print("-----------------------------------------------------------------------------")
                print(f"Validation Loss: {min_loss:.4f}")
                print(f"Balanced Binary Accuracy: {balanced_bin_accuracy:.4f}")
                print(f"Balanced Accuracy: {balanced_acc:.4f}")
                print(f"Sensitivity: {sensitivity:.4f}")
                print(f"Specificity: {specificity:.4f}")
                print(f"balanced_performance: {balanced_performance:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")

                # Create a new row as a dictionary
                new_perf_row = pd.DataFrame([{
                    'method_name': method,
                    'kfold_id': k,
                    'epoch_of_min_loss': epoch_of_min_loss,
                    'Validation Loss': min_loss,
                    'Balanced Binary Accuracy': balanced_bin_accuracy,
                    'Balanced Accuracy': balanced_acc,
                    'Sensitivity': sensitivity,
                    'Specificity': specificity,
                    'balanced_performance': balanced_performance,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                }])

                methods_performance_summary = pd.concat([methods_performance_summary, new_perf_row], ignore_index=True)
                allfolds_in_df = pd.concat([allfolds_in_df, test_df], ignore_index=True)

            # Save the concatenated CSV to the base directory
            if not allfolds_in_df.empty and not methods_performance_summary.empty:
                exp_filename1 = f'{method}_inference_results.csv'
                concatenated_csv_path = os.path.join(base_path, output_dir, exp_filename1)
                allfolds_in_df.to_csv(concatenated_csv_path, index=False)
                print(f"Saved INFERENCE for method '{method}' to {concatenated_csv_path}")

                exp_filename2 = f'{method}_summary.csv'
                concatenated_csv_path = os.path.join(base_path, output_dir, exp_filename2)
                methods_performance_summary.to_csv(concatenated_csv_path, index=False)
                print(f"Saved SUMMARY for method '{method}' to {concatenated_csv_path}")

                # Print details
                print(f"\nDetails for method '{method}':")
                print(f"CSV File Name: {exp_filename1}")
                print(f"CSV File Name: {exp_filename2}")
                print(f"Number of samples: {len(allfolds_in_df)}")
                print("-----------------------------------------------------------------------------")

    return os.path.join(base_path, output_dir), methods_performance_summary


def generate_pdf_and_plot(df, output_folder, experiment_name, method_color_style_mapping):
    """
    Generate a PDF report containing bias and variance details in a table from a DataFrame,
    and plot performance metrics with specified line styles.

    Args:
    - df (DataFrame): Dataframe with bias variance analysis.
    - output_folder (str): The path to the folder where the PDF will be saved.
    - experiment_name (str): Name of the experiment, used to name the output file.
    - method_color_style_mapping (dict): A dictionary specifying colors and line styles for plotting.

    Returns:
    None. The report is saved in the specified output folder, and the plot is displayed.
    """
    # Save the DataFrame directly to a PDF
    pdf_path = os.path.join(output_folder, f"{experiment_name.replace(' ', '_')}_bias_variance_report.pdf")
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(12, 4))  # figsize can be adjusted to your needs
        ax.axis('off')  # Hide axes
        df_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    # Create a DF for each metric (f1, balanced accuracy, etc...)
    grouped_metrics_df = df.groupby('Metric')
    # Iterate through each group and print it
    for metric_name, group in grouped_metrics_df:
        # Plot performance metrics
        metrics = group.columns[1:-1]
        plt.figure(figsize=(20, 10))
        for method in df['Method Name'].unique():
            method_data = df[df['Method Name'] == method]
            values = method_data.iloc[0][metrics].values
            color, style, width = method_color_style_mapping.get(method, ('black', '-', 1))
            plt.plot(metrics, values, label=method, color=color, linestyle=style, linewidth=width)

        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title(f'{experiment_name} Performance Metrics Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Optionally, save the plot as an image file
        plot_output_file = f"{output_folder}/{experiment_name.replace(' ', '_')}_performance_metric_{metric_name}.png"
        plt.savefig(plot_output_file)
        plt.close()
        plt.close('all')