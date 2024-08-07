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

import glob
# Import built-in modules
import os
import re
import xml.etree.ElementTree as ET

# Import plot modules
import matplotlib.pyplot as plt
# Import 3rd party modules
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.model_selection import StratifiedKFold
# Import PyTorch modules
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import datasets
import utils
from utils import map_to_class_gt

# Import custom modules

# File information
__author__ = "Giorgio Roffo, Dr."
__copyright__ = "Copyright 2024, COSMO IMD"
__credits__ = ["Giorgio Roffo"]
__license__ = "Private"
__version__ = "1.0.0"
__maintainer__ = "groffo"
__email__ = "groffo@cosmoimd.com"
__status__ = "Production"  # can be "Prototype", "Development" or "Production"


class CIFAR100Dataset(Dataset):
    def __init__(self, params, phase='training', test_id=0, num_folds=6, transform=None, mode='per_object'):

        root = params['dataset_csv_path']
        train = True if phase == 'training' else False

        download = False

        self.data = datasets.CIFAR100(root=root, train=train, download=download, transform=None,
                                      target_transform=None)
        self.transform = transform
        self.target_transform = None
        self.targets = self.data.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        if self.target_transform:
            label = self.target_transform(label)


        # Convert to PyTorch tensors
        image = ToTensor()(image)  # normalize to [0,1]

        # take all the image - no bboxes in this task
        loc_mask = np.ones((image.shape[-2], image.shape[-1]))

        loc_mask = ToTensor()(loc_mask)  # normalize to [0,1]

        if self.transform:
            image, loc_mask = self.transform(image, loc_mask)

        return image, loc_mask, label, idx

####################################
# Polyp Sizing - Dataset Class
####################################
class PolypSizeEstimationDataset(Dataset):
    """
    A PyTorch Dataset class for polyp size estimation tasks.

    Args:
        params (dict): A dictionary of parameters, including:
            - output_folder (str): The path to the directory where output files will be stored.
            - precision_float_accuracy (float): The precision/accuracy of the float data.
            - target_process_min (float): The minimum size for target processing.
            - target_process_max (float): The maximum size for target processing.
            - n_frames_to_sample (int): The number of frames to sample from each video.
            - batch_size (int): The number of samples per gradient update.
            - input_dim_resize_to (int): The dimension to which the input images will be resized.
            - filter_centered_samples_ratio (float): The ratio for filtering out the samples whose bounding boxes are not considered centered.
            - image_preprocessing (dict): The parameters for preprocessing the images (e.g., normalization, circular cropping).
            - include_depth_maps (bool): Whether or not to include depth maps in the dataset.
            - include_rgb_mask (bool): Whether or not to include RGB masks in the dataset.
            - num_workers (int): The number of worker threads for loading the data.
            - datasets (list): The list of datasets to load. Each dataset is specified by a list containing its name,
            the path to its CSV file, and the name of the GT column.

    Returns:
        A PyTorch Dataset object for polyp size estimation.
    """
    def __init__(self, params, phase='training', test_id=0, num_folds=6, transform=None, mode='per_object'):

        assert phase in ['train', 'training', 'testing', 'test', 'validation', 'valid']

        self.params = params

        self.transform = transform
        self.dataframe = self.load_dataframe(params['dataset_csv_path'], params['datasets'])
        print(f"> Dataset duplicated items: {self.dataframe.duplicated(subset=['unique_id', 'frame_id']).sum()}")

        if mode == 'per_frame':
            file_name = 'kfold_polypsizing_lists_per_frame.csv'
        else:
            file_name = 'kfold_polypsizing_lists_per_object.csv'

        # Load the lists, kfold splits.
        lists = pd.read_csv(os.path.join(params['dataset_csv_path'], file_name))
        print(f"> Lists duplicated items: {lists.duplicated(subset=['unique_id', 'frame_id']).sum()}")

        assert self.dataframe.duplicated(subset=['unique_id', 'frame_id']).sum() == 0
        assert lists.duplicated(subset=['unique_id', 'frame_id']).sum() == self.dataframe.duplicated(subset=['unique_id', 'frame_id']).sum()


        # Convert the 'unique_id' and 'frame_id' columns to strings for both dataframes
        self.dataframe['unique_id'] = self.dataframe['unique_id'].astype(str)
        self.dataframe['frame_id'] = self.dataframe['frame_id'].astype(str)

        lists['unique_id'] = lists['unique_id'].astype(str)
        lists['frame_id'] = lists['frame_id'].astype(str)

        # Merge the relevant columns from the lists dataframe into self.dataframe based on 'unique_id' and 'frame_id'
        self.dataframe = self.dataframe.merge(
            lists[['unique_id', 'frame_id', 'gt_group', 'kfold_id']],
            on=['unique_id', 'frame_id'],
            how='left'
        )
        #     When k is 0, the validation set is fold 1.
        #     When k is 1, the validation set is fold 2.
        #     ...
        #     When k is 4, the validation set is fold 5.
        #     When k is 5 (the last fold), the validation set wraps around to fold 0.
        valid_id = (test_id + 1) % num_folds

        # Filter the dataframe based on the phase
        if phase in ['train', 'training']:
            self.dataframe = self.dataframe[(lists['kfold_id'] != test_id) & (lists['kfold_id'] != valid_id)]
        elif phase in ['valid', 'validation']:
            self.dataframe = self.dataframe[lists['kfold_id'] == valid_id]
        elif phase in ['test', 'testing']:
            self.dataframe = self.dataframe[lists['kfold_id'] == test_id]

        # Filter the dataframe after k-folding...
        if params['filter_centered_samples_ratio'] < 1.0:
            self.dataframe = self._filter_centered_samples(center_ratio=params['filter_centered_samples_ratio'], mode="center")

        # Get the list of unique object ids
        self.list_of_unique_object_ids = np.asarray(self.dataframe['unique_id'].unique().tolist())

        # Get the list of all object ids
        self.list_object_ids = np.asarray(self.dataframe['unique_id'].tolist())

        # Get the list of all frame ids
        self.list_frame_ids = np.asarray(self.dataframe['frame_id'].tolist())

    def _filter_centered_samples(self, center_ratio=0.9, mode="center"):
        """
        Filter out the samples whose bounding boxes are not considered centered.

        Args:
            center_ratio (float): The ratio of the image to consider as the center.
            mode (str): The mode to use for checking if the bounding boxes are centered.

        Returns:
            None
        """

        # Compute the bounding box width and height
        bbox_width = self.dataframe['bbox_xmax'] - self.dataframe['bbox_xmin']
        bbox_height = self.dataframe['bbox_ymax'] - self.dataframe['bbox_ymin']

        # Compute the center of the bounding box
        bbox_xcenter = self.dataframe['bbox_xmin'] + bbox_width / 2
        bbox_ycenter = self.dataframe['bbox_ymin'] + bbox_height / 2

        # Retrieve the image dimensions
        img_width = self.dataframe['width']
        img_height = self.dataframe['height']

        # Convert the bounding box to YOLO format
        self.dataframe['bbox_yolo_x'] = bbox_xcenter / img_width
        self.dataframe['bbox_yolo_y'] = bbox_ycenter / img_height
        self.dataframe['bbox_yolo_w'] = bbox_width / img_width
        self.dataframe['bbox_yolo_h'] = bbox_height / img_height

        # If center_ratio is 1.0, return all the bounding boxes
        if center_ratio >= 1.0:
            return self.dataframe

        total_items = len(self.dataframe)  # Get the total number of items

        # Apply the is_bbox_centered function to the dataframe
        centered = self.dataframe.apply(lambda row: utils.is_bbox_centered(
            (row['bbox_yolo_x'], row['bbox_yolo_y'], row['bbox_yolo_w'], row['bbox_yolo_h']),
            center_ratio,
            mode), axis=1)

        # Filter the dataframe based on the centered values
        self.dataframe = self.dataframe[centered]

        remaining_items = np.sum(centered)  # Get the number of remaining items after filtering
        discarded_items = total_items - remaining_items  # Calculate the number of discarded items

        # Print the number of total items, remaining items, and discarded items
        print(f"Total items: {total_items}")
        print(f"Remaining items after filtering: {remaining_items}")
        print(f"Discarded items: {discarded_items}")

        return self.dataframe


    def get_macros_values(self, datasets):
        """
        Method to generate a mapping from the dataset names to the corresponding paths.

        Args:
            datasets (list): A list of tuples. Each tuple contains the information about one dataset and includes the dataset's name,
            path to its CSV file, the frames' folders path, data format, ground truth column, and object ID column.

        Returns:
            macro_mapping (dict): A dictionary where keys are dataset names and values are the corresponding frames' folders paths.
        """
        macro_mapping = {}
        for dataset in datasets:
            name, csv_path, frames_folders_path, format, gt_column, object_id_column = dataset
            macro_mapping[name] = frames_folders_path
        return macro_mapping

    def load_dataframe(self, csv_folder, datasets):
        """
        Method to load and concatenate all CSV files from a given directory into one DataFrame.
        It also handles the replacement of macros in the 'frame_path' column with the corresponding dataset paths.

        Args:
            csv_folder (str): The path to the directory containing the CSV files.
            datasets (list): A list of tuples. Each tuple contains the information about one dataset and
            includes the dataset's name, path to its CSV file, the frames' folders path, data format, ground truth column,
            and object ID column.

        Returns:
            df (pandas.DataFrame): A DataFrame containing all the data from the CSV files,
            with macros in the 'frame_path' column replaced with actual paths.
        """
        print('     > Create a mapping of dataset names to paths', end=' ')
        # Create a mapping of dataset names to paths
        macro_mapping = self.get_macros_values(datasets)
        print('[done]')

        print('     > Get a list of all CSV files in the specified directory', end=' ')
        # Get a list of all CSV files in the specified directory
        all_files = glob.glob(os.path.join(csv_folder, "polypsizing_*.csv"))
        print('[done]')

        print('     > Load and concatenate all CSV files into one DataFrame', end=' ')
        # Load and concatenate all CSV files into one DataFrame
        df = pd.concat((pd.read_csv(f) for f in all_files))
        df.drop_duplicates(subset=['unique_id', 'frame_id'], inplace=True)
        print('[done]')

        print("     > Replace macros in the frame_path column with the actual paths", end=' ')
        # Replace macros in the 'frame_path' column with the actual paths
        df['frame_path'] = df.apply(
            lambda row: row['frame_path'].replace(f'$SUN_DATA_PATH' if row["database"] == 'SUN' else '$REAL_COLON_PATH',
                                                  macro_mapping[row["database"]]), axis=1)
        print('[done]')

        print('     > Changing data types of DataFrame columns', end=' ')

        # Defauls columns: Specify the data types you want for each column
        dtypes = {
            'database': 'string',
            'unique_id': 'string',
            'frame_id': 'string',
            'bbox_xmin': 'int',
            'bbox_xmax': 'int',
            'bbox_ymin': 'int',
            'bbox_ymax': 'int',
            'width': 'int',
            'height': 'int',
            'gt': 'float32',
            'frame_path': 'string'
        }

        # Check for each column in your dtype dictionary
        for column in dtypes.keys():
            # If the column exists in your DataFrame, change its data type
            if column in df.columns:
                df[column] = df[column].astype(dtypes[column])
            else:
                print(f"Warning: The column '{column}' does not exist in the DataFrame and will be skipped.")
        print('[done]')

        return df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset at the given index 'idx'.

        This method reads the correct row from the DataFrame, uses the frame path to read the RGB image using PIL,
        retrieves the ground truth from the 'gt' column, retrieves the bounding box coordinates and image size,
        converts the bounding box coordinates to YOLO format, and creates a binary location mask using the bounding box.

        Args:
        idx (int): Index of the item to be retrieved.

        Returns:
        tuple: A tuple containing three elements:
            - image (PIL.Image.Image): The image at the specified path.
            - binary_mask (numpy.ndarray): The binary location mask representing the bounding box. The mask has the same size as the image, with ones inside the bounding box and zeros elsewhere.
            - gt (float): The ground truth value.
        """
        # Retrieve the correct row from DataFrame using the index
        row = self.dataframe.iloc[idx]

        # Read the image using PIL
        image_path = row['frame_path']
        image = Image.open(image_path).convert('RGB')

        # Retrieve the ground truth
        gt = row['gt']

        # Retrieve the bounding box coordinates
        bbox_xmin, bbox_ymin = row['bbox_xmin'], row['bbox_ymin']
        bbox_xmax, bbox_ymax = row['bbox_xmax'], row['bbox_ymax']

        # Retrieve the image dimensions
        img_width, img_height = row['width'], row['height']

        # Compute the bounding box width and height
        bbox_width = bbox_xmax - bbox_xmin
        bbox_height = bbox_ymax - bbox_ymin

        # Compute the center of the bounding box
        bbox_xcenter = bbox_xmin + bbox_width / 2
        bbox_ycenter = bbox_ymin + bbox_height / 2

        # Convert the bounding box to YOLO format
        bbox_yolo_format = (
            bbox_xcenter / img_width, bbox_ycenter / img_height, bbox_width / img_width, bbox_height / img_height)

        # Create a binary mask for the bounding box
        binary_mask = np.zeros((img_height, img_width))
        binary_mask[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax] = 1

        # Convert to PyTorch tensors
        image = ToTensor()(image)  # normalize to [0,1]
        binary_mask = ToTensor()(binary_mask)  # normalize to [0,1]

        if self.transform:
            image, binary_mask = self.transform(image, binary_mask)

        return image, binary_mask, gt, idx


##############################################
# COSMO IMD: K-Fold Splitter Class
##############################################
class IMDKfoldSplitter:
    def __init__(self, dataset_folder, num_splits=5):
        """
        Initialize the splitter.

        Args:
            dataset_folder (str): The path to the folder containing the preprocessed dataset.
            num_splits (int): The number of splits to create. Default is 5.
        """
        self.dataset_folder = dataset_folder
        self.num_splits = num_splits
        self.kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=1)

    def create_splits(self, mode='per_object'):
        """
        Create the k-fold splits and save the result back to a CSV file.

        No arguments. No return values.
        """

        # Concatenate all parts of the preprocessed dataset into a single DataFrame
        print(f'Dataset path: {self.dataset_folder}')
        df_parts = []
        for filename in os.listdir(self.dataset_folder):
            if filename.startswith('polypsizing_') and filename.endswith('.csv'):
                df_part = pd.read_csv(os.path.join(self.dataset_folder, filename))
                df_parts.append(df_part)
        df = pd.concat(df_parts, ignore_index=True)
        df.drop_duplicates(subset=['unique_id', 'frame_id'], inplace=True)

        print(f' -> {len(df_parts)} data parts read from: {self.dataset_folder}')
        print(f'Dataframe read: {df.shape}, {df.columns}')

        # Create a new column 'gt_group' to categorize 'gt' values into 'D', 'S', 'L' groups
        df['gt_group'] = df['gt'].apply(map_to_class_gt)

        if mode == 'per_frame':
            # Create a new column 'kfold_id' with the ID of the split each sample belongs to
            df['kfold_id'] = -1  # initially set all kfold_id to -1
            for fold, (_, test_index) in enumerate(self.kf.split(df, df['gt_group'])):
                df.loc[test_index, 'kfold_id'] = fold

            # Export the DataFrame with 'unique_id', 'gt', and 'kfold_id' to a new CSV file
            df.to_csv(os.path.join(self.dataset_folder, 'kfold_polypsizing_lists_per_frame.csv'), index=False)

        elif mode == 'per_object':
            # Group by 'unique_id' and take the first sample from each group for stratification
            representative_samples = df.groupby('unique_id').first()

            # Create a new column 'kfold_id' with the ID of the split each sample belongs to
            df['kfold_id'] = -1  # initially set all kfold_id to -1

            # Use the kf.split on the representative samples with their 'gt_group'
            for fold, (_, test_index) in enumerate(
                    self.kf.split(representative_samples, representative_samples['gt_group'])):
                # Get the unique_ids of the chosen representative samples from the index
                test_unique_ids = representative_samples.index[test_index].values

                # Set the 'kfold_id' for all rows with the chosen unique_ids
                df.loc[df['unique_id'].isin(test_unique_ids), 'kfold_id'] = fold

            # Export the DataFrame with 'unique_id', 'gt', and 'kfold_id' to a new CSV file
            df.to_csv(os.path.join(self.dataset_folder, 'kfold_polypsizing_lists_per_object.csv'), index=False)


class IMDDataStats:
    def __init__(self, dataset_folder, mode='per_object'):
        """
        Initialize the stats generator.

        Args:
            dataset_folder (str): The path to the folder containing the dataset.
        """
        self.dataset_folder = dataset_folder
        self.mode = mode
        if mode=='per_frame':
            file_name = 'kfold_polypsizing_lists_per_frame.csv'
        else:
            file_name = 'kfold_polypsizing_lists_per_object.csv'

        self.df = pd.read_csv(os.path.join(dataset_folder, file_name))

    def plot_distribution(self):
        """
        Plot the distribution of 'gt' and the distribution of samples across each fold.

        No arguments. No return values.
        """
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        sns.countplot(x='gt_group', data=self.df, palette="Set2")
        plt.title('Distribution of GT Classes')
        plt.xlabel('GT Group')
        plt.ylabel('Number of Samples')
        plt.savefig(os.path.join(self.dataset_folder, f'{self.mode}_gt_distribution.png'))

        plt.subplot(122)
        sns.countplot(x='kfold_id', data=self.df, palette="Set2")
        plt.title('Distribution of K-Fold Classes')
        plt.xlabel('K-Fold ID')
        plt.ylabel('Number of Samples')

        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset_folder, f'{self.mode}_kfold_distribution.png'))

        # Plot the distribution of GT groups within each fold
        plt.figure(figsize=(15, 10))
        for fold in range(self.df['kfold_id'].nunique()):
            plt.subplot(2, 3, fold + 1)
            sns.countplot(x='gt_group', data=self.df[self.df['kfold_id'] == fold], order=['D', 'S', 'L'],
                          palette="Set2")
            plt.title(f'Distribution of GT Classes in Fold {fold}')
            plt.xlabel('GT Group')
            plt.ylabel('Number of Samples')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset_folder, f'{self.mode}_gt_group_distribution_per_fold.png'))

        # If 'unique_id' has less than 4 characters, assign 'sun', else assign the first three characters
        self.df['dataset'] = self.df['unique_id'].apply(lambda x: 'sun' if len(x) < 4 else x[:3])

        # List all unique datasets
        datasets = self.df['dataset'].unique()

        print(datasets)

        # Determine the number of rows needed for the subplots
        num_rows = (len(datasets) + 1) // 2

        # Create a new figure
        fig, axs = plt.subplots(num_rows, 2, figsize=(20, 10 * num_rows), sharey=True)
        axs = axs.ravel()  # Flatten the array of axes to loop over them

        # For each unique dataset
        for i, dataset in enumerate(datasets):
            # Filter the data for the current dataset
            df_dataset = self.df[self.df['dataset'] == dataset]

            # Plot the distribution of GT groups for the current dataset
            ax = sns.countplot(x='gt_group', data=df_dataset, order=['D', 'S', 'L'], palette="Set2", ax=axs[i])

            # Get the current y-axis values for positioning the labels
            y_values = ax.get_yticks()

            # Set the title with the name of the dataset and the total number of samples
            axs[i].set_title(f'Dataset: {dataset}\nTotal Samples: {len(df_dataset)}', fontweight='bold')

            # Add the number of samples at the top of each bin
            for p in ax.patches:
                height = p.get_height()
                # Skip NaN values
                if np.isnan(height):
                    continue

                ax.text(p.get_x() + p.get_width() / 2., height,
                        f'{int(height)}',
                        fontsize=12, color='black', ha='center', va='bottom')

            axs[i].set_xlabel('GT Group')
            axs[i].set_ylabel('Number of Samples')

        # Delete any unused subplots
        for j in range(i + 1, num_rows * 2):
            fig.delaxes(axs[j])

        # Save the figure
        fig.tight_layout()
        fig.savefig(os.path.join(self.dataset_folder, f'{self.mode}_gt_group_distribution_per_datasets.png'))

    def plot_distribution_per_unique_id(self):
        """
        Plot the distribution of GT classes per unique_id across each fold.
        """
        # Counting unique IDs for each GT class in each fold
        df_grouped = self.df.groupby(['kfold_id', 'gt_group'])['unique_id'].nunique().reset_index()

        # Plotting
        plt.figure(figsize=(15, 10))
        for fold in sorted(df_grouped['kfold_id'].unique()):
            plt.subplot(2, 3, fold + 1)
            sns.barplot(x='gt_group', y='unique_id', data=df_grouped[df_grouped['kfold_id'] == fold], palette="Set2")
            plt.title(f'Distribution of GT Classes per Unique ID in Fold {fold}')
            plt.xlabel('GT Group')
            plt.ylabel('Number of Unique IDs')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset_folder, f'{self.mode}_gt_group_distribution_per_unique_id_per_fold.png'))

    def show_stats_per_unique_id(self):
        """
        Print statistics for unique_id, such as the number of unique IDs in each fold and their GT distribution.
        """
        # Counting unique IDs in each fold
        fold_unique_id_counts = self.df.groupby('kfold_id')['unique_id'].nunique()
        print('\nNumber of Unique IDs in Each Fold:')
        for fold, count in fold_unique_id_counts.items():
            print(f'Fold {fold}: {count}')

        # GT distribution for unique IDs in each fold
        print('\nGT Distribution for Unique IDs in Each Fold:')
        for fold in range(self.df['kfold_id'].nunique()):
            fold_data = self.df[self.df['kfold_id'] == fold]
            unique_id_gt_distribution = fold_data.groupby('unique_id')['gt_group'].first().value_counts()
            print(f'Fold {fold}:')
            for group, count in unique_id_gt_distribution.items():
                print(f'    {group}: {count}')

    def show_stats(self):
        """
        Compute and print statistics, such as the number of samples in each 'gt' group and the total number of samples.

        No arguments. No return values.
        """
        # Print the number of samples in each 'gt' group
        gt_group_counts = self.df['gt_group'].value_counts()
        print('\nNumber of Samples in Each GT Group:')
        for group, count in gt_group_counts.items():
            print(f'{group}: {count}')

        # Print the total number of samples
        print(f'\nTotal Number of Samples: {len(self.df)}')

        # Print the number of samples in each fold
        fold_counts = self.df['kfold_id'].value_counts().sort_index()
        print('\nNumber of Samples in Each Fold:')
        for fold, count in fold_counts.items():
            print(f'Fold {fold}: {count}')

        # Print the number of unique IDs
        print(f'\nTotal Number of Unique IDs: {self.df["unique_id"].nunique()}')

        # Print the distribution of GT groups within each fold
        print('\nDistribution of GT Groups Within Each Fold:')
        for fold in range(self.df['kfold_id'].nunique()):
            fold_group_counts = self.df[self.df['kfold_id'] == fold]['gt_group'].value_counts()
            print(f'Fold {fold}:')
            for group, count in fold_group_counts.items():
                print(f'    {group}: {count}')


######################################
# REAL COLON DATABASE PREPROCESSOR
######################################
class IMDRealColonDatasetPreprocessor:
    def __init__(self, base_dataset_path: str, dataset_tuple: tuple, output_folder: str, saving_step: int):
        """
        Initialize the preprocessor.

        Args:
            base_dataset_path (str): The path to the base dataset.
            dataset_tuple (tuple): A tuple containing the dataset name, the path to its CSV file, and the name of the GT column.
            output_folder (str): The path where the processed dataset will be saved.
            saving_step (int): If the dataframe has reached saving_step rows, save it to the output folder.
        """
        self.base_dataset_path = base_dataset_path
        self.output_folder = output_folder
        self.dataset_tuple = dataset_tuple
        self.saving_step = saving_step
        self.columns = ['database', 'unique_id', 'frame_id', 'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax',
                        'width', 'height', 'frame_path', 'gt']
        self.df = pd.DataFrame(columns=self.columns)
        self.part = 1

    def save_df(self):
        """
        Save the dataframe to the output folder and create a new empty dataframe.

        No arguments. No return values.
        """
        filename = os.path.join(self.output_folder, f'polypsizing_real_part{self.part}.csv')
        print(f'<!> Exporting file: {filename}.')
        self.df.to_csv(filename, index=False)
        self.df = pd.DataFrame(columns=self.columns)
        self.part += 1

    def process_xml_file(self, filename: str, annotations_folder: str, csv_df: pd.DataFrame,
                         frames_folders_path: str, format: str) -> list:
        """
        Process an individual XML file and return the extracted data in a list of dictionary format.

        Args:
            filename (str): The filename of the XML file.
            annotations_folder (str): The folder where the XML file is located.
            frames_folder_path (str): The base path to the folder containing the frames.
            frames_folder_format (str): The format of the frames' folder name.

        Returns:
            data (list): A list of dictionaries containing the parsed XML data.

        """
        xml_path = os.path.join(annotations_folder, filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        objects = root.findall('object')
        data = []
        if objects:
            frame_id = filename.split('_')[-1].replace('.xml', '')
            database = root.find('source/database').text
            for obj in objects:
                unique_id = obj.find('unique_id').text
                bbox = obj.find('bndbox')
                bbox_xmin = int(bbox.find('xmin').text)
                bbox_xmax = int(bbox.find('xmax').text)
                bbox_ymin = int(bbox.find('ymin').text)
                bbox_ymax = int(bbox.find('ymax').text)

                # Compute frame path
                video_id = self.get_video_id(unique_id)
                object_dir_path = os.path.join("$REAL_COLON_PATH", format % video_id)
                frame_path = os.path.join(object_dir_path, video_id + '_' + frame_id + '.jpg')

                data.append({
                    'database': self.dataset_tuple[0],
                    'unique_id': unique_id,
                    'frame_id': frame_id,
                    'bbox_xmin': bbox_xmin,
                    'bbox_xmax': bbox_xmax,
                    'bbox_ymin': bbox_ymin,
                    'bbox_ymax': bbox_ymax,
                    'width': width,
                    'height': height,
                    'frame_path': frame_path,
                    'gt': csv_df.loc[csv_df['object_id'] == unique_id, 'gt'].values[0] if unique_id in csv_df[
                        'object_id'].values else None
                })
        return data

    def get_video_id(self, object_id) -> str:
        """
        Extract the video_id from the object_id string by removing the details after underscore ('_').

        Args:
            object_id: The object_id which can be a string in the format 'XXX-XXX_X' or other types of input.

        Returns:
            video_id (str): The video_id string which is in the format 'XXX-XXX', or the original input converted to string if it doesn't contain an underscore ('_') or isn't a string.
        """
        object_id = str(object_id)  # Convert the input to string in case it isn't a string
        if "_" in object_id:
            return object_id.split('_')[0]
        else:
            return str(object_id)

    @staticmethod
    def load_csv_dataset(dataset):
        """
        Load a CSV dataset into a pandas DataFrame.

        Args:
            dataset (tuple): A tuple containing the dataset name, the path to its CSV file, and the name of the GT column.

        Returns:
            DataFrame: The loaded dataset.
        """
        name, csv_path, frames_folders_path, format, gt_column, object_id_column = dataset

        df = pd.read_csv(csv_path)

        df["database"] = name
        df = df[['database', object_id_column, gt_column]]
        df.rename(columns={object_id_column: "object_id", gt_column: "gt"}, inplace=True)

        return df, frames_folders_path, format

    def process_dataset(self):
        """
        Process the raw dataset by traversing all the directories and XML annotations.

        No arguments. No return values.
        """
        # Load the CSV dataset and prepare it for use
        csv_df, frames_folders_path, format = self.load_csv_dataset(self.dataset_tuple)

        # Iterate over the directories in the base dataset path
        for dir_name in os.listdir(self.base_dataset_path):
            print(f'> IMDRealColonDatasetPreprocessor: Working on {dir_name}...')
            # If the directory is a video frames directory, skip it
            if "_frames" in dir_name:
                print(f'{dir_name} contains .jpg files.')
                continue

            # If the directory is an annotations directory
            if "_annotations" in dir_name:
                annotations_folder = os.path.join(self.base_dataset_path, dir_name)
                frames_folder_path = annotations_folder.replace('_annotations', '_frames')
                frames_folder_format = os.path.basename(frames_folder_path)

                # Iterate over the XML annotations in the annotations folder
                for filename in os.listdir(annotations_folder):
                    # Extract the data from the XML file
                    new_data = self.process_xml_file(filename, annotations_folder, csv_df, frames_folders_path, format)

                    # Append new data to the DataFrame
                    self.df = pd.concat([self.df, pd.DataFrame(new_data, columns=self.columns)], ignore_index=True)
                    self.df.drop_duplicates(subset=['unique_id', 'frame_id'], inplace=True)

                    # If the dataframe has reached self.saving_step rows, save it to the output folder
                    if self.df.shape[0] >= self.saving_step:
                        self.save_df()
                        print(f'> Creating new file: Part {self.part}...', end=' ')

        # Save the remaining dataframe if it has less than self.saving_step rows but not empty
        if not self.df.empty:
            self.save_df()


###############################
# SUN DATABASE PREPROCESSOR
###############################
class SUNDatasetPreprocessor:
    def __init__(self, base_dataset_path: str, dataset_tuple: tuple, output_folder: str, saving_step: int):
        """
        Initialize the SUNDatasetPreprocessor.

        Args:
            base_dataset_path (str): The path to the base dataset.
            dataset_tuple (tuple): A tuple containing the dataset name, the path to its CSV file, and the name of the GT column.
            output_folder (str): The path where the processed dataset will be saved.
            saving_step (int): If the dataframe has reached saving_step rows, save it to the output folder.
        """
        self.base_dataset_path = base_dataset_path  # frames -> /renamed/caseM/*.jpg, & /annotation_txt/caseM.txt
        self.output_folder = output_folder
        self.dataset_tuple = dataset_tuple
        self.saving_step = saving_step
        self.columns = ['database', 'unique_id', 'frame_id', 'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax',
                        'width', 'height', 'frame_path', 'gt']
        self.df = pd.DataFrame(columns=self.columns)
        self.part = 1

    def save_df(self):
        """
        SUNDatasetPreprocessor
        Save the dataframe to the output folder and create a new empty dataframe.

        No arguments. No return values.
        """
        filename = os.path.join(self.output_folder, f'polypsizing_sun_part{self.part}.csv')
        print(f'<!> Exporting file: {filename}.')
        self.df.to_csv(filename, index=False)
        self.df = pd.DataFrame(columns=self.columns)
        self.part += 1

    def extract_info(self, line):
        """
        Extract bounding box and frame name from a line of text.

        Args:
            line (str): A line of text containing frame name and bounding box.

        Returns:
            bbox (tuple): Bounding box coordinates as integers.
            frame_name (str): Frame name.
            frame_id (str): frame id e.g., 001
        """
        components = line.split()
        filename = components[0]
        bbox_values = components[1].split(',')

        # Extract bbox
        bbox = tuple(map(int, bbox_values[:4]))

        # Extract and format frame name
        raw_frame_name = re.search(r'image\d+.jpg', filename).group()
        frame_seq_num = re.search(r'\d+', raw_frame_name).group()
        frame_number = str(int(frame_seq_num))
        frame_name = f'image_{frame_number.zfill(3)}.jpg'
        frame_id = frame_number.zfill(3)
        return bbox, frame_name, frame_id

    def process_txt_file(self, filename: str, annotations_folder: str, csv_df: pd.DataFrame, format: str) -> list:
        """
        Process an individual XML file and return the extracted data in a list of dictionary format.

        Args:
            filename (str): The filename of the XML file.
            annotations_folder (str): The folder where the XML file is located.
            csv_df (pd.DataFrame): DataFrame containing information of 'object_id' and 'gt'.
            frames_folders_path (str): The base path to the folder containing the frames.
            format (str): The format of the frames' folder name.

        Returns:
            data (list): A list of dictionaries containing the parsed XML data.
        """
        txt_anno_path = os.path.join(annotations_folder, filename)

        data = []

        with open(txt_anno_path, 'r') as f:
            lines = f.readlines()

        print(f'> SUN {filename}: number of frames {len(lines)}...')
        for line in lines:
            bbox, frame_name, frame_id = self.extract_info(line)

            # Get the unique lesion ID
            raw_frame_name = re.search(r'case\d+.txt', filename).group()
            unique_id = int(re.search(r'\d+', raw_frame_name).group())

            frame_path = os.path.join("$SUN_DATA_PATH", "renamed", format % unique_id, frame_name)

            # Get the GT size in mm
            gt = csv_df.loc[csv_df['object_id'] == unique_id, 'gt'].values[0]

            # Read 1 single frame to get width & height
            width, height = Image.open(
                os.path.join(annotations_folder.split('annotation_txt')[0], 'renamed', format % unique_id,
                             frame_name)).size

            row = {
                'database': self.dataset_tuple[0],
                'unique_id': unique_id,
                'frame_id': frame_id,
                'bbox_xmin': bbox[0],
                'bbox_xmax': bbox[2],
                'bbox_ymin': bbox[1],
                'bbox_ymax': bbox[3],
                'width': width,
                'height': height,
                'frame_path': frame_path,
                'gt': gt
            }

            data.append(row)

        return data

    def get_video_id(self, object_id) -> str:
        """
        SUNDatasetPreprocessor
        Extract the video_id from the object_id string by removing the details after underscore ('_').

        Args:
            object_id: The object_id which can be a string in the format 'XXX-XXX_X' or other types of input.

        Returns:
            video_id (str): The video_id string which is in the format 'XXX-XXX', or the original input converted to string if it doesn't contain an underscore ('_') or isn't a string.
        """
        object_id = str(object_id)  # Convert the input to string in case it isn't a string
        if "_" in object_id:
            return object_id.split('_')[0]
        else:
            return str(object_id)

    @staticmethod
    def load_csv_dataset(dataset):
        """
        SUNDatasetPreprocessor
        Load a CSV dataset into a pandas DataFrame.

        Args:
            dataset (tuple): A tuple containing the dataset name, the path to its CSV file, and the name of the GT column.

        Returns:
            DataFrame: The loaded dataset.
        """
        name, csv_path, frames_folders_path, format, gt_column, object_id_column = dataset

        df = pd.read_csv(csv_path)

        df["database"] = name
        df = df[['database', object_id_column, gt_column]]
        df.rename(columns={object_id_column: "object_id", gt_column: "gt"}, inplace=True)

        return df, frames_folders_path, format

    def convert_mm_to_float(self, df):
        """
        Convert the gt column from string to float.
        Expected string format is 'Xmm', where X can be an integer or float.

        Args:
            df (pd.DataFrame): DataFrame with a 'gt' column containing strings in 'Xmm' format.

        Returns:
            df (pd.DataFrame): DataFrame with the 'gt' column converted to float.
        """
        df['gt'] = df['gt'].str.replace('mm', '').str.replace('-', '').astype(float)

        print(
            f"> SUN Database loaded:\n  Minimum size: {df['gt'].min()} mm, maximum size {df['gt'].max()} mm.\n  Average size {df['gt'].mean()} mm +/- {df['gt'].std()}, median size {df['gt'].median()}")

        return df

    def process_dataset(self):
        """
        SUNDatasetPreprocessor
        Process the raw dataset by traversing all the directories and XML annotations.

        No arguments. No return values.
        """
        # Load the CSV dataset and prepare it for use
        csv_df, frames_folders_path, format = self.load_csv_dataset(self.dataset_tuple)
        csv_df = self.convert_mm_to_float(csv_df)
        # Iterate over the directories in the base dataset path
        for dir_name in os.listdir(self.base_dataset_path):
            print(f'> SUNDatasetPreprocessor: Working on {dir_name}...')
            # If the directory is an annotations directory
            if "annotation_txt" in dir_name:
                annotations_folder = os.path.join(self.base_dataset_path, dir_name)

                # Iterate over the XML annotations in the annotations folder
                for filename in os.listdir(annotations_folder):
                    # Extract the data from the TXT file
                    new_data = self.process_txt_file(filename, annotations_folder, csv_df, format)

                    # Append new data to the DataFrame
                    self.df = pd.concat([self.df, pd.DataFrame(new_data, columns=self.columns)], ignore_index=True)
                    self.df.drop_duplicates(subset=['unique_id', 'frame_id'], inplace=True)

                    # If the dataframe has reached self.saving_step rows, save it to the output folder
                    if self.df.shape[0] >= self.saving_step:
                        self.save_df()
                        print(f'> Creating new file: Part {self.part}...', end=' ')

        # Save the remaining dataframe if it has less than self.saving_step rows but not empty
        if not self.df.empty:
            self.save_df()

