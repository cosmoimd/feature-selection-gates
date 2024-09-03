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
import os

import matplotlib
# Import plot modules
import matplotlib.pyplot as plt
# Import 3rd party modules
import numpy as np

# Import custom modules
from utils import is_bbox_centered

# Import PyTorch modules

# File information
__author__ = "Giorgio Roffo"
__copyright__ = "Copyright 2023, COSMO IMD"
__credits__ = ["Giorgio Roffo", "add others"]
__license__ = "Private"
__version__ = "1.0.0"
__maintainer__ = "Giorgio Roffo"
__email__ = "groffo@cosmoimd.com"
__status__ = "Development"  # can be "Prototype", "Development" or "Production"



matplotlib.use('Agg')

def scatter_plot(x, y, title='Scatter Plot', xlabel='x', ylabel='y', save_path=None, color='blue', grid=True, s=30):
    """
    This function creates and saves a scatter plot.

    Args:
        x (array-like): The data to be plotted on the x-axis.
        y (array-like): The data to be plotted on the y-axis.
        title (str, optional): The title of the plot. Defaults to 'Scatter Plot'.
        xlabel (str, optional): The label for the x-axis. Defaults to 'x'.
        ylabel (str, optional): The label for the y-axis. Defaults to 'y'.
        save_path (str, optional): The path where the plot will be saved. If None, the plot will not be saved. Defaults to None.
        color (str, optional): The color of the points in the plot. Defaults to 'blue'.
        grid (bool, optional): Whether to show a grid in the plot. Defaults to True.
        s (float, optional): The size of the points. Defaults to 30.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, color=color, s=s)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(grid)
    plt.xlim([-1, 19])
    plt.ylim([0, 20])

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

def histogram(data, bins=10, labels=None, title='Histogram', xlabel='Values', ylabel='Frequency', save_path=None, colors=None, grid=True):
    """
    This function creates and saves a histogram for multiple data sets.

    Args:
        data (list of array-like): The data sets to create a histogram of.
        bins (int, optional): The number of bins to use in the histogram. Defaults to 10.
        labels (list of str, optional): Labels for the data sets. Defaults to None.
        title (str, optional): The title of the histogram. Defaults to 'Histogram'.
        xlabel (str, optional): The label for the x-axis. Defaults to 'Values'.
        ylabel (str, optional): The label for the y-axis. Defaults to 'Frequency'.
        save_path (str, optional): The path where the histogram will be saved. If None, the histogram will not be saved. Defaults to None.
        colors (list of str, optional): The colors of the histograms for each data set. Defaults to None.
        grid (bool, optional): Whether to show a grid in the histogram. Defaults to True.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    for i, data_set in enumerate(data):
        plt.hist(data_set, bins=np.arange(0.5, 21.5, 1), alpha=0.5, label=labels[i] if labels else None,
                 color=colors[i] if colors else None)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(grid)

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def time_series(data, labels=None, title='Time Series Plot', xlabel='Time', ylabel='Value', save_path=None, colors=None, grid=True):
    """
    This function creates and saves a time series plot for multiple data sets.

    Args:
        data (list of array-like): The data sets to create a time series plot of.
        labels (list of str, optional): Labels for the data sets. Defaults to None.
        title (str, optional): The title of the plot. Defaults to 'Time Series Plot'.
        xlabel (str, optional): The label for the x-axis. Defaults to 'Time'.
        ylabel (str, optional): The label for the y-axis. Defaults to 'Value'.
        save_path (str, optional): The path where the plot will be saved. If None, the plot will not be saved. Defaults to None.
        colors (list of str, optional): The colors of the plot for each data set. Defaults to None.
        grid (bool, optional): Whether to show a grid in the plot. Defaults to True.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    for i, data_set in enumerate(data):
        plt.plot(data_set, label=labels[i] if labels else None, color=colors[i] if colors else None)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(grid)
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def create_boxplots_per_gt(preds, gt_values, validation_metrics, exp_tag, epoch, output_folder):
    """
    Create and save a separate boxplot for each unique ground truth (gt) value.

    Args:
        preds (list): Predictions associated with each ground truth value.
        gt_values (list): Ground truth values.
        validation_metrics (dict): Dictionary containing validation metrics.
        exp_tag (str): Experiment tag.
        epoch (int): Current epoch.
        output_folder (str): Output folder path to save the plots.
    """
    # Flatten the lists if necessary
    preds = np.array(preds).squeeze()
    gt_values = np.array(gt_values).squeeze()

    # Get unique GT values
    unique_gt_values = sorted(set(gt_values))

    # Loop through each unique GT value
    for gt in unique_gt_values:
        # Gather predictions for current GT value
        current_preds = [pred for pred, gt_val in zip(preds, gt_values) if gt_val == gt]

        # Create boxplot
        plt.figure(figsize=(12, 8))
        plt.boxplot(current_preds, labels=[str(gt)])
        plt.xlabel('GT')
        plt.ylabel('Predictions')

        # Compute some metrics for the title
        float_accuracy = np.around(100 * validation_metrics["float_accuracy"], 1)
        dsl_accuracy = np.around(100 * validation_metrics["dsl_accuracy"], 1)
        balanced_accuracy_dsl = np.around(100 * validation_metrics["balanced_accuracy_dsl"], 1)

        # Add details to the title
        title = (f'{exp_tag} - Box Plot for GT={gt}\n'
                 f'Per-Frame: Float. Acc. {float_accuracy}%, '
                 f'DSL Acc. {dsl_accuracy}%, '
                 f'DSL B.Acc. {balanced_accuracy_dsl}%')
        plt.title(title)
        plt.xticks(rotation=90)
        plt.xlim([-1, 19])
        plt.ylim([0, 20])

        # Save figure
        plot_filename = f'Ep{epoch}_{float_accuracy}_boxplot_gt_{gt}.jpg'
        plt.savefig(os.path.join(output_folder, plot_filename))
        plt.close()

        print(f'Boxplot for GT={gt} saved as {plot_filename} in {output_folder}')


def create_single_boxplot_per_gt(preds, gt_values, validation_metrics, exp_tag, epoch, kfold, output_folder, phase):
    """
    Create and save a single boxplot for all unique ground truth (gt) values.

    Args:
        preds (list): Predictions associated with each ground truth value.
        gt_values (list): Ground truth values.
        validation_metrics (dict): Dictionary containing validation metrics.
        exp_tag (str): Experiment tag.
        epoch (int): Current epoch.
        output_folder (str): Output folder path to save the plots.
        phase (str): train, valid, test.
    """
    # Flatten the lists if necessary
    preds = np.array(preds).squeeze()
    gt_values = np.array(gt_values).squeeze()

    # Get unique GT values
    unique_gt_values = sorted(set(gt_values))

    all_preds = []
    labels = []

    # Loop through each unique GT value
    for gt in unique_gt_values:
        # Gather predictions for current GT value
        current_preds = [pred for pred, gt_val in zip(preds, gt_values) if gt_val == gt]

        all_preds.append(current_preds)
        labels.append(str(gt))

    # Create boxplot
    plt.figure(figsize=(12, 8))
    plt.boxplot(all_preds, labels=labels)
    plt.xlabel('GT')
    plt.ylabel('Predictions')

    # Compute some metrics for the title
    float_accuracy = np.around(100 * validation_metrics["float_accuracy"], 1)
    dsl_accuracy = np.around(100 * validation_metrics["dsl_accuracy"], 1)
    balanced_accuracy_dsl = np.around(100 * validation_metrics["balanced_accuracy_dsl"], 1)

    # Add details to the title
    title = (f'{exp_tag} - Box Plots for All GT\n'
             f'Per-Frame: Float. Acc. {float_accuracy}%, '
             f'DSL Acc. {dsl_accuracy}%, '
             f'DSL B.Acc. {balanced_accuracy_dsl}%')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.xlim([-1, 19])
    plt.ylim([0, 20])

    # Save figure
    plot_filename = f'{phase.upper()}_Ep{epoch}_Kfold{kfold}_{balanced_accuracy_dsl}_boxplot_all_gt.jpg'
    plt.savefig(os.path.join(output_folder, plot_filename))
    plt.close()

    print(f'Boxplot for all GT saved as {plot_filename} in {output_folder}')

def _test_is_bbox_centered(n_bboxes=25, bbox_size=0.2, center_ratio=0.7, mode="center", im_size = 512):
    """
        Generate and plot random bounding boxes, color-coding them based on whether they are considered centered.

    Args:
        n_bboxes (int): The number of random bounding boxes to generate.
        bbox_size (float): The maximum size of the bounding boxes, relative to the image size.
        center_ratio (float): The ratio of the image to consider as the center.
        mode (str): The mode to use for checking if the bounding boxes are centered.
        im_size (int): width and height of the image, e.g. 512.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Create a new figure
    fig, ax = plt.subplots(1)

    # Display an image
    img = np.ones((im_size, im_size, 3))
    ax.imshow(img)

    # Add a circle representing the center region
    center_circle = patches.Circle((im_size//2, im_size//2), center_ratio * im_size//2, edgecolor='g', facecolor='none')
    ax.add_patch(center_circle)

    # Add random bounding boxes
    for _ in range(n_bboxes):
        # Generate a random bounding box
        bbox = np.random.rand(4)
        bbox[2:] *= bbox_size  # Adjust the size of the bounding box

        # Check if the bounding box is centered
        if is_bbox_centered(bbox, center_ratio, mode):
            edgecolor = 'g'  # Color the bounding box green if it is centered
        else:
            edgecolor = 'r'  # Color the bounding box red if it is not centered

        # Create a rectangle patch
        # Adjust the position of the bounding box so that its center is at (x_center, y_center)
        rect = patches.Rectangle((bbox[0] * im_size - bbox[2] * im_size / 2, bbox[1] * im_size - bbox[3] * im_size / 2),
                                 bbox[2] * im_size, bbox[3] * im_size, linewidth=1, edgecolor=edgecolor, facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Plot the center of the bounding box
        plt.text(bbox[0] * im_size, bbox[1] * im_size, f'({bbox[0]:.2f}, {bbox[1]:.2f})',
                 horizontalalignment='center', verticalalignment='center', color='blue')

    plt.show()
