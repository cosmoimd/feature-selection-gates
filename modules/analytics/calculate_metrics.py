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

# Import PyTorch modules

import os

# Import plot modules
import numpy as np
# Import 3rd party modules
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, mean_absolute_error, mean_squared_error)
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

# Import custom modules

# File information
__author__ = "Giorgio Roffo, Dr."
__copyright__ = "Copyright 2024, COSMO IMD"
__credits__ = ["Giorgio Roffo"]
__license__ = "Private"
__version__ = "1.0.1"
__maintainer__ = "groffo"
__email__ = "groffo@cosmoimd.com,giorgio.roffo@gmail.com"
__status__ = "Production"  # can be "Prototype", "Development" or "Production"


def training_metrics_logger(y_true, y_pred, preds, targets):
    """
    Computes a variety of metrics to evaluate the performance of a model.

    Args:
    - y_true (list or array-like): True target values.
    - y_pred (list or array-like): Predicted target values by the model.

    Returns:
    - dict: A dictionary containing values of various metrics.
    """

    # Initialize metrics dictionary with default values
    metrics = {
        'accuracy': -1,
        'f1': -1,
        'precision': -1,
        'recall': -1,
        'MAE': -1,
        'PE': -1,
        'RMSE': -1,
        'MSE': -1
    }

    try:
        # Old Metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=1)
        metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=1)
        metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=1)

        # Check for invalid values in preds and targets
        if np.any(np.isnan(preds)) or np.any(np.isnan(targets)) or \
           np.any(np.isinf(preds)) or np.any(np.isinf(targets)):
            raise ValueError("Input contains NaN or infinity.")

        # New Metrics
        metrics['MAE'] = mean_absolute_error(targets, preds)
        metrics['MSE'] = mean_squared_error(targets, preds)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])

        # Calculating Percentage Error
        # Avoid division by zero
        non_zero_targets = targets != 0
        if np.any(non_zero_targets):
            metrics['PE'] = np.mean(((targets - preds) / targets)[non_zero_targets]) * 100
        else:
            metrics['PE'] = 0
    except Exception as e:
        print(f"An error occurred during metrics computation: {e}")

    return metrics


def bias_variance_analysis(df, metric="balanced_performance"):
    """
    Perform bias-variance analysis on the given dataframe based on a specified metric.

    Args:
    - df (pd.DataFrame): DataFrame containing the columns 'gt', 'pred', and 'kfold_id'.
    - metric (str): The metric to be computed.
    One of ["sensitivity", "specificity", "balanced_performance", "precision", "recall", "f1", "balanced_acc", "accuracy"]

    Returns:
    - tuple: A tuple containing (bias, variance) for the data based on the specified metric.
    """
    # Validate input DataFrame columns
    expected_columns = {'object_id', 'frame_id', 'gt', 'pred', 'gt_class', 'pred_class', 'gt_bin_lass', 'pred_bin_class', 'kfold_id'}
    assert set(df.columns) & expected_columns == expected_columns, "DataFrame missing required columns."

    metrics_names = ['sensitivity', 'specificity', 'balanced_perf', 'precision', 'recall', 'f1', 'balanced_accuracy', 'accuracy']
    metric_index = 2 # Default is balanced_performance

    if metric in metrics_names:
        metric_index = np.argwhere(np.array(metrics_names) == metric).flatten().item()


    # Compute the metric for each fold - DSL Classes
    data_triclass = df.groupby('kfold_id').apply(lambda x: compute_metrics_testing(x['gt_class'], x['pred_class'])).values

    # Computing mean and variance for each metric
    data_triclass = np.stack(data_triclass)
    means = np.mean(data_triclass, axis=0)
    variances = np.var(data_triclass, axis=0)

    bias_triclass = means[metric_index]
    variance_triclass = variances[metric_index]

    # Compute the metric for each fold - DSL Classes
    data_binary = df.groupby('kfold_id').apply(lambda x: compute_metrics_testing(x['gt_bin_lass'], x['pred_bin_class'])).values

    # Computing mean and variance for each metric
    data_binary = np.stack(data_binary)
    means = np.mean(data_binary, axis=0)
    variances = np.var(data_binary, axis=0)

    bias_binary= means[metric_index]
    variance_binary = variances[metric_index]

    return bias_triclass, variance_triclass, bias_binary, variance_binary



def compute_specificity(conf_matrix):
    """
    Compute the specificity for each class and return the mean specificity.
    Handles the case where the denominator is zero.
    """
    specificities = []
    for i in range(len(conf_matrix)):
        TN = conf_matrix[i, i]
        FP = np.sum(conf_matrix[:, i]) - TN

        # To avoid division by zero, check if the denominator is zero
        if FP + TN == 0:
            specificity = 0  # or choose a suitable default value
        else:
            specificity = TN / (FP + TN)

        specificities.append(specificity)

    return np.mean(specificities)

def compute_metrics_testing(gt, pred, average='weighted'):
    """
    Computes evaluation metrics for multi-class classification using ground truth labels and predictions.

    Args:
    - gt (array-like): Ground truth (correct) target values. This can be a list, numpy array, or a pandas series of integers.
    - pred (array-like): Estimated targets as returned by a classifier. The format is the same as for `gt`.
    - average (str): The averaging strategy for the precision, recall, and F1 score calculations.
      ``'micro'``: Calculate metrics globally by counting the total true positives, false negatives and false positives.
      ``'macro'``: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
      ``'weighted'``: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
                    This alters 'macro' to account for label imbalance; it can result in an F-score that is not between precision and recall.

    Returns:
    - tuple: A tuple containing the following metrics, in order:
        - sensitivity (float): The mean sensitivity (true positive rate) across all classes.
        - specificity (float): The mean specificity (true negative rate) across all classes.
        - balanced_perf (float): The average of sensitivity and specificity.
        - precision (float): Macro-average precision across all classes.
        - recall (float): Macro-average recall across all classes.
        - f1 (float): Macro-average F1 score across all classes.
        - balanced_accuracy (float): Balanced accuracy across all classes.
        - accuracy (float): Overall accuracy of the model.

    Note:
    - This function assumes binary or multi-class classification. It calculates specificity using a helper function
      `compute_specificity`, which must handle the confusion matrix for multi-class scenarios.
    - Sensitivity and specificity for multi-class classification are computed per class and then averaged.
    - It is critical that `gt` and `pred` are aligned and have the same length.
    - `zero_division=1` parameter in precision and recall calculations ensures that divisions by zero are handled gracefully,
      assigning a value of 1 to precision or recall if there are no positive predictions or true positives, respectively.

    This function should be the exclusive method for computing metrics in the project to ensure consistency in performance evaluation.
    """
    assert len(gt) == len(pred) # Ensure that the ground truth and predictions have the same length

    conf_matrix = confusion_matrix(gt, pred)
    # Calculate metrics for each class
    precision = precision_score(gt, pred, average=average, zero_division=1)
    recall = recall_score(gt, pred, average=average, zero_division=1)
    f1 = f1_score(gt, pred, average=average, zero_division=1)

    # Sensitivity and specificity calculations for multi-class
    sensitivity = np.mean([conf_matrix[i, i] / np.sum(conf_matrix[i, :]) for i in range(len(conf_matrix))])
    specificity = compute_specificity(conf_matrix)
    balanced_perf = (sensitivity+specificity) / 2
    accuracy = accuracy_score(gt, pred)
    balanced_accuracy = balanced_accuracy_score(gt, pred)

    return sensitivity, specificity, balanced_perf, precision, recall, f1, balanced_accuracy, accuracy
