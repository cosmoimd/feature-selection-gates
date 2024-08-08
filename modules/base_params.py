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

# -----------------------------------------------------------------
# FSAG: Feature-Selection-Attention Gates
# Gradient Routing (GR) for Online-Feature-Selection (OFS)
# ----------------------------------------------------------------
# File information
__author__ = "Giorgio Roffo, Dr."
__copyright__ = "Copyright 2024, COSMO IMD"
__credits__ = ["Giorgio Roffo"]
__license__ = "Private"
__version__ = "1.0.1"
__maintainer__ = "groffo"
__email__ = "groffo@cosmoimd.com,giorgio.roffo@gmail.com"
__status__ = "Production"  # can be "Prototype", "Development" or "Production"

# ensure that all keys exist in the dictionary and set default values
# update these defaults as necessary
default_loss_params = {
    'alpha': 0.25,
    'beta': 0.25,
    'gamma': 0.50,
    'size_threshold_1': 5.0,
    'weight_1': 1.0,
    'size_threshold_2': 10.0,
    'weight_2': 1.0,
}

# ensure that all keys exist in the dictionary and set default values
# update these defaults as necessary
default_model_params = {
    'depth_maps': False,
    'num_classes': 1,
    'num_fc_layers': 3,
    'nodes_fc_layers': [256, 256],
    'dropout_rate': 0.35,
    'pretrained': True,
}

# ensure that all keys exist in the dictionary and set default values
# update these defaults as necessary
default_dataset_params = {
    'output_folder': '/path/to/folder/',
    'precision_float_accuracy': 1.0,  # +/- 1 mm
    'target_process_min': 0.5,
    'target_process_max': 25.0,
    'n_frames_to_sample': 100,
    'batch_size': 64,
    'input_dim_resize_to': 384,
    'filter_centered_samples_ratio': 1.0,
    'image_preprocessing': {
        "mean": [0.32239652, 0.22631808, 0.17500061],
        "std": [0.31781745, 0.2405859, 0.19327126],
        "circular_crop": True
    },
    'include_depth_maps': True,
    'include_rgb_mask': True,
    'num_workers': 4,
    'datasets': [
        ['data1', '/path/data1.csv', 'gt_name1'],
        ['data2', '/path/data2.csv', 'gt_name2'],
        ['data3', '/path/data3.csv', 'gt_name3']
    ],
}
