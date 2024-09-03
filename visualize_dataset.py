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

# Installation Instructions:
# Before running this script, ensure that you have FiftyOne installed.
# FiftyOne is an open-source tool for exploring, visualizing, and analyzing datasets.
# You can learn more and access documentation here: https://voxel51.com

# To install FiftyOne, follow these steps:
# 1. Upgrade pip to the latest version:
#    pip install --upgrade pip
# 2. Install FiftyOne:
#    pip install fiftyone

import fiftyone as fo
from fiftyone import types

# Specify the path to your dataset
data_path = "/path/to/your/dataset/png_jpg/"

# Create a FiftyOne dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=data_path,
    dataset_type=fo.types.ImageDirectory,
)

# Launch the FiftyOne app with your dataset
session = fo.launch_app(dataset)

