<div align="center">

![image](https://github.com/user-attachments/assets/05f1ea5d-5436-45f0-95a8-b70f105dd965)


</div>

<div align="center">

<table>
  <tr>
   <td><a href="https://arxiv.org/pdf/2407.04400" target="_blank">Paper PDF</a></td>
   <td><a href="https://github.com/cosmoimd/feature-selection-gates/blob/main/MICCAI_2024_official_dataset_splits/MICCAI2024-FSG-GR-datasets-official-splits-.zip" target="_blank">Dataset Splits</a></td>
    <td><a href="https://www.researchgate.net/profile/Giorgio-Roffo" target="_blank">Author Page</a></td>
    <td><a href="https://scholar.google.it/citations?user=cD2LFuUAAAAJ&hl=en" target="_blank">Google Scholar</a></td>
  </tr>
</table>

</div>

**Keywords**: Endoscopic Image Computing, Feature Selection Gates, Hard-Attention Gates, Gradient Routing, CNNs, Vision Transformers, Gastroenterological Polyp Size Estimation, Medical Image Analysis, Overfitting Reduction, Model Generalization.

## MICCAI 2024, the 27th International Conference on Medical Image Computing and Computer Assisted Intervention, Marrakech, Morocco, October 2024.

## Work in Progress... 7-9 October 2024. 

# Feature Selection Gates with Gradient Routing Toolbox

This repository contains the official implementation of the paper "Feature Selection Gates with Gradient Routing for Endoscopic Image Computing", presented at MICCAI 2024. This toolbox provides implementations for CNNs, multistream CNNs, ViTs, and their augmented variants using Feature-Selection Gates (FSG) or Hard-Attention Gates (HAG) with Gradient Routing (GR). The primary objective is to enhance model generalization and reduce overfitting, specifically in the context of gastroenterological polyp size assessment.

# Citing this Work

If you find this toolbox useful in your research, please cite the following papers:

Accepted Publication:
~~~~
@inproceedings{roffo2024FSG,
   title={Feature Selection Gates with Gradient Routing for Endoscopic Image Computing},
   author={Giorgio Roffo and Carlo Biffi and Pietro Salvagnini and Andrea Cherubini},
   booktitle={MICCAI 2024, the 27th International Conference on Medical Image Computing and Computer Assisted Intervention, Marrakech, Morocco, October 2024.},
   year={2024},
   organization={Springer}
}
~~~~


Preprint Version:
~~~~
@misc{roffo2024hardattention,
   title={Hard-Attention Gates with Gradient Routing for Endoscopic Image Computing},
   author={Giorgio Roffo and Carlo Biffi and Pietro Salvagnini and Andrea Cherubini},
   year={2024},
   eprint={2407.04400},
   archivePrefix={arXiv},
   primaryClass={eess.IV}
}
~~~~
We extend our gratitude to the MICCAI community and all collaborators for their invaluable contributions and support.

## Summary

In this work, we present *Feature-Selection Gates* (FSG), also known as *Hard-Attention Gates* (HAG), along with a novel approach called *Gradient Routing* (GR) for Online Feature Selection (OFS) in deep learning models. This method aims to enhance performance in endoscopic image computing by reducing overfitting and improving generalization.

**Key contributions:**

- **FSG/HAG:** Implements sparsification with learnable weights, serving as a regularization strategy to promote sparse connectivity in neural networks (Convolutional and Vision Transformer models).
- **GR:** Optimizes FSG/HAG parameters through dual forward passes, independent of the main model, refining feature re-weighting.
- **Performance Improvement:** Validated across multiple datasets, including CIFAR-100 and specialized endoscopic datasets (REAL-Colon, Misawa, and SUN), showing significant gains in binary and triclass polyp size classification.


## Toolbox Structure
Feature Selection/Attention Gates with Gradient Routing for Online Feature Selection.

~~~~
├── example_configs
├── gr_checkpoints
│   ├── miccai24_FSG_GR_vit.zip
│   └── pretrained_models.txt
├── MICCAI_2024_official_dataset_splits
│   ├── MICCAI2024-FSG-GR-datasets-official-splits-.zip
│   ├── per_object_gt_group_distribution_per_fold.png
│   ├── per_object_gt_group_distribution_per_unique_id_per_fold.png
│   └── per_object_kfold_distribution.png
├── modules
│   ├── analytics
│   │   └── calculate_metrics.py
│   ├── datasets
│   │   ├── dataset.py
│   │   └── sampler.py
│   ├── losses
│   │   ├── classification_loss.py
│   │   └── weighted_size_combined_loss.py
│   ├── models
│   │   ├── gr_transfutils
│   │   │   ├── fsg_vision_transformers.py
│   │   │   ├── multi_stream_nets.py
│   │   │   └── vision_transformers.py
│   ├── schedulers
│   │   └── cosine_annealing_warm_restarts.py
│   ├── transforms
│   │   ├── transforms_sizing.py
│   │   └── base_params.py
├── runners
│   └── build_configuration.py
├── utils.py
└── README.md

~~~~

## Modules Overview

* analytics: Includes tools for evaluating FSG score distributions and calculating various performance metrics.
* datasets: Handles data loading, preprocessing, and sampling for various dataset formats.
* losses: Defines custom loss functions, including weighted loss strategies for size estimation.
* models: Contains PyTorch model definitions, with scripts for constructing FSG-enhanced CNNs and ViTs.
* schedulers: Learning rate schedulers to optimize training dynamics.
* transforms: Image transformation and augmentation techniques for data preprocessing.

## Datasets Download & Preparation

### REAL-colon Dataset

- **Download Link**: [REAL-colon Dataset on Figshare](https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866)
- **GitHub Repository**: [REAL-colon Dataset Code](https://github.com/cosmoimd/real-colon-dataset)

The REAL (Real-world multi-center Endoscopy Annotated video Library) - colon dataset comprises 60 recordings of real-world colonoscopies from four different clinical studies, each contributing 15 videos. The dataset includes:

- **Total Size**: Approximately 880.78 GB
- **Frames**: 2,757,723 total frames
- **Polyps**: 132 removed colorectal polyps
- **Annotations**: 351,264 bounding box annotations

The dataset is organized as follows:

- 60 compressed folders named `{SSS}-{VVV}_frames` containing video frames (`SSS` indicates the clinical study, `VVV` represents the video name).
- 60 compressed folders named `{SSS}-{VVV}_annotations` containing video annotations for each recording.
- `video_info.csv`: Metadata for each video.
- `lesion_info.csv`: Metadata for each lesion, including endoscope brand, bowel cleanliness score, number of surgically removed colon lesions, and more.
- `dataset_description.md`: A README file with information about the dataset.

To automatically download the dataset, run `figshare_dataset.py` from the GitHub repository. The script will download the dataset into the `./dataset` folder by default. You can change the output folder by setting the `DOWNLOAD_DIR` variable in `figshare_dataset.py`. Given the large size of the dataset, ensure you have sufficient storage and bandwidth before starting the download.


### SUN Colonoscopy Video Database

- **Download Link**: [SUN Colonoscopy Video Database](http://sundatabase.org/)
- **Request Download Link**: Email hitoh@mori.m.is.nagoya-u.ac.jp

The database is available for non-commercial research or educational purposes only. Commercial use is prohibited without permission. Proper citation is required when using the dataset:
For access, send a request email to [hitoh@mori.m.is.nagoya-u.ac.jp](mailto:hitoh@mori.m.is.nagoya-u.ac.jp).


The SUN (Showa University and Nagoya University) Colonoscopy Video Database is designed for evaluating automated colorectal-polyp detection systems. It includes:

- **Total Frames**: 158,690 frames
  - **Polyp Frames**: 49,136 frames from 100 polyps, annotated with bounding boxes
  - **Non-Polyp Frames**: 109,554 frames

The database is organized as follows:

- **Polyp Frame Annotations**: Each polyp frame is annotated with bounding boxes provided in text files. Each line in the text file corresponds to a bounding box in the format: `Filename min_Xcoordinate,min_Ycoordinate,max_Xcoordinate,max_Ycoordinate,class_id`. Class_id 0 represents polyp frames, and class_id 1 represents non-polyp frames.
- **Image Formats**: JPEG for images, text files for bounding box annotations.

Database characteristics:

- **Patients**: 99 (71 males, 28 females)
- **Median Age**: 69 years (IQR: 58–74)
- **Polyps**: 100 polyps with details including size, morphology, location, and pathological diagnosis.


## Usage

* Clone the repository.
* Configure your experiments using the YAML files within the config/ directory.
* To preprocess your datasets, run preprocess_raw_datasets.py.
* To begin training, execute main_train_and_infer.py.
* For evaluation and testing, use main_testing.py.


## Dataset Preprocessing Script (preprocess_raw_datasets.py)

This script (`preprocess_raw_datasets.py`) preprocesses raw datasets like the REAL-colon and SUN Colonoscopy Video Database, making them ready for deep learning model training.

### What the Script Does

1. **Parameter File Handling**:
    - Reads a provided parameter file or uses a default configuration file.
    - Specifies dataset paths, output folder, and other settings.

2. **Dataset Preprocessing**:
    - Checks and prepares the output folder.
    - Processes datasets to extract frames containing polyps and saves them in CSV format.
    - Handles REAL-colon and SUN datasets specifically with appropriate preprocessors.

3. **K-Fold Split Creation**:
    - Generates K-fold splits for cross-validation if creating or recreating the dataset.
    - You can download and use the **official MICCAI 2024 splits** or create new ones.
   
4. **Data Statistics Generation**:
    - Produces and displays statistics about the dataset.

### How to Use the Script

1. **Download the Datasets**:
    - Download the REAL-colon dataset from [Figshare](https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866).
    - Request access to the SUN Colonoscopy Video Database by emailing [hitoh@mori.m.is.nagoya-u.ac.jp](mailto:hitoh@mori.m.is.nagoya-u.ac.jp).

2. **Run the Script**:
    - Ensure the datasets are downloaded to the specified paths.
    - Execute the script with the parameter file:
      ```bash
      python preprocess_raw_datasets.py -parFile path/to/your/parameter_file.yml
      ```
    - If no parameter file is provided, the script will use a default configuration file.

The script simplifies dataset preparation, enabling efficient training of deep learning models on standardized data.

## Support

For inquiries or support regarding the implementation or the paper, please reach out to the corresponding authors (#giorgioroffo) via the contact information provided in the paper.

## License

This project is licensed for use and is subject to the terms outlined in the LICENSE file.
