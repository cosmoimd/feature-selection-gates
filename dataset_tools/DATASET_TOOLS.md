Here's a sample `README.md` file for the script:

---

# Feature Selection Gates Preprocessing Script

This repository contains a Python script designed for preprocessing raw datasets of endoscopic images, focusing specifically on frames that contain polyps. The script is part of the work presented in the following paper:

**Accepted Paper:**  
*Feature Selection Gates with Gradient Routing for Endoscopic Image Computing*  
Giorgio Roffo, Carlo Biffi, Pietro Salvagnini, Andrea Cherubini  
MICCAI 2024, 27th International Conference on Medical Image Computing and Computer-Assisted Intervention, Marrakech, Morocco, October 2024.  
Please cite this paper if you use any part of the code or methods presented here.

**Preprint Version:**  
*Hard-Attention Gates with Gradient Routing for Endoscopic Image Computing*  
Giorgio Roffo, Carlo Biffi, Pietro Salvagnini, Andrea Cherubini  
arXiv:2407.04400

## Overview

The script provided in this repository processes raw endoscopic image datasets, extracting frames that contain polyps and saving relevant information in CSV files. It supports two main datasets: IMDRealColon and SUN. The script also includes functionalities for splitting the dataset into K-folds for cross-validation and for generating statistical summaries of the processed data.

## Features

- **Dataset Preprocessing**: The script reads raw image datasets and XML annotations to identify and extract frames containing polyps.
- **CSV Generation**: Extracted frames are saved into CSV files for further use in machine learning pipelines.
- **K-Fold Splitting**: The script can split the preprocessed dataset into multiple folds for cross-validation.
- **Statistical Analysis**: Generate and visualize statistics about the dataset.

## Requirements

- Python 3.8+
- PyTorch
- Other dependencies are specified in the `requirements.txt` file.

## Script Description

### Command Line Arguments

- `-parFile`: (Optional) Path to the parameter file in YAML format. This file contains dataset details such as paths, output folders, and other configurations. If not provided, a default parameter file located at `config/config_resnet18_RGB.yml` will be used.

### Workflow

1. **Parameter File Loading**:  
   The script begins by loading the specified parameter file. If the file path is invalid or not provided, it defaults to a predefined YAML file.

2. **Output Folder Setup**:  
   The script checks whether the output folder specified in the parameter file exists:
   - If it does, the user is prompted to either delete and recreate it or continue using the existing folder.
   - If it doesn’t exist, the folder is created.

3. **Dataset Preprocessing**:  
   For each dataset specified in the parameter file:
   - The script instantiates a dataset-specific preprocessor.
   - It traverses all directories and XML annotations within the dataset, extracting relevant data into a DataFrame.
   - The DataFrame is periodically saved into CSV files in the output folder.

4. **K-Fold Splitting**:  
   If dataset processing is performed, the script generates K-fold splits for cross-validation.

5. **Statistical Analysis**:  
   The script generates various statistical summaries and visualizations for the processed dataset.

### Example Usage

To run the script with a specific parameter file:

```bash
python preprocess.py -parFile /path/to/parameter_file.yml
```

To use the default parameter file:

```bash
python preprocess.py
```

### Output

- **CSV Files**: The output folder will contain CSV files with data extracted from the datasets.
- **K-Fold Splits**: If K-fold splitting is enabled, the output folder will also contain files representing each fold.
- **Statistics**: Plots and statistics files will be generated, providing insights into the dataset's composition and distribution.

### Notes

- Ensure that the parameter file is correctly configured, including paths to datasets and output directories.
- The script assumes that the datasets are structured and annotated as expected (e.g., XML annotations are available for each frame).

## License

This code is licensed under a private license. Please refer to the `LICENSE` file for more details.

## Contact

For any questions or issues, please contact Giorgio Roffo at groffo@cosmoimd.com.

---
