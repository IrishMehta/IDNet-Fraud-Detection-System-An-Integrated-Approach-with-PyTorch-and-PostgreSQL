# IDNet-Fraud-Detection-System-An-Integrated-Approach-with-PyTorch-and-PostgreSQL
This repository is an end to end identity card fradud detection system that uses inference querying for generating insights. This project was completed as part of CSE 598- Data Intensive Systems for Machine Learning under Prof. Jia Zou

# Fraud Detection CNN Pipeline

This project implements a CNN-based pipeline to detect visual fraud indicators in identity documents. It automates the process from data preparation to model training and deployment on Hugging Face Hub.

## Project Overview

Addresses the challenge of identity document fraud by providing an automated detection system using Convolutional Neural Networks (CNNs). Aims for high accuracy (>95%) and low latency (<2 seconds per document).

**(Note:** The full project scope includes PostgresML integration, which is not covered by this specific code pipeline.)

## Features

*   **Data Preparation:** Extracts images from base64 encoded data and splits them into stratified train/test/out-of-sample sets.
*   **Hyperparameter Tuning:** Uses Optuna to find optimal hyperparameters for the ResNet50 model.
*   **Model Training:** Trains a ResNet50 model for binary classification (Fraud/Non-Fraud) with class weighting, learning rate scheduling, and early stopping.
*   **Evaluation:** Generates classification reports, confusion matrices, and training history plots.
*   **Hugging Face Integration:** Packages the trained model, configuration, and processor for easy upload and use via the Hugging Face Hub.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/IrishMehta/IDNet-Fraud-Detection-System-An-Integrated-Approach-with-PyTorch-and-PostgreSQL.git
    cd "https://github.com/IrishMehta/IDNet-Fraud-Detection-System-An-Integrated-Approach-with-PyTorch-and-PostgreSQL.git"
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```
    pip install torch torchvision pandas scikit-learn optuna matplotlib Pillow huggingface_hub transformers requests
    ```

4.  **Prepare Data:** Place your `idimage.csv`, `idmeta.csv`, and `idlabel.csv` files inside a `data_files` directory (or update paths in `config.json`).

## Usage

The pipeline consists of several steps, executed via Python scripts. Ensure `config.json` is correctly configured before running.

1.  **Prepare Dataset:**
    Extracts images and creates train/test/OOS splits.
    ```bash
    python dataset_creation.py
    ```
    *(Outputs images to `extracted_images/` and splits to `dataset/`)*

2.  **Hyperparameter Tuning:**
    Runs Optuna to find the best hyperparameters.
    ```bash
    python hyperparameter_tuning.py
    ```
    *(Outputs best parameters to `hpo_output/best_hyperparams.json`)*
    *(Note: Current `training.py` uses hardcoded hyperparameters; update it if you want to use tuned ones.)*

3.  **Train Model:**
    Trains the final model using parameters defined in the script or `config.json`.
    ```bash
    python training.py
    ```
    *(Outputs model (`best_model.pth`), history, and plots to `training_output/`)*

4.  **Upload to Hugging Face Hub:**
    Packages the model and uploads it to the specified Hugging Face repository.
    ```bash
    python hf_upload.py
    ```
    *(Creates `fraud_detection_hf/` folder locally before upload. Requires Hugging Face Hub credentials set in `config.json` or environment variables.)*

## Configuration

Pipeline behavior is controlled by `config.json`. Key sections:

*   `data_prep`: Paths for input CSVs, output directories for extracted images and dataset splits, split ratios.
*   `hpo`: Paths for hyperparameter optimization, validation split ratio, number of classes/epochs/trials, output directory.
*   `training`: Paths for data directories, validation split ratio, best model output path, output directory for training artifacts.
*   `huggingface`: Paths for Hugging Face model packaging, local model weights, HF tokens, repository name, and an optional sample image for inference testing.

**Note:** Ensure file paths and Hugging Face credentials (`HF_WRITE_TOKEN`, `HF_READ_TOKEN`, `HF_REPO_NAME`) are correctly set either in the config or as environment variables.

## Project Structure

```
.
├── config.json                 # Configuration file
├── dataset_creation.py         # Script for data preparation and splitting
├── hyperparameter_tuning.py    # Script for HPO using Optuna
├── training.py                 # Script for model training and evaluation
├── hf_upload.py                # Script for packaging and uploading to Hugging Face Hub
├── data_files/                 # Input CSV files (idimage.csv, idmeta.csv, idlabel.csv)
├── extracted_images/           # Output directory for decoded images
├── dataset/                    # Output directory for train/test/oos splits
│   ├── train/
│   ├── test/
│   └── out_of_sample_test/
├── hpo_output/                 # Output directory for HPO results
├── training_output/            # Output directory for training results (model, plots, history)
├── fraud_detection_hf/         # Staging directory for Hugging Face upload
└── requirements.txt            # Python dependencies
```

# Integration with PostgreSQL



