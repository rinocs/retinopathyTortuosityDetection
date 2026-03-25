# Retinopathy Tortuosity Detection

A deep learning pipeline designed to estimate the tortuosity of retinal vessels using regression models. This repository contains tools to process retinal vessel images, extract tortuosity metrics (like Vessel Tortuosity Index - VTI), and train neural networks to estimate these values directly from images.

## Overview

Retinal vessel tortuosity is an important biomarker for diagnosing and monitoring several retinopathies. This project provides a series of scripts to:
1. Generate datasets by applying morphological operations to vessel segments.
2. Measure ground-truth tortuosity using mathematical formulations.
3. Train deep learning regressors to estimate tortuosity.
4. Compare ground-truth metrics with model predictions.

Intended for medical researchers, students, and machine learning practitioners interested in automated retinal image analysis.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Pipeline](#usage-pipeline)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Model Training](#2-model-training)
  - [3. Evaluation](#3-evaluation)
- [Methodology](#methodology)

## Installation

Ensure you have Python 3.8+ installed. Clone this repository and install the required dependencies:

```bash
git clone <repository_url>
cd retinopathyTortuosityDetection
pip install -r requirements.txt
```

## Project Structure

The codebase has been organized into a modular structure:

- `src/` - Core source code:
  - `src/data/`: Scripts for data loading (`data_nn_loader.py`), augmentation (`augmenti_images.py`), and dataset generation (`tort_data_gen.py`).
  - `src/models/`: Contains the core regressor training code (`regressor.py`).
- `scripts/`: Runnable scripts for evaluating and comparing predictions (`compare_art_tort.py`, `rot-test-artery.py`, etc.).
- `utils/`: Helper utilities for image processing, math (fractals, tortuosity metrics), and keras model architectures.
- `notebooks/`: Jupyter notebooks for exploratory analysis and training (`Transfer_learning_reg_kfold.ipynb`).
- `experimental/`: Scratchpads and experimental scripts.

## Usage Pipeline

### 1. Data Preparation
To train the model, you first need to generate training data containing cropped vessel segments and their calculated tortuosity values.

Run the data generation scripts located in `src/data/`. To ensure Python resolves internal module imports correctly, run the script as a module from the repository root:
```bash
python -m src.data.tort_data_gen
```
This script processes raw retinal vessel skeletons, crops them, calculates tortuosity features (like VTI), and outputs the formatted data into the `Data/` directory.

### 2. Model Training
Once the dataset is prepared, you can train the deep learning regressor. The regressor script uses Convolutional Neural Networks (CNNs) built with Keras to learn the mapping between the vessel image and its tortuosity.

Run the model training as a module:
```bash
python -m src.models.regressor
```
This will read the generated `.csv` or `.h5` files, split the data, train the model, and save checkpoints.

*Note: You can also use the interactive Jupyter notebook in `notebooks/Transfer_learning_reg_kfold.ipynb` to run experiments with k-fold cross-validation.*

### 3. Evaluation
To evaluate how well the model predictions align with the mathematical tortuosity metrics, use the evaluation scripts in the `scripts/` directory. Again, run them as modules from the root directory:

```bash
python -m scripts.compare_art_tort
```
This will generate scatter plots and calculate R-squared/MSE scores comparing the ground truth metrics against the model's output.

## Methodology

This pipeline utilizes several mathematical metrics to define the ground-truth tortuosity of a vessel segment:
- **Vessel Tortuosity Index (VTI):** A comprehensive metric combining distance, inflection points, and density.
- **Distance Ratio:** The ratio between the actual path length of the vessel and the linear distance between its endpoints.
- **Fractal Dimension:** Measures the complexity of the vessel path.

The CNN regressor learns to approximate these features directly from the 2D skeletonized or segmented vessel images.