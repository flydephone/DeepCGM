
# DeepCGM: A Knowledge-Guided Deep Learning Crop Growth Model

**DeepCGM** is a novel deep learning architecture that leverages domain knowledge and sparse observational data to simulate crop growth dynamics. This repository contains the model code, dataset formatting scripts, figures, and utilities for training and evaluating the model.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Data](#data)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Crop growth modeling is essential for understanding and predicting agricultural outcomes. Traditional **process-based crop models**, like ORYZA2000, are effective but often suffer from oversimplification and parameter estimation challenges. **Machine learning methods**, though promising, are often criticized for being "black-box" models and requiring large datasets that are frequently unavailable in real-world agricultural settings.

**DeepCGM** addresses these limitations by integrating knowledge-guided constraints into a deep learning model to ensure physically plausible crop growth simulations, even with sparse data.

## Features

- **Mass-Conserving Deep Learning Architecture**: Adheres to crop growth principles such as mass conservation to ensure physically realistic predictions.
- **Knowledge-Guided Constraints**: Includes crop physiology and model convergence constraints, enabling accurate predictions with sparse data.
- **Improved Accuracy**: Outperforms traditional process-based models and classical deep learning models on real-world crop datasets.
- **Multivariable Prediction**: Simulates multiple crop growth variables (e.g., biomass, leaf area) in a single framework.

## Installation

To install the dependencies, clone the repository and install the required packages using the command below:

```bash
git clone https://github.com/yourusername/DeepCGM.git
cd DeepCGM
pip install -r requirements.txt
```

## Repository Structure

- **`DeepCGM.yaml`**: Configuration file for model training and evaluation.
- **`train.py`**: Script to train the DeepCGM model.
- **`utils.py`**: Utility functions for data preprocessing and model support.
- **`fig_5.py`, `fig_6.py`, `fig_7.py`, `fig_9.py`, etc.**: Scripts to generate figures for model results.
- **`models_aux`**: Folder containing auxiliary models and related code.
- **`format_dataset`**: Folder for scripts related to dataset formatting and preprocessing.
- **`figure`**: Folder for storing figures generated during model evaluation and analysis.

## Usage

### 1. Preprocess the Data

Format your input data using the provided scripts in the `format_dataset` folder.

### 2. Train the Model

Run the `train.py` script to train the model using your formatted data:

```bash
python train.py --config DeepCGM.yaml
```

You can modify the training parameters, such as learning rate, batch size, and epochs, in the `DeepCGM.yaml` file.

### 3. Evaluate the Model

Use the figure scripts (e.g., `fig_5.py`, `fig_6.py`, etc.) to generate visualizations of the model's performance. Example:

```bash
python fig_5.py
```

These scripts generate figures to evaluate the model's predictions across multiple variables and datasets.

## Model Architecture

DeepCGM is a **deep learning-based crop growth model** with a mass-conserving architecture. The architecture ensures that simulated crop growth adheres to physical principles, including:

- **Mass-Conserving Layers**: Ensure physically plausible crop growth curves.
- **LSTM/GRU Layers**: Capture temporal dependencies in crop growth data.
- **CNN Layers**: Extract spatial patterns from multi-source datasets.

## Training

Training is done using the `train.py` script with configurations defined in `DeepCGM.yaml`. Key features include:

- **Knowledge-Guided Constraints**: Crop physiology is incorporated to avoid unrealistic predictions.
- **Sparse Data Training**: Efficiently uses limited datasets typical of agricultural research.

```bash
python train.py --config DeepCGM.yaml
```

## Evaluation

Evaluate the model's performance using various metrics such as **normalized mean square error (MSE)**, and generate figures using the provided scripts (e.g., `fig_5.py`, `fig_6.py`).

```bash
python fig_5.py
```

## Data

The dataset should include:

- **Weather Data**: Temperature, rainfall, humidity, etc.
- **Soil Data**: Nutrient levels, pH, moisture content, etc.
- **Management Data**: Sowing date, irrigation schedule, etc.

Ensure the data format follows the structure described in the `format_dataset` folder.

## Results

DeepCGM generates **physically plausible crop growth curves** and outperforms traditional process-based models:

- **Normalized MSE Improvement**: 
  - From 0.0381 to 0.0338 (2019)
  - From 0.0473 to 0.0397 (2018)

The results are saved in the `figure` folder, and detailed evaluation figures are generated using the provided scripts.

## Contributing

We welcome contributions! To contribute:

1. Fork this repository.
2. Create a branch (`git checkout -b feature-xyz`).
3. Commit your changes (`git commit -m "Add feature xyz"`).
4. Push to the branch (`git push origin feature-xyz`).
5. Create a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
