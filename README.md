# DeepCGM: A Knowledge-Guided Deep Learning Crop Growth Model

**DeepCGM** is a novel deep learning architecture that leverages domain knowledge and sparse observational data to simulate crop growth dynamics. This repository contains the model code, dataset formatting scripts, figures, and utilities for training and evaluating the model.

![Model Structure](figure/Framework%20for%20training%20the%20DeepCGM%20model%20and%20assessing%20the%20performance.svg)

## License

All rights reserved. This work is **under review, and no permissions** are granted for use, modification, or distribution until further notice.

The source code and data will be made publicly available after the publication of the paper.

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
git clone https://github.com/flydephone/DeepCGM.git
cd DeepCGM
pip install -r requirements.txt
```

## Repository Structure

- **`requirements.txt`**: Requirements.
- **`train.py`**: Script to train the DeepCGM model.
- **`utils.py`**: Utility functions.
- **`fig_5.py`, `fig_6.py`, `fig_7.py`, `fig_9.py`, etc.**: Scripts to generate figures for model results.
- **`models_aux`**: Folder containing models.
  - **`DeepCGM.py`** is the DeepCGM model 100% following the [detail process of DeepCGM](figure)
  - **`DeepCGM_fast.py`** improve the model speed by combining the gate calculation according to [this suggestion](https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/) and by combining the redistribution calculation.
  - **`MCLSTM.py`** and **`MCLSTM_fast.py`** are raw MCLSTM and speed improved MCLSTM
- **`format_dataset`**: Formatted dataset.
- **`figure`**: Folder for storing figures generated during model evaluation and analysis.

## Model Architecture

DeepCGM is a **deep learning-based crop growth model** with a [mass-conserving architecture](https://arxiv.org/abs/2101.05186) and crop growth process. The architecture ensures that simulated crop growth adheres to physical principles:

![Model Structure](figure/DeepCGM.svg)

## Data

All data are time series data.

- **Input**: Radiation, Maximun temperature, Minimun temperature, cumulative nigrogen, Development stage (simulated by ORYZA2000), etc.
- **Output**: Plant area index (PAI), Organ biomass (leaf, stem, grain), Above ground biomass, Yield

### Train the Model

Run the `train.py` script to train the model:

```bash
python train_from_scratch.py --model DeepCGM --target spa --input_mask 1 --convergence_loss 1 --tra_year 2018
```

You can modify the training parameters, such as model type, knowledge triggers, and training years

### Arguments:

- **--model**: Specifies the model type (`NaiveLSTM`,`MCLSTM`, `DeepCGM`).
- **--target**: Specifies the training label ( `spa` for sparse dataset and `int` for interpolated dataset).
- **--input_mask**: Enables the input mask (`1` to enable, `0` to disable).
- **--convergence_trigger**: Enables the convergence_loss (`1` to enable, `0` to disable).
- **--tra_year**: Specifies the training year (e.g., `2018` and `2019`).

## Training flowchat

The `fitting loss`, `convergence loss` and `input mask` can be used in training DeepCGM

![Training flowchart](figure/Traing.svg)

### Evaluate the Model

Use the figure scripts (e.g., `fig_5.py`, `fig_6.py`, etc.) to generate visualizations of the model's performance. Example:

```bash
python fig_5.py
```

These scripts generate figures to evaluate the model's predictions across multiple variables and datasets, the result is:

![Time series result](figure/Fig.5%20Crop%20growth%20simulation%20results.svg)

**DeepCGM outperforms traditional process-based models (Normlized MSE):**

|           | 2018-train 2019-test | 2019-train 2018-test |
| --------- | -------------------- | -------------------- |
| ORYZA2000 | 0.0381               | 0.0474               |
| DeepCGM   | 0.0349               | 0.0393               |

More results are saved in the `figure` folder, and detailed evaluation figures are generated using the provided scripts.
