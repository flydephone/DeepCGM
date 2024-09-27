
# DeepCGM: A Deep Learning-Based Crop Growth Model

**DeepCGM** is a deep learning model designed to predict crop growth and yield based on various environmental and crop-specific features. This repository contains the code, data preprocessing scripts, and training configurations for replicating the results and further developing the model.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Data](#data)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

DeepCGM leverages deep learning techniques to simulate crop growth processes under different environmental conditions. The model is capable of learning from spatial and temporal data, offering robust predictions on crop yield and growth patterns.

**Key Objectives**:
- Predict crop growth dynamics.
- Simulate the impact of environmental stress on crops.
- Facilitate decision-making for precision agriculture.

## Features

- **Deep Learning Framework**: Utilizes advanced neural networks like LSTMs, GRUs, and CNNs.
- **Crop-Specific Modeling**: Supports multiple crops such as maize, wheat, and sugar beet.
- **Scalability**: Built for high-dimensional spatial data and large datasets.
- **Data Integration**: Handles raster and tabular datasets including weather, soil, and crop information.
- **Customizable**: Easily customizable for different regions, crops, and environmental conditions.

## Installation

To install the dependencies, clone the repository and use the following commands:

```bash
git clone https://github.com/yourusername/DeepCGM.git
cd DeepCGM
pip install -r requirements.txt
```

## Usage

1. **Preprocess Data**: Prepare your input data using the provided scripts.
2. **Train the Model**: Train DeepCGM using your data with custom parameters.
3. **Evaluate the Model**: Test model performance on unseen data.

Example command to start training:
```bash
python train.py --data_dir /path/to/data --epochs 100 --batch_size 32
```

## Model Architecture

DeepCGM consists of the following components:
- **Input Layer**: Accepts multi-source data (e.g., weather, soil properties, and crop characteristics).
- **LSTM/GRU Layers**: Capture temporal dependencies in crop growth.
- **CNN Layers**: Extract spatial patterns from raster data.
- **Fully Connected Layers**: Integrate learned features and output crop growth predictions.

You can customize the architecture by modifying the `model.py` script.

## Training

To train the model, ensure you have the necessary dataset in the correct format. You can adjust hyperparameters like learning rate, batch size, and the number of epochs in `config.json`.

```bash
python train.py --config config.json
```

## Evaluation

Evaluate the model performance using various metrics such as RMSE, MAE, and R2:

```bash
python evaluate.py --model_path /path/to/saved_model
```

The results will be saved in the `results/` folder.

## Data

The dataset includes:
- **Weather data** (temperature, rainfall, humidity, etc.).
- **Soil data** (nutrient levels, pH, moisture, etc.).
- **Crop management data** (sowing date, irrigation schedule, etc.).

Ensure your data follows the structure specified in `data/README.md`.

## Results

Here are some results from our test runs:
- **RMSE**: 5.4
- **R2 Score**: 0.89

For detailed results and model performance on different crops, check the `results/` folder.

## Contributing

We welcome contributions from the community. To contribute:
1. Fork this repository.
2. Create a branch (`git checkout -b feature-xyz`).
3. Commit your changes (`git commit -m "Add feature xyz"`).
4. Push to the branch (`git push origin feature-xyz`).
5. Create a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
