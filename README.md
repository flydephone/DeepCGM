
# DeepCGM: A Knowledge-Guided Deep Learning Crop Growth Model

**DeepCGM** is a novel deep learning architecture that leverages domain knowledge and sparse observational data to simulate crop growth dynamics. The model is designed to address the limitations of both process-based crop models and classical machine learning approaches by incorporating crop growth mechanisms as constraints, ensuring physically plausible predictions even with limited data.

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

Crop growth modeling is essential for understanding and predicting agricultural outcomes. Traditional **process-based crop models**, like ORYZA2000, are effective but often suffer from oversimplification and parameter estimation challenges. Meanwhile, **machine learning methods**, though promising, are often criticized for being "black-box" models that ignore the underlying biology of crop growth and require large datasets that are often unavailable.

**DeepCGM** overcomes these limitations by combining the strengths of process-based models and machine learning. It integrates crop growth principles directly into its architecture, resulting in more accurate, interpretable, and data-efficient predictions for crop growth.

### Key Objectives

- **Accurate Crop Growth Simulation**: Simulate crop growth dynamics for multiple variables while ensuring physically plausible results.
- **Knowledge-Guided Learning**: Incorporate crop growth principles (such as mass conservation) as knowledge constraints into the learning process.
- **Data Efficiency**: Achieve accurate predictions with sparse datasets, making the model suitable for real-world agricultural applications with limited data.

## Features

- **Mass-Conserving Deep Learning Architecture**: Adheres to crop growth principles such as mass conservation to ensure physically realistic predictions.
- **Knowledge-Guided Constraints**: Includes constraints related to crop physiology and model convergence, enabling accurate predictions even with sparse observational data.
- **Improved Accuracy**: Outperforms traditional process-based models and classical deep learning models on real-world crop datasets.
- **Multivariable Prediction**: Simulates multiple crop growth variables (e.g., biomass, leaf area) in a single framework.

## Installation

To install the dependencies, clone the repository and use the following commands:

```bash
git clone https://github.com/yourusername/DeepCGM.git
cd DeepCGM
pip install -r requirements.txt
```

## Usage

1. **Preprocess Data**: Use the provided scripts to preprocess the input data.
2. **Train the Model**: Train DeepCGM using your data with custom parameters.
3. **Evaluate the Model**: Compare the model's performance with traditional crop models or other machine learning methods.

Example command to start training:
```bash
python train.py --data_dir /path/to/data --epochs 100 --batch_size 32
```

## Model Architecture

DeepCGM is a **deep learning-based crop growth model** with a unique architecture designed to respect the biological processes of crop growth. The model includes:

- **Mass-Conserving Layers**: Ensure physically plausible crop growth curves by enforcing mass conservation across crop variables.
- **Knowledge-Guided Constraints**: Use crop growth principles to guide the model training, preventing unrealistic predictions.
- **LSTM/GRU Layers**: Capture temporal dependencies in the growth process.
- **CNN Layers**: Extract spatial patterns from multi-source data.

## Training

The model is trained using sparse datasets, with the following features:

- **Crop Physiology Constraints**: Crop physiology is incorporated directly into the training process.
- **Convergence Constraints**: Help ensure that the model reaches realistic steady-state solutions during training.
- **Efficient Data Usage**: Designed to work with small or incomplete datasets, typical in agricultural experiments.

To start training the model:
```bash
python train.py --config config.json
```

## Evaluation

DeepCGM is evaluated using an observational dataset from a two-year rice experiment involving 105 plots. It is compared against the **ORYZA2000 process-based model** and classical deep learning models.

Metrics such as **normalized mean square error (MSE)** and **R2 score** are used to assess the model's performance:

```bash
python evaluate.py --model_path /path/to/saved_model
```

## Data

The dataset includes:

- **Weather Data**: Temperature, rainfall, and humidity.
- **Soil Data**: Nutrient levels, pH, moisture content.
- **Management Data**: Sowing date, irrigation schedule, etc.

Ensure your data follows the structure specified in `data/README.md`.

## Results

DeepCGM produces **physically plausible crop growth curves** and outperforms both traditional process-based models and classical deep learning models:

- **Normalized MSE Improvement**: 
  - From 0.0381 to 0.0338 (2019)
  - From 0.0473 to 0.0397 (2018)

The results are stored in the `results/` folder.

## Contributing

We welcome contributions! To contribute:

1. Fork this repository.
2. Create a branch (`git checkout -b feature-xyz`).
3. Commit your changes (`git commit -m "Add feature xyz"`).
4. Push to the branch (`git push origin feature-xyz`).
5. Create a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
