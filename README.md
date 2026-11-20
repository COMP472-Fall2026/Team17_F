# COMP 472 – Fall 2026 Project

Machine learning classification and clustering project developed for COMP 472 at Concordia University.

## Project Overview

This repository contains implementations and experiments for machine learning algorithms applied to image and sentiment datasets.

The currently documented implementations are:

- **Decision Tree:** A classification algorithm implemented from scratch.
- **Neural Network:** A feedforward neural network implemented using PyTorch.

The project includes experiments for evaluating model performance under different configurations and hyperparameters.

## Repository Structure

```text
COMP472-Fall2026-Project/
├── models/
│   ├── decision_tree.py
│   ├── neural_network.py
│   └── neural_network_training.py
├── experiments/
│   ├── decision_tree_experiment.py
│   ├── neural_network_experiment.py
│   └── neural_network_evaluation.py
├── best_neural_network_batch_norm.pth
├── requirements.txt
├── .gitignore
└── README.md
```

### File Descriptions

| File | Description |
|---|---|
| `models/decision_tree.py` | Decision Tree classifier implementation |
| `models/neural_network.py` | Neural Network architecture |
| `models/neural_network_training.py` | Neural Network training and validation functions |
| `experiments/decision_tree_experiment.py` | Decision Tree experiments on image and sentiment datasets |
| `experiments/neural_network_experiment.py` | Neural Network hyperparameter experiments |
| `experiments/neural_network_evaluation.py` | Evaluation of the saved Neural Network |
| `best_neural_network_batch_norm.pth` | Saved weights of the selected Neural Network model |
| `requirements.txt` | Python dependencies |

## Installation

Clone the repository:

```bash
git clone https://github.com/mobinamoghadam/COMP472-Fall2026-Project.git
cd COMP472-Fall2026-Project
```

Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Dataset Setup

The datasets are not included in the GitHub repository.

Create a folder named `data` in the project root and place the provided CSV files inside it:

```text
data/
├── image_train.csv
├── image_test.csv
├── sentiment_train.csv
├── sentiment_test.csv
└── sample_submission.csv
```

The experiment scripts read the training datasets directly from this folder.

### Data Preprocessing

The classification experiments prepare features and labels from the CSV datasets.

For the Neural Network, the image dataset is divided into training and validation subsets using an 80/20 split.

The resulting subsets contain:

- Training samples: 16,000
- Validation samples: 4,000
- Input features: 500
- Output classes: 50

The Neural Network uses PyTorch tensors and data loaders to process the training and validation samples.

## Decision Tree

The Decision Tree classifier is implemented in:

`models/decision_tree.py`

Its experiments are defined in:

`experiments/decision_tree_experiment.py`

### Running Decision Tree Experiments

Open `experiments/decision_tree_experiment.py` and locate:

```python
if __name__ == "__main__":
    experiment = 8
```

Change `experiment` to the desired experiment number.

| Experiment | Description |
|---|---|
| 1 | Maximum depth |
| 2 | Training dataset size |
| 3 | Minimum samples required to split |
| 4 | Minimum samples per leaf |
| 5 | Sentiment classification |
| 6 | Sentiment maximum depth |
| 7 | Sentiment minimum samples per leaf |
| 8 | Sentiment minimum samples required to split |

Run the selected experiment from the project root:

```bash
python -m experiments.decision_tree_experiment
```

## Neural Network

The Neural Network is implemented using PyTorch.

The relevant files are:

- `models/neural_network.py` — model architecture
- `models/neural_network_training.py` — training and validation
- `experiments/neural_network_experiment.py` — experimental configurations

### Model Architecture

The selected Neural Network uses:

- Input layer: 500 features
- Hidden layers: 256 and 64 neurons
- Output layer: 50 classes
- ReLU activation
- Dropout: 0.4
- Batch normalization

The model is trained using the Adam optimizer.

### Running Neural Network Experiments

Open `experiments/neural_network_experiment.py` and locate:

```python
if __name__ == "__main__":
    experiment = 11
```

Select an experiment number from 1 to 11.

Run:

```bash
python -m experiments.neural_network_experiment
```

The experiments compare different configurations, including learning rates, dropout probabilities, hidden-layer sizes, weight decay, and batch normalization.

Experiment 11 is the selected batch-normalization configuration and saves the model weights as:

```text
best_neural_network_batch_norm.pth
```

### Evaluating the Saved Neural Network

The repository includes the trained model checkpoint.

To load and evaluate it, run:

```bash
python -m experiments.neural_network_evaluation
```

The evaluation script loads the saved model weights, prepares the image validation data, and reports validation accuracy.

### Selected Neural Network Results

| Metric | Value |
|---|---|
| Best validation accuracy | 33.95% |
| Best epoch | 30 |
| Training epochs | 50 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Dropout | 0.4 |
| Batch normalization | Enabled |

## Notes

- Run all commands from the repository's root directory.
- Ensure the datasets are placed inside `data/` before running the experiments.
- The experiment scripts select configurations using the `experiment` variable.
- Running training experiments may overwrite existing model checkpoint files.
- Additional classification and clustering implementations may be added as the project develops.
