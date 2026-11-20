Machine learning classification and clustering project developed for COMP 472 at Concordia University.

## Project Overview

This repository contains implementations and experiments for machine learning algorithms applied to image and sentiment datasets.

The currently documented implementations are:

- **Decision Tree:** A classification algorithm implemented from scratch.
- **Random Forest:** A classifier that combines predictions from several Decision Trees, evaluated on image and sentiment features.
- **Neural Network:** A feedforward neural network implemented using PyTorch.
- **k-Nearest Neighbours (KNN):** A custom distance-based classifier evaluated on image and sentiment features.
- **K-Means:** Unsupervised clustering experiments on sentiment features.
- **KNN Error Analysis:** Per-class evaluation and inspection of misclassified examples.

The project includes experiments for evaluating model performance under different configurations and hyperparameters.

## Repository Structure

```
COMP472-Fall2026-Project/
├── data/                           # provided CSV datasets (not tracked in Git)
├── models/
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── neural_network.py
│   ├── neural_network_training.py
│   ├── main.py
│   ├── knn.py
│   ├── kmeansclustering.py
│   └── erroranalysis.py
├── experiments/
│   ├── decision_tree_experiment.py
│   ├── random_forest_experiment.py
│   ├── neural_network_experiment.py
│   ├── neural_network_sentiment.py
│   └── neural_network_evaluation.py
├── results/
│   ├── clustering/
│   └── erroranalysis_knn/
│       ├── image/
│       └── sentiment/
├── outputs/                        # generated sentiment files (local)
│   └── neural_network_sentiment/
├── best_neural_network_batch_norm.pth
├── requirements.txt
├── .gitignore
└── README.md

```

### File Descriptions

| **File**                                   | **Description**                                                                                    |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `models/decision_tree.py`                  | Decision Tree classifier implementation                                                            |
| `models/random_forest.py` | Random Forest implementation using bootstrap samples, random feature selection, and majority voting |
| `models/neural_network.py`                 | Neural Network architecture                                                                        |
| `models/neural_network_training.py`        | Neural Network training and validation functions                                                   |
| `experiments/decision_tree_experiment.py`  | Decision Tree experiments on image and sentiment datasets                                          |
| `experiments/random_forest_experiment.py` | Runs the selected Random Forest configurations on both datasets automatically |
| `experiments/neural_network_experiment.py` | Neural Network hyperparameter experiments                                                          |
| `experiments/neural_network_sentiment.py` | Sentiment Neural Network training, validation, and test prediction export |
| `experiments/neural_network_evaluation.py` | Evaluation of the saved Neural Network                                                             |
| `models/knn.py`                            | Custom KNN classifier: distance calculations, nearest-neighbour selection, and voting              |
| `models/main.py`                           | KNN dataset preparation, validation experiments, hyperparameter selection, and prediction workflow |
| `models/kmeansclustering.py`               | K-Means sentiment clustering experiments, numerical summaries, and plots                           |
| `models/erroranalysis.py`                  | KNN validation error analysis: per-class metrics, confusion matrices, and misclassified examples   |
| `results/clustering/`                      | Generated clustering result tables and graphs                                                      |
| `results/erroranalysis_knn/`               | Generated KNN error-analysis reports, split into `image/` and `sentiment/`                         |
| `best_neural_network_batch_norm.pth`       | Saved weights of the selected Neural Network model                                                 |
| `requirements.txt`                         | Python dependencies                                                                                |

## Installation

Clone the repository:

```
git clone https://github.com/mobinamoghadam/COMP472-Fall2026-Project.git
cd COMP472-Fall2026-Project
```

Create a Python virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
```

Install the required libraries:

```
pip install -r requirements.txt
```

## Dataset Setup

The datasets are not included in the GitHub repository.

Create a folder named `data` in the project root and place the provided CSV files inside it:

```
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

The KNN implementation uses the provided numerical `feature_` columns. The sample `id` is not used as a predictive feature, and the `label` column is used as the target for supervised evaluation.

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

```
if __name__ == "__main__":
    experiment = 8
```

Change `experiment` to the desired experiment number.

| **Experiment** | **Description**                             |
| -------------- | ------------------------------------------- |
| 1              | Maximum depth                               |
| 2              | Training dataset size                       |
| 3              | Minimum samples required to split           |
| 4              | Minimum samples per leaf                    |
| 5              | Sentiment classification                    |
| 6              | Sentiment maximum depth                     |
| 7              | Sentiment minimum samples per leaf          |
| 8              | Sentiment minimum samples required to split |

Run the selected experiment from the project root:

```
python -m experiments.decision_tree_experiment
```

## Random Forest

`models/random_forest.py` reuses the shared Decision Tree code to train trees on random row samples and feature subsets, then combines their votes.

Run both selected experiments from the project root:

```bash
python -m experiments.random_forest_experiment
```

Images run first, then sentiment. Each run prints training accuracy, validation accuracy, and training time.

Selected settings and previously recorded validation results, using seed `42`:

| Dataset | Trees | Depth | Features per split | Train/validation | Validation accuracy |
|---|---|---|---|---|---|
| Images | 100 | 16 | `"sqrt"` (22) | 85/15 | 20.60% |
| Sentiment | 50 | 8 | 50 | 80/20 | 57.71% |

Both splits are stratified. Compare models on matching validation splits; the same image forest settings achieved **19.55%** on an 80/20 split.

## Neural Network

The Neural Network is implemented using PyTorch.

The relevant files are:

- `models/neural_network.py` — model architecture
- `models/neural_network_training.py` — training and validation
- `experiments/neural_network_experiment.py` — experimental configurations
- `experiments/neural_network_sentiment.py` — sentiment training, validation, and test predictions

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

```
if __name__ == "__main__":
    experiment = 11
```

Select an experiment number from 1 to 11.

Run:

```
python -m experiments.neural_network_experiment
```

The experiments compare different configurations, including learning rates, dropout probabilities, hidden-layer sizes, weight decay, and batch normalization.

Experiment 11 is the selected batch-normalization configuration and saves the model weights as:

```
best_neural_network_batch_norm.pth

```

### Evaluating the Saved Neural Network

The repository includes the trained model checkpoint.

To load and evaluate it, run:

```
python -m experiments.neural_network_evaluation
```

The evaluation script loads the saved model weights, prepares the image validation data, and reports validation accuracy.

### Selected Neural Network Results

| **Metric**               | **Value** |
| ------------------------ | --------- |
| Best validation accuracy | 33.95%    |
| Best epoch               | 30        |
| Training epochs          | 50        |
| Batch size               | 64        |
| Learning rate            | 0.001     |
| Dropout                  | 0.4       |
| Batch normalization      | Enabled   |

### Sentiment Neural Network

`experiments/neural_network_sentiment.py` uses the shared Neural Network model and training functions.

| Setting | Value |
|---|---|
| Network | 768 → 256 → 64 → 3, with ReLU |
| Optimizer / loss | Adam / CrossEntropyLoss |
| Learning rate | 0.0001 |
| Epochs / batch size | 20 / 64 |
| Dropout / batch normalization / weight decay | 0.0 / disabled / 0.0 |
| Training / validation | Stratified 80/20: 9,600 / 2,400 examples |
| Split, initialization, and batch-order seed | 42 |
| Recorded training / validation accuracy | 74.48% / 66.88% |

Train, then export the sentiment test predictions:

```bash
python -m experiments.neural_network_sentiment --train
python -m experiments.neural_network_sentiment --export
```

Training restores and saves the best epoch's weights. Results and weights are saved locally under `outputs/neural_network_sentiment/`; exclude these files from source commits.

Export creates `sentiment_predictions.csv` in that folder, containing 5,000 rows with columns `id,label`. Labels are `0` (negative), `1` (neutral), and `2` (positive).

For Kaggle, combine this CSV with the image KNN predictions by ID using `sample_submission.csv` as the template, producing 10,000 predictions.

## k-Nearest Neighbours (KNN)

The KNN classifier is implemented in `models/knn.py`. It predicts a sample's label by finding the closest labeled training examples and combining their votes. The implementation supports Euclidean or cosine distance, uniform or distance-weighted voting, and batched prediction to reduce memory usage.

`models/main.py` runs the KNN workflow for both image recognition and sentiment analysis. It loads the feature columns, makes stratified 80/20 training and validation splits, evaluates candidate values of `k`, and reports validation accuracy. Depending on the selected setting, features can be left unchanged, standardized, or normalized to unit length (L2). The split uses `RANDOM_SEED = 42` for reproducibility.

### KNN training and validation split

Before evaluating a KNN configuration, `main.py` divides each **labeled training dataset** into two non-overlapping subsets using `train_test_split` from scikit-learn:

| **Dataset**        | **Total labeled examples** | **Training (80%)** | **Validation (20%)** |
| ------------------ | -------------------------- | ------------------ | -------------------- |
| Image recognition  | 20,000                     | 16,000             | 4,000                |
| Sentiment analysis | 12,000                     | 9,600              | 2,400                |

The split uses `test_size=0.2`, `random_state=RANDOM_SEED` (currently `42`), and `stratify=y`:

- **80% training:** KNN stores these feature vectors and their correct labels, then uses them as neighbours when predicting validation samples.
- **20% validation:** These held-out labeled examples are passed to the classifier for prediction. The predicted labels are compared with the actual labels to calculate validation accuracy.
- **Stratification:** `stratify=y` preserves the relative proportions of the classes in both subsets. This is particularly useful for image recognition, which has 50 classes, and sentiment analysis, which has three classes.
- **Reproducibility:** Keeping `RANDOM_SEED = 42` produces the same split on repeated runs, allowing a fairer comparison between different KNN settings.

Preprocessing is applied **after the split**. For standardization, the feature means and standard deviations are learned from the training subset only and then applied to validation features. L2 normalization operates independently on each feature vector. This avoids using validation information to fit preprocessing parameters (data leakage).

For every candidate `k` in `K_VALUES`, the script fits KNN on the same training subset and predicts the same validation subset. It calculates accuracy as the number of correct predictions divided by the number of validation examples. The configuration with the highest validation accuracy is selected. If two `k` values tie, the existing code chooses the larger `k`.

This training/validation split is distinct from the separate unlabeled test CSV files; validation accuracy is calculated using examples whose correct labels are already known.

### Configuring KNN experiments

Open `models/main.py` and adjust the configuration values near the top:

| **Setting**       | **Meaning**                                             |
| ----------------- | ------------------------------------------------------- |
| `K_VALUES`        | Neighbour counts to compare on the validation set       |
| `DISTANCE_METRIC` | `"euclidean"` or `"cosine"`                             |
| `VOTING_METHOD`   | `"uniform"` or `"distance"`                             |
| `PREPROCESSING`   | `"none"`, `"standard"`, or `"l2"`                       |
| `RANDOM_SEED`     | Fixed seed for the validation split                     |
| `BATCH_SIZE`      | Maximum number of prediction samples processed together |

Run from the repository root:

```
python models/main.py
```

`main.py` imports `KNNClassifier` from `knn.py`; `knn.py` is not run as a standalone program. Any standardization parameters are calculated using the training portion, then applied to validation data.

## K-Means Clustering

`models/kmeansclustering.py` investigates whether sentiment examples form meaningful groups based on their 768-dimensional feature vectors. K-Means forms clusters without looking at the true sentiment labels. The labels are used afterward to describe each cluster's composition.

The script experiments with different numbers of clusters and reports inertia (within-cluster squared distances) and silhouette scores. It repeats cluster initialization and retains the best result according to inertia.

### Configuring and running clustering

The settings at the top of `models/kmeansclustering.py` include:

- `K_CLUSTER_COUNT`: cluster counts to test (currently 2–10).
- `MAX_NUM_ITERATIONS`: maximum iterations per K-Means run.
- `TOLERANCE`: convergence threshold.
- `NUM_INITALIZATIONS`: number of initialization attempts (spelling matches the current script).
- `RANDOM_SEED`: reproducibility of the initial centroids.

Run from the project root:

```
python models/kmeansclustering.py
```

The generated files in `results/clustering/` are:

| **Output**                      | **Contents**                                        |
| ------------------------------- | --------------------------------------------------- |
| `kmeans_results.csv`            | Inertia and silhouette results by cluster count     |
| `cluster_label_counts.csv`      | Counts of true sentiment labels within each cluster |
| `cluster_label_percentages.csv` | Percentage composition of each cluster              |
| `elbow_method.png`              | Inertia plotted against cluster count               |
| `silhouette_scores.png`         | Silhouette score plotted against cluster count      |
| `cluster_composition.png`       | Visual breakdown of sentiment labels per cluster    |

## KNN Error Analysis

`models/erroranalysis.py` examines the mistakes made by a selected KNN configuration on held-out validation data. It reuses `read_task()` and `prepare()` from `models/main.py` and the custom classifier in `models/knn.py`.

To analyze a particular experiment, edit these settings at the top of `models/erroranalysis.py`:

```
TASK = "sentiment"  # or "image"
K = 40
DISTANCE_METRIC = "euclidean"
VOTING_METHOD = "distance"
PREPROCESSING = "none"
```

These values are examples from the current script; change them to the configuration being analyzed. Run:

```
python models/erroranalysis.py
```

Reports are written to `results/erroranalysis_knn/image/` or `results/erroranalysis_knn/sentiment/`, depending on `TASK`:

| **Output**                  | **Contents**                                                   |
| --------------------------- | -------------------------------------------------------------- |
| `summary.csv`               | Accuracy, macro F1, configuration, and number of errors        |
| `class_metrics.csv`         | Precision, recall, F1-score, and support for every class       |
| `confusion_matrix.csv`      | Actual-versus-predicted label counts                           |
| `confused_pairs.csv`        | Most frequent misclassification pairs                          |
| `misclassified_samples.csv` | IDs, actual labels, and predicted labels of incorrect examples |

This evaluation uses the labeled validation split. Macro F1 gives each class equal weight when averaging its individual F1-score.

### Paths for the reorganized repository

All four scripts above are intended to reside in `models/`. Before running them after relocation, ensure their dataset and output paths point to the repository root rather than to `models/` itself. For example, `main.py` should use:

```
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
```

The K-Means script should similarly resolve `data/sentiment_train.csv` and `results/clustering/` from the repository root, and the error-analysis script should save to `results/erroranalysis_knn/<task>/`. These path edits are made in the Python scripts; editing the README alone does not change their behavior.

## Notes

- Run all commands from the repository's root directory.
- Ensure the datasets are placed inside `data/` before running the experiments.
- The experiment scripts select configurations using the `experiment` variable.
- The added Random Forest script instead runs both datasets automatically using the selected settings.
- The sentiment Neural Network script uses `--train` for training and `--export` for test predictions.
- Running training experiments may overwrite existing model checkpoint files.
- Keep generated clustering and error-analysis reports under `results/` to separate experiment outputs from source code.
- Keep generated sentiment Neural Network outputs under `outputs/neural_network_sentiment/` and exclude them from source commits.
