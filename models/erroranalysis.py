"""
this file runs error analysis for one knn task at a time.

we first enter the best configuration that we already found during our experiments. then we load and split the data, train our own knn classifier,
and compare its validation predictions with the correct labels.

we reuse the loading and preprocessing functions in main.py and our knn
classifier in knn.py. 

the evaluation metrics come from scikit-learn because the assignment does not require us to implement precision, recall, or f1 from scratch.

to run this file, use: python erroranalysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# we import main so we can reuse the functions within it like read_task() and prepare()
import main
from knn import KNNClassifier


# experiment settings
# first, choose which task we want to evaluate. use "sentiment" or "image".
# then enter the best configs we already found for that specific task across the experiments
# to evaluate the other dataset later, we change these values and run again.
TASK = "sentiment"
K = 40
DISTANCE_METRIC = "euclidean"
VOTING_METHOD = "distance"
PREPROCESSING = "none"

# use the same split seed, validation proportion, and batch size as main.py.
# this is important because we want to evaluate our model on the same validation examples that we used during our earlier experiments.
RANDOM_SEED = main.RANDOM_SEED
BATCH_SIZE = main.BATCH_SIZE


def evaluate_knn(task, y_validation, validation_prediction, validation_ids, output_dir):
    """save the metrics and mistakes for the selected knn model."""

    # first, create the output folder if it does not already exist. parents=true also creates any missing parent folders, while exist_ok=true means we can run the program again without an error.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # then we list every possible class, including any class that the model might not predict at all. this keeps the reports consistent.
    # the sentiment labels are 0=negative, 1=neutral, and 2=positive.
    labels = list(range(50)) if task == "image" else [0, 1, 2]
    class_names = [str(label) for label in labels] if task == "image" else ["negative", "neutral", "positive"]

    # accuracy tells us what fraction of all validation predictions is correct.
    # macro f1 first calculates the f1-score for each class, then takes their average, so each class receives equal importance 
    # this is important because if the number of samples is diff for all classes, they still contribute equally to the f1 score overall. 
    accuracy = accuracy_score(y_validation, validation_prediction)
    macro_f1 = f1_score(y_validation, validation_prediction, labels=labels, average="macro", zero_division=0)

    # next, classification_report calculates precision, recall, and f1-score separately for every class. 
    # support means how many actual validation samples belong to that class, regardless of what the model predicted.
    # output_dict=true gives us the values in a form that pandas can save.
    report = classification_report(
        y_validation,
        validation_prediction,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # now we keep only the individual class rows in the per-class csv, this makes it easy to identify which classes have low recall or f1.
    class_metrics = pd.DataFrame(report).transpose().loc[class_names]
    class_metrics.index.name = "class"
    class_metrics.to_csv(output_dir / "class_metrics.csv")

    # then we create the confusion matrix. rows represent the actual labels, and columns represent the predicted labels. values on the diagonal are correct predictions, while values outside the diagonal are mistakes.
    matrix = confusion_matrix(y_validation, validation_prediction, labels=labels)
    confusion = pd.DataFrame(matrix, index=class_names, columns=class_names)
    confusion.index.name = "actual_class"
    confusion.columns.name = "predicted_class"
    confusion.to_csv(output_dir / "confusion_matrix.csv")

    # now we save each sample where the prediction does not match its actual label. because we kept the ids when splitting the dataset, we can identify these samples without including their hundreds of feature values.
    mistakes_mask = np.asarray(y_validation) != np.asarray(validation_prediction)
    mistakes = pd.DataFrame({
        "id": np.asarray(validation_ids)[mistakes_mask],
        "actual_label": np.asarray(y_validation)[mistakes_mask],
        "predicted_label": np.asarray(validation_prediction)[mistakes_mask],
    })
    mistakes.to_csv(output_dir / "misclassified_samples.csv", index=False)

    # finally, we count the most frequent actual-to-predicted mistakes. for example, actual=1 and predicted=2 means a neutral sample was incorrectly classified as positive. these pairs are directional:
    # actual=1, predicted=2 is different from actual=2, predicted=1.
    if len(mistakes) > 0:
        confused_pairs = (
            mistakes.groupby(["actual_label", "predicted_label"])
            .size()
            .reset_index(name="count")
            .sort_values(["count", "actual_label", "predicted_label"], ascending=[False, True, True])
        )
    else:
        confused_pairs = pd.DataFrame(columns=["actual_label", "predicted_label", "count"])
    confused_pairs.to_csv(output_dir / "confused_pairs.csv", index=False)

    # we also save a short summary, so we can compare this model against the best configurations of the other algorithms later in our report.
    pd.DataFrame([{
        "task": task,
        "k": K,
        "distance_metric": DISTANCE_METRIC,
        "voting_method": VOTING_METHOD,
        "preprocessing": PREPROCESSING,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "validation_samples": len(y_validation),
        "incorrect_predictions": len(mistakes),
    }]).to_csv(output_dir / "summary.csv", index=False)

    # then we display the main numbers and the most common mistakes in the
    # terminal as a quick overview. the full results are in the csv files.
    print(f"\n{task} error analysis: accuracy={accuracy:.2%}, macro f1={macro_f1:.4f}", flush=True)
    print(f"incorrect predictions: {len(mistakes)} / {len(y_validation)}", flush=True)
    print("most frequent class confusions:", flush=True)
    print(confused_pairs.head(10).to_string(index=False), flush=True)
    print(f"saved evaluation files in {output_dir}", flush=True)


def run_evaluation():
    """run the selected knn configuration and examine its validation errors."""

    # first, make sure the task and hyperparameters are valid, so typing a wrong setting does not silently produce an unexpected experiment.
    if TASK not in ("image", "sentiment"):
        raise ValueError("task must be 'image' or 'sentiment'")
    if PREPROCESSING not in ("none", "l2", "standard"):
        raise ValueError("preprocessing must be 'none', 'l2', or 'standard'")

    # now load the features and labels using the same function as main.py.
    # we only need the labeled training data here, since the kaggle test data does not contain correct labels that we could compare predictions with.
    X, y, _, _, *_ = main.read_task(TASK)

    # next, read just the id column from the original labeled csv. these ids are not knn features, but they will help us identify the specific validation samples that were predicted incorrectly.
    train_ids = pd.read_csv(main.DATA_DIR / f"{TASK}_train.csv", usecols=["id"])["id"].to_numpy()

    # then split the features, correct labels, and ids together.
    # this keeps every validation sample aligned with its actual label and id.
    # the split settings match main.py, so the same rows go into validation.
    X_train, X_validation, y_train, y_validation, ids_train, ids_validation = train_test_split(
        X, y, train_ids, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # our original prepare() function in main.py reads its preprocessing
    # setting from main.PREPROCESSING. so we set that value to the method
    # chosen above, then reuse the function instead of copying its logic.
    # this only changes the setting in this running python process; it does
    # not edit main.py or affect a later run of python main.py.
    main.PREPROCESSING = PREPROCESSING
    X_train, X_validation = main.prepare(X_train, X_validation)

    # now create knn using the exact best configurations we selected.
    # fit() stores the training samples, and predict() uses their nearest neighbours to assign a label to each validation sample.
    model = KNNClassifier(k=K, metric=DISTANCE_METRIC, weights=VOTING_METHOD, batch_size=BATCH_SIZE)
    model.fit(X_train, y_train)
    validation_prediction = model.predict(X_validation)

    # finally, compare the predictions with the correct labels and save the reports. image and sentiment have separate folders, so running the script again for the other task will not overwrite this task's results.
    evaluate_knn(
        task=TASK,
        y_validation=y_validation,
        validation_prediction=validation_prediction,
        validation_ids=ids_validation,
        output_dir=Path(__file__).resolve().parent.parent / "results" / "erroranalysis_knn" / TASK,
    )

# this means run_evaluation() starts only when we run evaluation.py directly.
if __name__ == "__main__":
    run_evaluation()



