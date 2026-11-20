"""
this script runs our k-nearest neighbours (knn) classifier on both datasets.

we have two classification tasks:
1. image recognition: classify images into one of 50 categories.
2. sentiment analysis: classify text as negative, neutral, or positive.

the datasets already contain numerical features, hence the main purpose of this script is to:
1. load both datasets from csv files.
2. split the labeled data into training and validation sets.
3. test our knn algorithm and calculate its validation accuracy.
4. select the best value of k based on the validation results.
5. use all available labeled data to prepare our final classifier.
6. predict the labels of the kaggle test samples.
7. generate a submission.csv file containing all 10,000 predictions.

to run this script, we execute: python main.py
"""

# pathlib provides the path class, which lets us work with file paths.
# we use it so we don't have to manually write the full directory path every time we want to access a dataset.
from pathlib import Path

#feature vectors and labels stored as numpy arrays.
import numpy as np
import pandas as pd

# train_test_split lets us divide our labeled dataset into training and validation subsets. this is allowed because it is a data preparation utility and we checked with Eeham to make sure this was okay.
from sklearn.model_selection import train_test_split

# standardscaler is used to standardize numerical features. normalize is used to scale individual feature vectors to unit length.
# both are optional preprocessing methods that we experimented with.
from sklearn.preprocessing import StandardScaler, normalize

# we import our own knn implementation from the knn.py file.
from knn import KNNClassifier

# experiment settings
# these variables control how our knn experiments will run. keeping them at the top makes it easier to change hyperparameters without modifying the rest of the program.
# k represents the number of nearest neighbours considered when predicting the class of a sample.
# we currently test only k=X, but we can test multiple values by writing:
# k_values = [1, 3, 5, 6, 7, 10, 20, 30, 40, 50, 70, 80, 100]
# the program will evaluate each value and select the one with the highest validation accuracy.
K_VALUES = [40, 70, 80, 90, 100, 120]

# DISTANCE_METRIC determines how we measure the distance between two feature vectors.
# euclidean: measures the straight-line distance between two points in a multidimensional space.
# cosine: measures how different the directions of two vectors are, rather than directly comparing their absolute positions.
# the exact distance calculations are implemented inside knn.py.
DISTANCE_METRIC = "euclidean"

# VOTING_METHOD determines how the neighbours influence the final prediction. uniform: every neighbour gets one equal vote, regardless of distance. distance: closer neighbours receive more influence than farther ones.
VOTING_METHOD = "distance"
# preprocessing determines whether we modify the feature values before passing them to the knn classifier.
# none: use the original feature vectors without changing them.
# standard: standardize each feature based on the training data's mean and standard deviation.
# l2: divide each feature vector by its euclidean length so that its total length becomes 1 (unless it is a zero vector), essentially making it a unit vector.
# preprocessing matters for knn because distances can change depending on the scale and representation of the features, this is explored in our experimentation.
PREPROCESSING = "none"

# random_seed controls the randomization used when splitting our data.
# using the same seed ensures that we get the same training and validation split each time we run the program.
# this is important because we want to compare different algorithms and hyperparameters using consistent validation data.
RANDOM_SEED = 42

# batch_size controls how many samples the knn classifier processes at once when making predictions.
# for up to 70 samples at a time instead of processing the entire validation or test dataset in one large batch.
# this helps control memory usage because knn needs to calculate
# distances between the samples being predicted and the training samples.
BATCH_SIZE = 70

# this allows the script to work even if our teammates have the project saved in different locations on their computers.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# function 1: loading the trining and test csv for one classification task, we can specify for which one we want.
def read_task(task):
    """
    task can be either "image" or "sentiment".

    this function returns:
    x: numerical features from the labeled training dataset.
    y: correct labels corresponding to those training examples.
    x_test: numerical features from the unlabeled kaggle test dataset.
    test_ids: unique ids for the kaggle test examples.

    """

    # loads the csv into a df
    train = pd.read_csv(DATA_DIR / f"{task}_train.csv")
    test = pd.read_csv(DATA_DIR / f"{task}_test.csv")

    # the datasets contain three types of columns:
    # id: identifies each sample. feature_0, feature_1, etc.: numerical values describing the sample. label: the correct class, available only in the training data.
    
    # we only want the feature columns when training the classifier. this list comprehension goes through the column names in train and keeps only those starting with "feature_".
    # for images, there should be 500 feature columns.
    # for sentiment, there should be 768 feature columns.

    feature_cols = [col for col in train if col.startswith("feature_")]

    # here, we select only the numerical feature columns from the training dataframe.
    # to_numpy converts the pandas dataframe into a numpy array.
    # x has one row per training sample and one column per feature, x will be used when we get a new example and we want to calculate the training distances to it. 
    X = train[feature_cols].to_numpy(dtype=np.float32)

    # next we extract the correct class labels from the training dataset.
    # image labels range from 0 to 49.
    # sentiment labels range from 0 to 2.
    # y[i] contains the correct label for the example x[i].
    y = train["label"].to_numpy(dtype=np.int64)

    # extract the numerical features from the kaggle test dataset.  
    # now we use the same feature_cols list so that the features appear in the same order as in the training dataset.
    X_test = test[feature_cols].to_numpy(dtype=np.float32)

    # return the training features, training labels, test features, and the unique ids associated with the test samples.
    # we keep the ids because kaggle submission requires every prediction to be associated with the correct sample id.
    return X, y, X_test, test["id"]

# function 2: preprocessing feature vectors
# if a method was selected for preprocessing, apply it to the two datasets. 
def prepare(train_features, other_features):
    """
    train_features contains the data used to train the classifier.
    other_features contains validation or kaggle test data when generating the submission csv
    it is important that any preprocessing parameters are learned only from the training features to avoid data leakage.
    """

    # if preprocessing is set to none, return the original arrays. no normalization or standardization is performed.
    if PREPROCESSING == "none":
        return train_features, other_features

    # l2 normalization scales each individual feature vector so that its euclidean distance becomes 1, if it is nonzero. its basically the process of converting to a unit vector
    # we normalize the training and other features separately because l2 normalization is calculated independently for each row, since they each have their own magnitude so we need to make them into unit vectors row by row.
    if PREPROCESSING == "l2":
        return normalize(train_features).astype(np.float32), normalize(other_features).astype(np.float32)

    # standardization adjusts features based on their mean and standard deviation.
    # for a feature value x, the transformation is:
    # standardized value = (x - mean) / standard deviation
    # after standardization, a feature typically has a mean near 0 and standard deviation near 1 in the training dataset.
    if PREPROCESSING == "standard":
        # create a standardscaler object that will calculate the mean and standard deviation of each feature.
        scaler = StandardScaler()

        # fit_transform first learns the mean and standard dev from train_features, then applies the transformation.
        # transform applies those same learned values to other_features, we do not call fit_transform on the validation data because that would allow information from validation example to affect preprocessing and make the evaluation less reliable.
        return (scaler.fit_transform(train_features).astype(np.float32),
                scaler.transform(other_features).astype(np.float32))

    # if preprocessing is not one of the supported options,
    # raise an error instead of silently using an unexpected method.
    raise ValueError("Unknown preprocessing setting")


# function 3: train, validate, and predict for one task
def run_task(task):
    """
    this is essentially what runs the entire knn workflow for either the image or sentiment task.

    the function:
    1. loads the data.
    2. creates training and validation subsets.
    3. evaluates the selected values of k.
    4. chooses the best k.
    5. fits the final classifier using all available labeled data.
    6. predicts the labels for the hidden kaggle test samples.

    the function returns a df containing ids and predictions.
    """

    # prints the task name so that we know which dataset is being processed, flush=true tells python to immediately display the output, instead of waiting for the output buffer to be flushed.
    print(f"\n- {task.upper()} -", flush=True)

    # call read_task to load the correct csv files. python automatically assigns each returned value to its corresponding variable.
    X, y, X_test, test_ids = read_task(task)

    # split the labeled examples into training and validation subsets. test_size=0.2 means 20% of the labeled examples are reserved for validation, while the other 80% are used for training.
    # random_state uses our fixed random seed so the split remains reproducible between executions.
    # stratify=y makes sure that the proportion of samples from each class is preserved in the two subsets.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # now we apply our chosen preprocessing method.
    # x_train is used to determine any preprocessing parameters,
    # and the same transformation is applied to x_val.
    X_train, X_validation = prepare(X_train, X_validation)

    # here, we store the validation results.
    # each result will contains a pair: (validation accuracy, value of k).
    validation_scores = []

    # then we have a loop to go through every k that we specified in the input earlier. 
    for k in K_VALUES:

        # create a new instance of our custom knn classifier.
        # for each run of the loop, we create a new instance of knnclassifier, the actual knn calculations happen inside knn.py.
        model = KNNClassifier(k=k, metric=DISTANCE_METRIC, weights=VOTING_METHOD, batch_size=BATCH_SIZE)

        # fit provides the classifier with the training examples and their correct labels. it just stores the features and labels so that they can be used later. 
        model.fit(X_train, y_train)

        # use the classifier to predict labels for every example in the validation dataset.
        validation_prediction = model.predict(X_validation)

        # then we compare every predicted label against its correct label.
        # validation_prediction == y_validation creates a boolean array. numpy treats true as 1 and false as 0 when calculating the mean. therefore, np.mean gives the proportion of correct predictions.
        # it essentially calculates correct predictions / all predictions.
        accuracy = np.mean(validation_prediction == y_validation)

        # weave the accuracy and its corresponding k value.
        # we store accuracy first because max() will compare the first element of each tuple before considering the second, and we want to take the one with the best accuracy.
        validation_scores.append((accuracy, k))

        # print the validation accuracy for the current value of k.
        print(f"k={k}: validation accuracy = {accuracy:.2%}", flush=True)

    # max(validation_scores) finds the tuple with the highest accuracy. because tuples are compared element by element, the value of k is used as a tie-breaker if two accuracies match. 
    # we choose the k with a larger value, as it represents a larger sample rather than smaller values which are more prone to noise.
    # the two returned values are assigned to best_accuracy and best_k.
    best_accuracy, best_k = max(validation_scores)

    # displaying
    print(f"Selected k={best_k} (validation accuracy {best_accuracy:.2%})", flush=True)

    # we have now finished using the validation split to select k.
    # once we have selected k, we can use all the labelled examples to train our model, and then we can upload our submission csv to kaggle where it will be tested.
    # prepare is called again using the full labeled dataset x, 
    
    # if standardization is enabled, this learns the preprocessing parameters from all the available labeled examples and then will be tested on kaggle.
    X_full, X_test = prepare(X, X_test)

    # then we create the final knn classifier with the selected k, as we did when we were testing / validating earlier.
    final_model = KNNClassifier(k=best_k, metric=DISTANCE_METRIC, weights=VOTING_METHOD, batch_size=BATCH_SIZE)

    # fit the final classifier using the complete labeled training dataset.
    final_model.fit(X_full, y)

    # generate predictions for the kaggle test examples.
    final_predictions = final_model.predict(X_test)

    # create a pandas df containing two columns:
    # id: the unique identifier of each kaggle test sample.
    # label: the class predicted by our knn classifier.
    # these columns are needed for the final kaggle submission.
    return pd.DataFrame({"id": test_ids, "label": final_predictions})



# function 4: run both tasks and generate the submission file
def main():
    

    # run the entire knn workflow for image recognition.
    image_results = run_task("image")

    # repeat the same workflow for sentiment analysis.
    sentiment_results = run_task("sentiment")

    # concat combines the two dataframes into one larger dataframe.
    # ignore_index=true creates a new continuous row index rather than preserving the original dataframe indices.
    # the combined dataframe now contains 10,000 predictions.
    predictions = pd.concat([image_results, sentiment_results], ignore_index=True)

    # load kaggle's sample_submission.csv file.
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    # verify that there are no duplicate ids in either dataframe
    # duplicate ids would make it unclear which prediction belongs to which sample.assert stops the program if the condition is false. and we want all ids to be unique.
    assert sample["id"].is_unique and predictions["id"].is_unique

    # merge combines the sample submission ids with our predictions. on="id" means that rows are matched using their id values.
    # how="left" keeps all rows from the sample submission and attaches the corresponding predicted label.
    # validate="one_to_one" checks that each id appears at most
    # once in each dataframe.
    # this is important because the test files and the sample submission may not have their rows in the same order.
    # matching by id prevents predictions from being assigned to the wrong samples.
    submission = sample[["id"]].merge(predictions, on="id", how="left", validate="one_to_one")

    # verify that the final submission contains exactly 10,000 rows.
    # the project has 5,000 image test examples and 5,000 sentiment
    # test examples, so any other row count would indicate a problem.
    assert len(submission) == 10000

    # notna checks whether each predicted label is not missing.
    # all() verifies that every row has a valid non-missing value.
    assert submission["label"].notna().all()

    # convert all predicted labels to integers.
    submission["label"] = submission["label"].astype(int)

    # create a boolean mask identifying all image samples.
    # startswith("img_") returns true for ids beginning with img_.
    img_rows = submission["id"].str.startswith("img_")

    # create another boolean mask identifying sentiment samples.

    # sentiment test ids begin with text_.
    text_rows = submission["id"].str.startswith("text_")

    # verify that the submission contains exactly:
    # 5,000 image predictions and 5,000 sentiment predictions.
    # summing a boolean mask counts the number of true values.
    assert img_rows.sum() == 5000 and text_rows.sum() == 5000

    # verify that all image predictions are valid class labels.
    # all() confirms that every image label satisfies this condition.
    assert submission.loc[img_rows, "label"].between(0, 49).all()

    # verify that all sentiment predictions are between 0 and 2.
    # 0 represents negative sentiment.
    # 1 represents neutral sentiment.
    # 2 represents positive sentiment.
    assert submission.loc[text_rows, "label"].between(0, 2).all()

    # create the path where the final submission file will be saved.
    output = Path(__file__).resolve().parent.parent / "results" / "knnsubmission.csv"

    # index=false prevents pandas from adding an extra index column, since kaggle requires exactly 2 cols.
    submission.to_csv(output, index=False)

    # print the file location and number of predictions
    # so we can confirm that the script completed successfully.
    print(f"\nSaved {output} with {len(submission)} predictions.")

    # also save the image predictions separately, without changing the complete submission this is because knn gave best img accuracy of them all.
    # image_results already contains the 5,000 image ids and their predicted labels, matching by id preserves the image order from sample_submission.csv.
    image_submission = sample.loc[img_rows, ["id"]].merge(
        image_results, on="id", how="left", validate="one_to_one"
    )

    # check that every image has a valid predicted class.
    assert len(image_submission) == 5000
    assert image_submission["label"].notna().all()
    image_submission["label"] = image_submission["label"].astype(int)
    assert image_submission["label"].between(0, 49).all()

    # save the image-only csv alongside submission.csv.
    image_output = Path(__file__).resolve().parent.parent / "results" / "knnimage_submission.csv"
    image_submission.to_csv(image_output, index=False)
    print(f"Saved {image_output} with {len(image_submission)} image predictions.")


# entry point
if __name__ == "__main__":
    main()
