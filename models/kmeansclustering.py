import os
from pathlib import Path

# importing these for the visualizations and the graphs and metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

# what k-means does
# 1. we start by choosing k centers (called centroids).
# 2. we assign each sample to the center closest to it.
# 3. we move each center to the average position of the samples assigned to it.
# 4. we repeat these steps until the centers move very little.


# here we set up the dataset path and the settings for our clustering experiment.
# we are using the sentiment feature vectors, but the actual sentiment labels are not given to k-means while it is forming the clusters since it is unsupervised
DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "sentiment_train.csv"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "clustering"


# this allows us to test specific values within a range, we decided to explore 2 clusters to 10 end not inclusive
K_CLUSTER_COUNT = range(2, 11)

# maximum number of iterations per each k-means run, this is basically to ensure theres no infinite loop, we stop early if the centers barely move.
MAX_NUM_ITERATIONS = 100

# we stop when cluster centers move less than this threshold. # 1e-4 means 0.0001. it is a small distance in standardized feature space.
TOLERANCE = 1e-4


# now, we run k-means 5 diff times with different initial centers, we keep the run with the lowest inertia.
NUM_INITALIZATIONS = 5

# picking random seed to keep same random choices when rerunning
RANDOM_SEED = 42

def calculate_squared_distances(X, centroids):
    # this function calculates the squared euclidean distance between every sample and every centroid. x has one row per sample, while centroids has one row per cluster center.
    # instead of comparing each sample to each centroid in a python loop, we use numpy to calculate all the distances at once.
    # the formula is: squared distance = x squared + c squared - 2(x dot c).
    sample_lengths = np.sum(X ** 2, axis=1)[:, None]
    # x ** 2 squares every feature value in each sample, axis=1 adds across the features, giving one squared length per sample. [:, none] changes the result into a column, with shape (number of samples, 1).

    centroid_lengths = np.sum(centroids ** 2, axis=1)[None, :]
    # here we repeat that calculation for every centroid.
    # [none, :] changes the result into a row, with shape (1, number of clusters), these different shapes let numpy use broadcasting for all sample-center pairs making it more efficient.


    # @ performs matrix multiplication and .t transposes the centroid matrix.
    # the result contains one squared distance for every sample and centroid.
    distances = sample_lengths + centroid_lengths - 2 * (X @ centroids.T)
    

    # rounding errors can sometimes create tiny negative values, even though squared distances cannot be negative, so we replace those values with zero to be sure
    return np.maximum(distances, 0)


# this function updates every centroid after we assign samples to clusters.
# a centroid is the average position of all samples assigned to its cluster.
def update_centroids(X, cluster_labels, centroids, random_generator):
    
    # copy() lets us calculate new centers without changing the old ones yet.
    new_centroids = centroids.copy()
    
    # we look at one cluster at a time, from cluster 0 to cluster k-1.
    for cluster_id in range(len(centroids)):
       
        samples_in_cluster = X[cluster_labels == cluster_id]
        # this keeps only the sample rows assigned to the current cluster.

        if len(samples_in_cluster) > 0:
            new_centroids[cluster_id] = np.mean(samples_in_cluster, axis=0)
            # axis=0 averages down the rows, separately for each feature.
            # this gives us the new centroid for the current cluster.
        else:
            # sometimes no samples are assigned to a cluster, since we cannot calculate an average of zero samples,
            # we move this centroid to a randomly selected sample instead.
            random_index = random_generator.integers(len(X))
            new_centroids[cluster_id] = X[random_index]

    return new_centroids


# this is our main k-means algorithm. it receives the feature matrix, the number of clusters, and the settings controlling when we stop.
def kmeans(X, k, max_num_iterations=100, tolerance=1e-4,
           num_initializations=5, random_seed=42):
    random_generator = np.random.default_rng(random_seed)

    # this creates the random generator using our chosen seed.
    best_inertia = float("inf")
    best_cluster_labels = None
    best_centroids = None
    # we start the best inertia at infinity because any finite result is smaller.
    # setting to default none as we look for cluster centers, this will be updated with the algo


    # k-means can find different results depending on its starting centers.
    # we therefore restart it several times and keep the run with the lowest inertia.
    for initialization in range(num_initializations):
        random_indices = random_generator.choice(len(X), size=k, replace=False)

        # we randomly choose k different sample indices to use as our starting centers, replace=false prevents the same sample from being chosen twice in one start.
        centroids = X[random_indices].copy()
        # these selected sample rows become our initial centroids.

        for iteration in range(max_num_iterations):

            # calculate distances and assign each sample to its closest centroid.
            squared_distances = calculate_squared_distances(X, centroids)
            cluster_labels = np.argmin(squared_distances, axis=1)
            # argmin returns the index of the smallest distance in each row, that index is the cluster number assigned to the sample.

            # then we calculate the avg position of the samples in each cluster.
            new_centroids = update_centroids(X, cluster_labels, centroids, random_generator)

            # then we measure how much the centers moved after the update.
            centroid_movement = np.linalg.norm(new_centroids - centroids, axis=1)
            
            # norm then calculates the euclidean distance between each old and new centroid.
            centroids = new_centroids
            if np.all(centroid_movement < tolerance):
                # if all centers moved less than our tolerance, we stop iterating.
                break

        # the last centroid update may have changed which center is closest. so we recalculate the assignments using the final centers.
        final_distances = calculate_squared_distances(X, centroids)
        final_cluster_labels = np.argmin(final_distances, axis=1)

        # inertia adds the squared distance from each sample to its own centroid, arange gives us every sample row, and final_cluster_labels selects the column containing that sample's assigned centroid.
        inertia = np.sum(final_distances[np.arange(len(X)), final_cluster_labels])

        if inertia < best_inertia:

            # we only keep this initialization if it improved the lowest inertia. we save the matching labels and centroids from the same run.
            best_inertia = inertia
            best_cluster_labels = final_cluster_labels.copy()
            best_centroids = centroids.copy()

    return best_cluster_labels, best_centroids, best_inertia

# this function loads our data, runs the experiments, and saves the results.
def main():

    # this creates the results folder if it does not already exist.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = pd.read_csv(DATASET_PATH)
    feature_cols = [col for col in dataset.columns if col.startswith("feature_")] # we only select numerical feature columns, not the id or true sentiment label.
    

    X = dataset[feature_cols].to_numpy(dtype=np.float64)
    true_labels = dataset["label"].to_numpy()
    # x contains the feature values used by k-means, true_labels contains the known answers, which we save only for later analysis since this is unsupervised.

    # we standardize each feature so that different feature scales do not unfairly affect the euclidean distances used by k-means.
    feature_means = np.mean(X, axis=0)
    feature_std = np.std(X, axis=0)
    feature_std[feature_std == 0] = 1

    # a constant feature has standard deviation zero, so we replace it with one to avoid dividing by zero when we standardize the data.
    X_scaled = (X - feature_means) / feature_std # we subtract each feature's mean and divide by its standard deviation.
    results = []
    labels_for_k3 = None
    # results stores the metrics for each k, while labels_for_k3 saves the cluster assignments for k=3 so we can analyze them after the experiments.


    # we then repeat the full k-means experiment for each number of clusters.
    for k in K_CLUSTER_COUNT:
        cluster_labels, centroids, inertia = kmeans(
            X_scaled, k,
            max_num_iterations=MAX_NUM_ITERATIONS,
            tolerance=TOLERANCE,
            num_initializations=NUM_INITALIZATIONS,
            random_seed=RANDOM_SEED,
        )

        # silhouette compares how close a sample is to its own cluster versus the nearest different cluster. higher scores mean clearer group seperation.
        # we use 2,000 samples to make this evaluation faster.
        silhouette = silhouette_score(
            X_scaled, cluster_labels, sample_size=2000, random_state=RANDOM_SEED
        )
        results.append({"k": k, "inertia": inertia, "silhouette_score": silhouette})
        
        # we save k, inertia, and silhouette together so they can form one table.
        print(f"k={k} | inertia={inertia:.2f} | silhouette={silhouette:.4f}")

        if k == 3:
            labels_for_k3 = cluster_labels.copy()
            # k=3 is especially useful because the dataset has three sentiment categories.

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/kmeans_results.csv", index=False)

    # the elbow graph shows how inertia decreases as we increase k.
    # we look for a point where the improvement starts slowing down.
    plt.plot(results_df["k"], results_df["inertia"], marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method - Sentiment Dataset")
    plt.xticks(list(K_CLUSTER_COUNT))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/elbow_method.png", dpi=300)
    plt.close()

    # this second graph compares how well separated the clusters are for each k. silhouette score graph.
    plt.plot(results_df["k"], results_df["silhouette_score"], marker="o", color="green")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs K")
    plt.xticks(list(K_CLUSTER_COUNT))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/silhouette_scores.png", dpi=300)
    plt.close()

    # we now compare the three discovered clusters with the actual sentiment labels.
    # this is only evaluation: the true labels were never used to form the clusters.
     # the table counts how many negative, neutral, and positive samples ended up in each discovered cluster.
    counts = pd.crosstab(
        pd.Series(labels_for_k3, name="Cluster"),
        pd.Series(true_labels, name="True Sentiment"),
    ).rename(columns={0: "Negative", 1: "Neutral", 2: "Positive"})

    # we divide each count by its cluster's total and multiply by 100. this lets us see whether a cluster contains mostly one sentiment or a mix.
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    counts.to_csv(f"{OUTPUT_DIR}/cluster_label_counts.csv")
    percentages.to_csv(f"{OUTPUT_DIR}/cluster_label_percentages.csv")

    # each bar represents one cluster, and the colored sections show its sentiments.
    percentages.plot(kind="bar", stacked=True, color=["#d95f5f", "#e3bd55", "#58a66c"])


    plt.xlabel("Discovered Cluster")
    plt.ylabel("Percentage of Samples (%)")
    plt.title("Sentiment Distribution Within Each Cluster")
    plt.legend(title="True Sentiment")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cluster_composition.png", dpi=300)
    plt.close()

    # idxmax finds the row with the highest silhouette score.
    # we print that row's k value and score as a final summary.
    best_row = results_df.loc[results_df["silhouette_score"].idxmax()]

    # this table shows the number of samples from each actual sentiment class that ended up in each cluster.
    # this helps us see whether certain sentiments are grouped together or whether the clusters contain a mixture of different sentiments.
    print("\nCluster VS True Sentiment")
    print(counts)

    # this table shows the same information as percentages instead of raw counts. this is useful because the clusters may contain different numbers of samples, making raw counts harder to compare.
    # by calculating the percentage of each sentiment within a cluster, we can see whether a particular sentiment dominates that cluster.
    print("\nCluster Composition (%)")
    print(percentages.round(2))

    print(f"best k by silhouette: {int(best_row['k'])}")
    print(f"highest silhouette score: {best_row['silhouette_score']:.4f}")
    print(f"results saved to: {OUTPUT_DIR}")

# entry point
if __name__ == "__main__":
  
    main()






