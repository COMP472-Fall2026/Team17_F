
"""
this file contains our implementation of the k-nearest neighbours (knn)
classification algorithm for the project.

we implemented the algorithm ourselves using numpy instead of using
scikit-learn's existing knn classifier, as required by the assignment.

the idea behind knn is that samples with similar features are likely
to belong to the same class.

for example, if we want to classify a new image, we compare its
500 numerical features against the features of our training images.

the algorithm works in five main steps:

1. store all the training examples and their known labels.
2. calculate the distance between a new example and every training example.
3. find the k training examples with the smallest distances.
4. use the labels of those neighbours to vote for a class.
5. return the class with the highest number of votes or voting weight, which is one of the configurations we experimented with.

we support two distance metrics:
- euclidean: measures the distance between vectors in feature space.
- cosine: compares the directions of vectors.

these were used as they are quite common for this use case. more details about this can be found in the report.

we also support two voting methods, as learned in class:
- uniform: each selected neighbour receives one vote.
- distance: closer neighbours receive larger voting weights.

because our datasets are large, we use numpy matrix operations and process predictions in batches to reduce memory usage.
"""

# numpy is a python library for working with numerical arrays.

# numpy allows us to work with the hundreds of numerical features within our examples and also allows for ease of use when doing math operations on arrays.
# we use numpy because our datasets contain thousands of examples, and each example is represented by hundreds of numerical features.

import numpy as np


# knn classifier
# a class groups related variables and methods into one object.
# this is a class, and helps group our variables and methods all into one object so that when we create an object we can configure the hyperparams.
# we can give it training examples to memorize / store using fit() and then predict labels for new examples with predict()
# here, the class represents our knn classifier.

# key details:
# fit() stores the training examples and prepares some values that will make distance calculations faster.
# predict() compares new examples against the training examples, finds the nearest neighbours, and determines the predicted labels.

class KNNClassifier:
    

    # 1. constructor (__init__) creates an object, so we've configured the classifier to take specific params/hyperparams when being created:
    # so it takes k, metric (which determines which distance metric is used), weight - this is either uniform or weighted so either all neighbors get same power in vote or we weight it
    # it also takes batch size, so how many examples to process at once, and then self is just the classifier object itself that we are working with.
    # we handled it with some default values here in case they aren't specified, but in main we have the entry point to the classifier so that these params can be configured
    # these were the main things we were changing when we were doing our experiments.
    
    def __init__(self, k=5, metric="euclidean", weights="uniform", batch_size=64):

        # we need at least one neighbour to make a prediction, so this checks that k is an int and also at least 1.
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")

        #  next, we check whether the selected distance metric is supported.
        # we implemented only two options:
    
        # euclidean: compares the distance between vector positions.
        # cosine: compares the directions of the vectors.
        #
        # if the user enters another string, the program raises an error.
        if metric not in ("euclidean", "cosine"):
            raise ValueError("metric must be 'euclidean' or 'cosine'")

        # we run the same check, basically santizing inputs before moving forward.
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")

        #  we are making sure that batch size is a positive int, so if we say that batch size is X, we handle X examples simulatenously when we classify them
        # this is just so that we can control memory and not have issues with our code. it also helps make the runs faster than doing them all at once.
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        # store the hyperparameters as attributes of this object, so they can be accessed by fit() and predict() later.
        self.k, self.metric, self.weights = k, metric, weights

        # store the batch size as another object attribute.
        self.batch_size = batch_size

    # 2. fit: store the training examples, so x contains all our features, y contains class labels, so we just store the training data since knn is lazy learning algo.
 
    def fit(self, X, y):

        # convert the feature data into a numpy array, and we store in 32bit to save some memory since we have so many numerical features.
        X = np.asarray(X, dtype=np.float32)

        # we convert the labels to an array as well
        y = np.asarray(y)

        # next we verify the shape of the training inputs. x must have exactly two dimensions:
        #
        # dimension 0: training samples (rows)
        # dimension 1: features (columns)
        #
        # for example, the full image dataset has shape:
        # (20000, 500)
        #
        # y must have exactly one dimension because it contains
        # one class label for each sample.
        #
        # len(x) != len(y) checks whether the number of training xamples matches the number of labels.
        # if any of these conditions fail, the input is invalid.
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2D and y must be 1D with matching rows")

        # here we are making sure there are enough k as training examples, otherwise how can we classify and compare?
        if len(X) < self.k:
            raise ValueError("k cannot exceed the number of training samples")

        # now we are just checking the data to make sure we can process it, so is it finite?
        #
        # np.isfinite() returns true for a normal finite number.
        #
        # it returns false for:
        # nan (not a number)
        # positive infinity
        # negative infinity
        #
        # all() checks whether every element passes.
        #
        # invalid numerical values could prevent us from calculating our distances.
        if not np.isfinite(X).all():
            raise ValueError("X has NaN or infinite values")

        # find the unique class labels in the training dataset.
    
        # np.unique() returns all distinct labels in sorted order. return_inverse=True also returns a mapping that tells us the index of each label in the unique class array.
        #
        # for example:
        #
        # y = [20, 10, 20, 30]
        #
        # self.classes_ = [10, 20, 30]
        # self.targets_ = [1, 0, 1, 2]
        #
        # this mapping is useful because later we use numerical indices to count votes for each class. 
        self.classes_, self.targets_ = np.unique(y, return_inverse=True)

        # we then store the training feature matrix inside the object.
        #
        # np.ascontiguousarray() ensures the data is stored in contiguous memory. this can make numerical operations more efficient, especially matrix multiplication.
        # after this line, the classifier remembers all the training feature vectors.
        self.X_train_ = np.ascontiguousarray(X)

        # we now prepare information depending on the distance metric we selected, this is done in main.py
        if self.metric == "euclidean":

            # euclidean distance is normally calculated using:
            # distance(a,b) = sqrt(sum((a_i - b_i)^2))
            # however, our implementation uses an equivalent
            # formula for squared euclidean distance:
            # ||a-b||² = ||a||² + ||b||² - 2(a · b)
            # here, ||b||² represents the squared length
            # of a training vector.
            # we use squared distances because te distances efficiently using numpy matrix operations without having to calculate the square root for every distance. 
            #Since squaring nonnegative distances preserves their ordering, it doesn't affect which k neighbours are selected. B
            #But we're aware that this changes the behaviour of distance-weighted voting if we directly use the squared distances. 
            #Specifically, using inverse-distance weighting would effectively become (1/d^2) instead of (1/d), giving closer neighbours relatively more influence, but that is also the point of weighted voting, so we used this to avoid more operations.
            # we calculate those squared lengths once during fit() because they will be reused during prediction.
            #
            # self.x_train_ ** 2 squares every feature value. np.sum(..., axis=1) sums the squared values across the columns of each row. the result is one squared length per training example.
            self.train_sq_ = np.sum(self.X_train_ ** 2, axis=1)

        else:

            # this section runs when the metric used is cosine.
            # note that here we arent actually calculating we are just prepping the training vectors so it can be calculated at prediction time. 
            # cosine similarity measures how closely two vectors point in the same direction. this was utilized in my big data class, and we explained the reasoning in the report.
            # the mathematical formula is:
            # cosine similarity(a,b) =
            # (a · b) / (||a|| * ||b||)
            # cosine distance = 1 - cosine similarity.
            # we can simplify this calculation by normalizing each training vector to have a length of 1. np.linalg.norm() calculates the length of a vector.
            
            # axis=1 means calculate the length separately
            # for every row in the training matrix.
            # keepdims=True keeps the output as a column,
            # making it easier to divide each row by its length.
            norms = np.linalg.norm(self.X_train_, axis=1, keepdims=True)

            # then we normalize every training vector by dividing its features by the length of that vector, so we can calculate the angle.
            #
            # np.maximum(norms, 1e-12) prevents division by zero.
            #
            # 1e-12 is scientific notation for:
            # 0.000000000001
            #
            # if a vector has length zero, we divide by
            # this small number instead of dividing by zero.
            #
            # the normalized training vectors are saved
            # so we do not need to repeat this operation
            # for every prediction.
            self.normalized_train_ = self.X_train_ / np.maximum(norms, 1e-12)

        # return the classifier object after storing its data so that we can chain methods.
        return self


    # 3. predict: classify new examples, recieves a matrix of feature bectors and the correct labels are unknown to the classifier. 
    # the purpose of this method is to return one predicted class label for every input example. 
    def predict(self, X):

       
        # hasattr() checks whether an object contains an attribute with a particular name. self.x_train_ only exists after fit() stores the training examples.
        # if the attribute does not exist, we cannot compare new examples against training data.
        if not hasattr(self, "X_train_"):
            raise RuntimeError("Call fit before predict")

        # convert the prediction features to a numpy array, again using float32.
        X = np.asarray(X, dtype=np.float32)

        # verify that the new samples have the correct shape.
        # x.ndim != 2 checks that x is a two-dimensional matrix.
        # x.shape[1] gives the number of features per sample.
        # self.x_train_.shape[1] gives the number of features
        # used by the training examples.
        # the feature dimensions must match.
        # for example, if the classifier was fitted on
        # 500-dimensional image vectors, a new example
        # must also contain exactly 500 features.
        if X.ndim != 2 or X.shape[1] != self.X_train_.shape[1]:
            raise ValueError("Prediction features must match training dimensions")

        # make sure the new feature vectors do not contain nan or infinite values.
        if not np.isfinite(X).all():
            raise ValueError("X has NaN or infinite values")

        # create an empty array to store predicted labels. len(x) determines how many predictions we need.
        # dtype=self.classes_.dtype ensures the predicted labels use the same data type as our original class labels. np.empty() allocates memory without initializing every element to zero.
        # we fill each position in the array with a prediction.
        results = np.empty(len(X), dtype=self.classes_.dtype)

        # as mentioned previously, instead of predicting every new example at once, we divide the prediction data into smaller batches.
        # range(start, stop, step) generates a sequence of values. if we processed thousands of prediction sample simultaneously, the distance matrix could become huge.
        # batching also limits the number of distances stored in memory at one time.
        for start in range(0, len(X), self.batch_size):

            # select the next group of prediction samples.
            # x[start:start + self.batch_size] uses slicing.
            # for example: when start=0 and batch_size=64,
            # we select rows 0 through 63.
            # np.ascontiguousarray() ensures the resulting batch is stored contiguously in memory.
            batch = np.ascontiguousarray(X[start:start + self.batch_size])

            # 4. calculate distances
            # select the correct distance calculation, based on the classifier's metric setting.
            if self.metric == "euclidean":

                # we calculate squared euclidean distances using:
                # ||a-b||² = ||a||² + ||b||² - 2(a · b)
                #
                # a represents a new sample being classified.
                # b represents a stored training sample.
    
                # normally euclidean distance includes a square root.
                # we skip the square root because it does not change the ordering of nonnegative distances, as we mentioned earlier.
                
                # the formula has three components:
                # np.sum(batch**2, axis=1) calculates the squared length of every new sample.
                
                # [:, None] adds a new dimension to the array. it converts a shape like (64,) into (64, 1).

            
                # self.train_sq_ contains the squared lengths of all the training examples. [None, :] converts this array into a row vector, so numpy can broadcast it during addition.
                
            
                # batch @ self.x_train_.t performs matrix multiplication. @ is the matrix multiplication operator. .t transposes the training matrix, switching its rows and columns.
                
                # this allows numpy to calculate dot products between all batch samples and all training examples.
                
                # the result d is a matrix of distances. each row represents one new sample. each column represents one training sample.
                
                # for example, if the batch has 64 examples
                # and we have 16,000 stored training examples:
            
                # d has shape (64, 16000).
                # d[i,j] is the squared distance between
                # new sample i and training sample j.
                # using matrix multiplication is much faster than manually looping through every feature of every training example in python.
                d = (np.sum(batch**2, axis=1)[:, None]
                     + self.train_sq_[None, :]
                     - 2 * (batch @ self.X_train_.T))

                # mathematically, squared distance cannot be negative.
                # however, computers use finite precision when representing floating-point numbers, because of rounding, a calculation may produce a small negative result
                # np.maximum(d, 0) replaces all negative values with 0.
                # out=d means numpy modifies the original distance array instead of creating a new one.
                np.maximum(d, 0, out=d)

            else:

                # this section runs when metric="cosine". cosine distance is defined as:
                # cosine distance = 1 - cosine similarity.
                # cosine similarity uses the dot product divided by the lengths of the vectors.
        
                # we already normalized our training examples inside fit() so now we normalize the new examples as well.
                # np.linalg.norm() calculates each new sample's vector length.
                # axis=1 calculates the length row by row.
                # keepdims=True preserves the column dimension so we can divide each feature vector by its length.
                norms = np.linalg.norm(batch, axis=1, keepdims=True)

                # now we normalize the new feature vectors, dividing by the vector length produces vectors with unit length, unless they originally have zero length.
                # np.maximum() prevents division by zero.
                normalized = batch / np.maximum(norms, 1e-12)

                # now we calculate cosine distances.
            
                # normalized contains the normalized new examples.
                # self.normalized_train_ contains the normalized training examples.
                # the matrix multiplication calculates the cosine similarities between every pair.
                # subtracting those similarities from 1 converts them into cosine distances.
                # smaller cosine distances indicate, greater similarity between vectors, so similar classes.
                d = 1 - normalized @ self.normalized_train_.T

               
                # due to floating-point rounding,
                # cosine similarity can sometimes be slightly greater than 1.
                # this would produce a slightly negative distance,
                # which we correct by replacing it with zero.
                np.maximum(d, 0, out=d)

        
            # 5. find the nearest neighbours
            # np.argpartition() finds the indices of the k smallest distances in each row.
            # i.e k =2, returns the indexes of the 2 smallest distances in the row
            # argpartition returns indices that place those two values in the first two positions of the returned array.
            # kth=self.k - 1 is used because python array indexing starts at zero.
            # for k=5, the kth position is index 4.
            # axis=1 means we perform this operation separately for each row of the distance matrix.
            # [:, :self.k] selects only the first k indices from the partitioned result, effectively evaluating k neighbors.
            # the selected indices are not necessarily sorted by their distances but thats fine because voting does not require that the neighbours are sorted.
            
            # the resulting array has shape: (number of batch samples, k)
            indices = np.argpartition(d, kth=self.k - 1, axis=1)[:, :self.k]

            # now we retrieve the distances of the selected neighbours.
            # np.take_along_axis() uses indices to select valuesfrom the distance matrix.
            # axis=1 tells numpy that the indices refer which to positions along the columns of each row.
            # near_d contains the distances of the k nearest training examples for each sample.
            near_d = np.take_along_axis(d, indices, axis=1)

            # retrieve the labels of the selected neighbours.  self.targets_ contains the encoded class index associated with every training example.
            # indices identifies which training examples were selected as nearest neighbours.
            # near_y contains the class indices of the selected training examples.
        
            # for example:
            # near_y = [2, 2, 1, 0, 2]
            # would mean three neighbours belong to class 2,
            # one belongs to class 1, and one belongs to class 0.
            near_y = self.targets_[indices]

         
            # 6. create a voting table
           

            # to do this, create a matrix filled with zeros to store votes. the number of rows is equal to the number of new examples in the current batch.
            # the number of columns is equal to the number of distinct classes.
            
            # for image recognition, there are 50 classes.
            # for sentiment analysis, there are 3 classes.
            # for example, with 64 sentiment examples,
            # scores has shape (64, 3).
            # each row represents one new example.
            # each column represents one possible class.
            # scores[i,c] stores the total number of votes or voting weight assigned to class c.
            # float64 is used because weighted votes can be decimal values, so we want to make sure that we are precise.
            scores = np.zeros((len(batch), len(self.classes_)), dtype=np.float64)

            # we then loop through every new example in the batch, so we now know which k neighbours are closest, so the remaining step is to calculate how much each neighbour contributes to the overall vote.
            for i in range(len(batch)):
                # check whether we are using uniform voting. uniform means every neighbour contributes exactly one vote.
                if self.weights == "uniform":
                    votes = np.ones(self.k)

                # this branch applies when we are using distance-weighted voting and at least one neighbour has a distance extremely close to zero.
            
                # near_d[i] contains the distances of the selected neighbours for sample i.
                # near_d[i] <= 1e-8 checks whether a neighbour's
                # distance is less than or equal to 0.00000001.
            
                # np.any() returns true when at least one neighbour satisfies the condition.
                # we treat these very small values as exact matches.
                # when exact matches exist, the implementation lets only those matching neighbours vote.
                # this avoids assigning extremely large inverse-distance weights to near-zero distances.
                elif np.any(near_d[i] <= 1e-8):

                    # astype(np.float64) converts:
                    # true to 1.0
                    # false to 0.0
                    # true means the neighbour is close enough to count as an exact match.
                    # therefore, exact matches receive one vote,
                    # while all other neighbours receive zero votes.
                    votes = (near_d[i] <= 1e-8).astype(np.float64)

                else:

                    # this branch applies when we are using distance-weighted voting and there are no neighbours close enough to zero, to count as exact matches.
                    # we calculate each neighbour's voting weight using the inverse of its distance.
                    # the general idea is: weight = 1 / distance

                    # therefore, closer neighbours receive larger voting weights.
                    
                    # np.maximum() ensures that the divisor is never smaller than 1e-8.
                    
                    # important detail:
                    # when metric="euclidean", near_d contains squared euclidean distances because we skipped the square root earlier.
                
                    # therefore, our euclidean weighted voting actually uses:
                
                    # weight = 1 / squared_distance
                    # but this is fine, because it gives more influence to closer neighbors but that is the point of weighted voting.
                    
                    # when metric="cosine", near_d contains
                    # cosine distances, so the formula uses
                    # the inverse of cosine distance instead.
                    votes = 1.0 / np.maximum(near_d[i], 1e-8)

               
                # 7. add the votes to their corresponding classes
                

                # np.add.at() adds voting weights to the scores associated with each class.
            
                # scores[i] is the voting score array for the current prediction sample.
                # near_y[i] contains the class indices of the selected neighbours.
                
                # votes contains their voting weights. np.add.at() is especially useful because multiple neighbours can belong to the same class. it correctly accumulates repeated additions, to the same index.
                #  we then use use np.argmax() to select the class with the highest total score
                np.add.at(scores[i], near_y[i], votes)

            
            # 8. choose the predicted class
            # np.argmax() finds the index of the largest value. scores contains the voting totals for every sample in the current batch.
            # axis=1 means we find the class with the highest score separately for each row.

            # for example:
    
            # class scores = [1, 1, 3]
            # np.argmax() returns 2 because the value 3 is located at index 2.
            # if multiple classes have the same highest score, np.argmax() returns the first matching index.
            # since np.unique() sorted the original class labels, during fit(), this means voting ties are resolved in favour of the smallest class label.
            # self.classes_[...] converts the winning class indices back into their original class labels.
            # results[start:start + len(batch)] selects the positions in the results array corresponding to the current batch.
            # we store each predicted label in its correct position.
            results[start:start + len(batch)] = self.classes_[np.argmax(scores, axis=1)]

        # after all batches have been processed, results contains one predicted label for every example that was passed into predict().
        # we can now compare these prediction against known validation labels 
        return results
