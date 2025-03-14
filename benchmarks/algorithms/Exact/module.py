from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import numpy as np
import logging

''' Note: sklearn.NearestNeighbors
sklearn.NearestNeighbors is an unsupervised learner for implementing neighbor searches.

Exact method and metric to search for the nearest neighbor could be chosen, among others parameters.

- Algorithm used to compute the nearest neighbors:
    ‘ball_tree’ will use BallTree (same function as sklearn.neighbors.BallTree)
    ‘kd_tree’ will use KDTree (same function as sklearn.neighbors.KDTree)
    ‘brute’ will use a brute-force search.
    ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.

- Metrics able to use (depending on the algorithm chosen):
    (KDTree.valid_metrics): 'euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity'
    (KDTree.valid_metrics):'euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity', 'seuclidean', 'mahalanobis', 'hamming', 'canberra', 'braycurtis', 'jaccard', 'dice', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'haversine', 'pyfunc'
    (NearestNeighbors.VALID_METRICS['brute']): 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'haversine', 'jaccard', 'l1', 'l2', 'mahalanobis', 'manhattan', 'minkowski', 'nan_euclidean', 'precomputed', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
    
    By default, the algorithm would be chosen according to the distance metric that is wanted to be used
    (KDTree and KDTree would be preferable (only) according to its performance),
    but it can also be provided by the user on config files
'''
# Using sklearn.neighbors.nearestNeighbors, build the index of nearest neighbors using the distance metric chosen
# By default, the exact algorithm chosen to build the index that will depend on the metric
# but can be also provided by the user
def Exact_nn_index(train_set, metric, exact_algorithm):

    if exact_algorithm == 'auto':
    # Based on the metric that is going to be used, choose an exact algorithm that supports it
        if metric == 'cosine' or 'haverstine':
            exact_algorithm = 'brute'
        else:
            exact_algorithm = 'kd_tree'

    elif exact_algorithm == 'LinearScan':
        return None

    # Determine the knn of each element on the train_set and build the index accordingly (build nn estimator)
    tree_index = NearestNeighbors(metric=metric, algorithm=exact_algorithm).fit(train_set)
    logging.info("Building index using " + metric + " metric and " + exact_algorithm + " algorithm")

    return tree_index

# Find the k nearest neighbors of the elements constituting the test set through a linear scan
def LinearScan_nn_search(train_set, test_set, k, metric, same_set=None):

    # Build the structures to store data regarding the future neighbors
    indices = np.empty([len(test_set), k], dtype=int)
    coords = np.empty([len(test_set), k, test_set.shape[1]], dtype=float)
    dists = np.empty([len(test_set), k], dtype=float)

    """ Naive implementation
    # Calculate the pairwise distances between the elements of two samples
    distances = np.empty(0)
    for i in range(0, len(sample1)):
        pair = np.array([sample1[i], sample2[i]])
        d = pdist(pair, 'euclidean')
        distances = np.insert(distances, len(distances), d)
    """

    # Update the metric name for compatibility with scipy
    if metric == 'manhattan':
        metric = 'cityblock'  # scipy cdist requires 'cityblock' instead of 'manhattan'


    # Calculate the pairwise distances between the elements of two samples
    # distances = cdist(test_set, train_set, metric)
    distances = pairwise_distances(test_set, train_set, metric=metric)

    # For every point in the testing set, take the k elements with the smallest distances (k-nn)
    for i in range(len(test_set)):
        sorted_points = np.argsort(distances[i])

        # If the test set is the same as the train set, we avoid considering the distances between the same points
        if same_set:
            sorted_points = sorted_points[1:k + 1]

        # If the test set is different from the train set, we consider the distances between the all the points
        else:
            sorted_points = sorted_points[:k]

        # Store the indices, coordinates and distances of the k nearest neighbors
        indices[i] = sorted_points
        dists[i] = distances[i][sorted_points]
        coords[i] = train_set[sorted_points]

    # Return knn and the number of distance computations required to obtain them
    #print(f"Los vecinos exactos son: {indices} con distancias {dists}")

    return indices, coords, dists, distances.size


# Find the k nearest neighbors of the elements constituting the test set through an exact method
def Exact_nn_search(vector_training, vector_testing, k, metric, tree_index, same_set=None):

    # Build the arrays to store the indices, coordinates and distances of the k nearest neighbors
    indices_vecinos = np.empty([len(vector_testing), k], dtype=int)
    coords_vecinos = np.empty([len(vector_testing), k, vector_testing.shape[1]], dtype=float)
    dists_vecinos = np.empty([len(vector_testing), k], dtype=float)

    # And number of distances computed
    n_distances_vecinos = np.empty(len(vector_testing))

    # If any index is provided, find the knn of the test_set elements through a linear scan
    if tree_index is None:

        # For every point in the testing set, find its k nearest neighbors
        for i in range(len(vector_testing)):
            #print("Punto número: ", i)
            punto = vector_testing[i].reshape(1, -1)

            indices, coords, dists, n_distances = LinearScan_nn_search(vector_training, punto, k, metric, same_set)

            indices_vecinos[i] = indices[0]
            coords_vecinos[i] = coords[0]
            dists_vecinos[i] = dists[0]
            n_distances_vecinos[i] = n_distances

    else:

        # For every point in the testing set, find its k nearest neighbors
        for i in range(len(vector_testing)):
            #print("Punto número: ", i)
            punto = vector_testing[i].reshape(1, -1)

            # Find the knn of the test_set elements between those contained on the train_set index
            dists, indices = tree_index.kneighbors(punto, k)

            # Get the coordinates of the found neighbors
            coords = np.array(vector_training[indices])

            indices_vecinos[i] = indices[0]
            coords_vecinos[i] = coords[0]
            dists_vecinos[i] = dists[0]

            # Unknown number of distance computations
            n_distances_vecinos[i] = None


        # Return knn and the number of distance computations required to obtain them
        #print(f"Los vecinos exactos son: {indices} con distancias {dists}")

    return indices_vecinos, coords_vecinos, dists_vecinos, n_distances_vecinos

