from pyflann import *
import logging
import os


def FLANN_nn_index(dataset, ncentroids, distance_type, algorithm):

    # Sets the distance type used. Possible values: euclidean, manhattan, minkowski, max_dist, hik, hellinger, cs, kl.
    set_distance_type(distance_type, order=0)

    # Create a FLANN instance and build and index
    flann = FLANN()
    flann.build_index(dataset, algorithm=algorithm)

    # Using kmeans, compute the n-centroids describing the data
    #centroids = flann.kmeans(dataset, num_clusters=ncentroids, max_iterations=None, mdtype=None)

    # print centroids

    # Store index built on disk to use it later on a file called 'index_'
    flann.save_index('./benchmarks/algorithms/FLANN/index_')
    logging.info("Saving FLANN index at 'index_'")

    # Store index on disk to obtain its size
    #with open("./algorithms/FLANN/MNIST_knn.pickle", 'wb') as handle:
        #dump(flann.nn_index, handle)

    return None


def FLANN_nn_search(dataset, seq_buscada, k, distance_type, algorithm):

    # Sets the distance type used. Possible values: euclidean, manhattan, minkowski, max_dist, hik, hellinger, cs, kl.
    set_distance_type(distance_type, order=0)

    # If there is an index stored on disk:
    if os.path.isfile('./benchmarks/algorithms/FLANN/index_'):

        # Create a FLANN instance
        flann = FLANN()

        # Load the index stored
        flann.load_index('./benchmarks/algorithms/FLANN/index_', dataset)
        #logging.info("Loading FLANN index from 'index_'...")

    else:
        logging.info("No FLANN index found. Creating a new one...")

        # Create a FLANN instance
        flann = FLANN_nn_index(dataset, 128, distance_type, algorithm)



    # Then, find the knn of each point in seq_buscada using the loaded/created index
    lista_indices, lista_coords, lista_dists = [], [], []

    # For every point contained on the train set (the complete dataset in this case), find its k
    # nearest neighbors on this dataset using FLANN and the index built previously
    for f in range(seq_buscada.shape[0]):
        # print("Point number " + str(f))
        indices, dists = flann.nn_index(seq_buscada[f], num_neighbors=k, algorithm=algorithm, checks=64)
        indices = indices.reshape(indices.size,)
        coords = np.array(dataset[indices])

        lista_indices.append(indices)
        lista_coords.append(coords)
        lista_dists.append(dists)


    # Return knn and their distances with the query points
    #logging.info(str(k) + "-Nearest Neighbors found using FLANN + " + distance_type + " distance + " + algorithm + " algorithm.")

    # The number of distance computations required to obtain the knn are unknown
    n_distances = np.NaN

    return np.array(lista_indices), np.array(lista_coords), np.array(lista_dists), n_distances

# Function that returns the accepted distances by FLANN
def FLANN_accepted_distances():
    return ['euclidean', 'manhattan', 'minkowski', 'max_dist', 'hik', 'hellinger', 'cs', 'kl']

# Function that returns the accepted algorithms by FLANN
def FLANN_accepted_algorithms():
    return ['kdtree', 'kmeans', 'linear', 'composite', 'kdtree_single', 'composite', 'autotuned', 'saved']