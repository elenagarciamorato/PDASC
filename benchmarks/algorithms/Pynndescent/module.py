import numpy as np
import pynndescent


def PYNN_nn_index(dataset, distance_type):

    # Create a PYNN instance and build and index
    index = pynndescent.NNDescent(dataset, metric=distance_type)
    index.prepare()

    return index

def PYNN_nn_search(train_set, test_set, k, d, index):

    # Find the knn of each point in seq_buscada using this index
    lista_indices, lista_coords, lista_dists = [], [], []

    # For every point contained on the train set (the complete dataset in this case), find its k
    # nearest neighbors on this dataset using the index built previously
    # and the distance used to build it

    for f in range(test_set.shape[0]):
        # print("Point number " + str(f))
        neighbors = index.query([test_set[f]], k)

        lista_indices.append(neighbors[0])
        lista_coords.append(train_set[neighbors[0][0]])
        lista_dists.append(neighbors[1])

    # Return knn and their distances with the query points
    #logging.info(str(k) + "-Nearest Neighbors found using PYNN + " + distance_type + " distance + " + algorithm + " algorithm.")

    # The number of distance computations required to obtain the knn are unknown
    n_distances = np.NaN

    return np.array(lista_indices), np.array(lista_coords), np.array(lista_dists), n_distances

# Function that returns the accepted distances by PYNNdescent
def PYNN_accepted_distances():
    return pynndescent.distances.named_distances

