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
    #logging.info(str(k) + "-Nearest Neighbors found using FLANN + " + distance_type + " distance + " + algorithm + " algorithm.")

    # The number of distance computations required to obtain the knn are unknown
    n_distances = np.NaN

    return np.array(lista_indices), np.array(lista_coords), np.array(lista_dists), n_distances


'''
def PYNN_nn_search_antiguo(train, test, k, distance, index):

    # Find the knn of each point in seq_buscada using this index
    neighbors = index.query([test[0]], k)
    print(neighbors)

    vecinos = np.empty([neighbors[0].shape[0], neighbors[0].shape[1]])
    coordenadas = np.empty([neighbors[0].shape[0], neighbors[0].shape[1], 2])
    distancias = np.empty([neighbors[1].shape[0], neighbors[1].shape[1]])

    for i in range(len(neighbors[0])):
        vecinos[i] = neighbors[0][i]
        coordenadas[i] = train[neighbors[0][i]]
        distancias[i] = neighbors[1][i]

    #logging.info(str(k) + "-Nearest Neighbors found using PYNN + " + distance + " distance.")

    return vecinos, coordenadas, distancias
'''
