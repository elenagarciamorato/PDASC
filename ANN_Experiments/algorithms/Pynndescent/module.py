import numpy as np
import pynndescent


def PYNN_nn_index(dataset, distance, n_neighbors, diversify_prob, pruning_degree_multiplier):

    # Create a PYNN instance and build and index
    # index = pynndescent.NNDescent(dataset, metric=distance_type)

    """
    ## Configuracion (buena para PDASC mala para PYNN) para MNIST
    index = pynndescent.NNDescent(
        dataset,
        # n_neighbors=10,  # muy pequeño → grafo ligero - default=15
        n_trees=2,  # un solo árbol → construcción ultrarápida - default=10
        n_iters=2,  # pocas iteraciones → baja convergencia - default = 10
        max_candidates=5,  # pocos candidatos - default = 60
        # delta=0.001,  # actualiza poco → menos refinamiento - default = 0.001
    )
    """

    """
    index = pynndescent.NNDescent(
        dataset,
        metric=distance,
        n_neighbors=n_neighbors,  # muy pequeño → grafo ligero - default=15
        n_trees=n_trees,  # un solo árbol → construcción ultrarápida
        n_iters=n_iters,  # pocas iteraciones → baja convergencia - default = 10
        max_candidates=max_candidates,  # pocos candidatos - default = 60
        delta=delta,  # actualiza poco → menos refinamiento - default = 0.001
    )
    """

    index = pynndescent.NNDescent(
        dataset,
        metric=distance,
        n_neighbors = n_neighbors, # default=30
        diversify_prob = diversify_prob, # default=1 -> probability that an edge identified as redundant will get pruned
        pruning_degree_multiplier = pruning_degree_multiplier # default=1.5 ->Higher multiples result in more accurate graphs with more edges that take longer to search

    )

    index.prepare()

    return index

# Function that searches the k nearest neighbors using a previously built index
def PYNN_nn_search(train_set, test_set, k, d, index, epsilon):

    # Find the knn of each point in seq_buscada using this index
    lista_indices, lista_coords, lista_dists = [], [], []

    # For every point contained on the train set (the complete dataset in this case), find its k
    # nearest neighbors on this dataset using the index built previously
    # and the distance used to build it

    estimated_distances = 0

    for f in range(test_set.shape[0]):
        # print("Point number " + str(f))

        if d == 'jaccard':
            neighbors = index.query(test_set[f], k, epsilon=epsilon)
        else:
            neighbors = index.query([test_set[f]], k, epsilon=epsilon)

        # Estimate the number of distance computations as the number of points examined by PYNN
        n_neighbors = min(30, train_set.shape[0])  # PYNN by default examines 30 neighbors
        default_search_k = n_neighbors * 3
        estimated_distances = estimated_distances + default_search_k * 1
        # print(f"Estimated distances for point {f}: {default_search_k}")

        lista_indices.append(neighbors[0])
        lista_coords.append(train_set[neighbors[0][0]])
        lista_dists.append(neighbors[1])

    # Return knn and their distances with the query points
    #logging.info(str(k) + "-Nearest Neighbors found using PYNN + " + distance_type + " distance + " + algorithm + " algorithm.")

    # The number of distance computations required to obtain the knn are unknown
    n_distances = np.nan

    return np.array(lista_indices), np.array(lista_coords), np.array(lista_dists), n_distances

# Function that returns the accepted distances by PYNNdescent
def PYNN_accepted_distances():
    return pynndescent.distances.named_distances

