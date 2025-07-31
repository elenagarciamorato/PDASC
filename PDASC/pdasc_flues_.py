import logging

import numpy as np

from PDASC.utils import *
from PDASC.pdasc_ import create_tree, recursive_approximate_knn_search_radius_pruning
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import random
import joblib

# Clustering methods to be used: k-means, k-medoids
# import sklearn.cluster  # k-means sklearn implementation
# from PDASC.clustering_algorithms import kmeans_kclust  # k-means k clust implementation
# import sklearn_extra.cluster  # k-medoids sklearn_extra implementation
import kmedoids as fast_kmedoids  # k-medoids fast_k-medoids (PAM) implementation

# Set up logging
logger = logging.getLogger(__name__)

# Set seed for reproducibility
SEED = 10
np.random.seed(SEED)
random.seed(SEED)

#####    INDEX STORAGE AND LOADING FUNCTIONS    #####
def store_PDASC_index_flue(dataset, distance_function, id_flue, index):
    file_path = f'benchmarks/logs/{dataset}/{str(dataset)}_{str(distance_function)}_index_{id_flue}.joblib'
    joblib.dump(index, file_path)
    print(f"Index for flue {id_flue} stored at {file_path}")

def load_PDASC_index_flue(dataset, distance_function, id_flue):
    file_path = f'benchmarks/logs/{dataset}/{str(dataset)}_{str(distance_function)}_index_{id_flue}.joblib'
    return joblib.load(file_path)


#####    DISTRIBUTED INDEX BUILDING FUNCTIONS    #####
def create_index_flues(training_set, dataset, group_size, n_centroids, n_flues, dist_func, algorithm, implementation):

    # Total number of points in the training set
    n_total = training_set.shape[0]

    # The size of each node is calculated by dividing the total number of points by the number of nodes, with the last node possibly being smaller
    tam_nodos = int(np.ceil(n_total / n_flues))  # Size of each node (or flue) in the tree

    print(f"Total points: {n_total}, Nodes: {n_flues}, Size of each node: {tam_nodos}")

    # We split the training set into partitions of size tam_nodos
    training_set_partitions = [np.array(training_set[i:i + tam_nodos]) for i in range(0, len(training_set), tam_nodos)]

    # We build the lowest layer of the PDASC index

    # Create a pool of workers to build the index corresponding to each flue in parallel using the create_tree function
    with Pool() as pool:
        build_index_flue = partial(create_tree, tam_grupo=group_size,
                                        n_centroides=n_centroids, metric=dist_func,algorithm=algorithm, implementation=implementation)

        indexes_flues = pool.map(build_index_flue, training_set_partitions)


    # Filter out None results
    indexes_flues = [ind_f for ind_f in indexes_flues if ind_f is not None]

    for i in range(len(indexes_flues)):
        # Store the index for each flue
        print(f"Storing index for flue {i} with {indexes_flues[i][0]} layers")
        store_PDASC_index_flue(dataset, dist_func, i, indexes_flues[i])

    return None


#####    DISTRIBUTED ANN SEARCH FUNCTIONS    #####
def recursive_ANN_search_flue(id_flue, punto_buscado, dataset, flue_size, n_centroids, dist_function, radius):

    # print(f"Processing flue {id_flue}")

    index_flue = load_PDASC_index_flue(dataset, dist_function, id_flue)  # Extraemos el DataFrame de la chimenea (flue) a procesar
    #print(len(index_flue))

    n_capas = index_flue[0]
    grupos_capa = index_flue[1]
    puntos_capa = index_flue[2]
    labels_capa = index_flue[3]
    promoted_points = index_flue[4]


    # Create an array of k_vecinos * 2 elements where the value of them are the initial radius
    # candidates = np.full(int(k_vecinos * 10), initial_radius, dtype=float)

    # Establish the query point
    punto_buscado = punto_buscado.reshape(1, -1)
    # print("El punto de query es: ", punto_buscado)

    # (At the first level, current layer=n_capas-1 and current_group = grupos_capa[n_layer].size[0]-1 = 0)
    inheritage = [0]

    # At the first lever, the radius to be used is the biggest one
    # first_layer_radius = max_radius
    # print(first_layer_radius)

    # We take the top-layer prototypes, including its coordinates and distances to the query point
    coordinates_top_prototypes = np.vstack(puntos_capa[n_capas - 1][:])
    distances_top_prototypes = get_distances(np.array(punto_buscado), coordinates_top_prototypes, dist_function)
    # print(coordinates_top_prototypes)
    # print(distances_top_prototypes)
    # print(distances_top_prototypes)

    # distances_top_prototypes = distance.cdist(np.array(punto_buscado), coordinates_top_prototypes, metric=metrica)[0]
    # distances_top_prototypes = pairwise_distances(np.array(punto_buscado), coordinates_top_prototypes, metric=metrica)[0]
    # print(distances_top_prototypes)

    # At the first lever, the radius to be used is the biggest one
    # first_layer_radius = max_radius
    # print(first_layer_radius)

    # We store the distances computed on the distances_computed lists
    distances_computed = distances_top_prototypes.tolist()
    n_distances_computed = len(distances_top_prototypes)

    # At the top layer, we explore every prototype
    neighbours = []
    for prototype_id in range(len(distances_top_prototypes)):
        # prototype_coords = coordinates_top_prototypes[prototype_id]
        prototype_distance = distances_top_prototypes[prototype_id]
        # aux_neighbors, aux_n_distances = explore_centroid_CDFradius1(punto_buscado, n_capas, inheritage, prototype_id, prototype_coords, puntos_capa, labels_capa, grupos_capa, promoted_points,n_centroids, metrica, [], 0, min_radius)
        aux_neighbors, aux_n_distances = explore_centroid_dynamicradius_minradius(punto_buscado, n_capas, inheritage,
                                                                                  prototype_id, prototype_distance,
                                                                                  puntos_capa, labels_capa, grupos_capa,
                                                                                  promoted_points, n_centroids,
                                                                                  dist_function, [], 0, radius, id_flue, flue_size)

        # Para todos los aux_neighbors, aux_neighbors[0]=aux_neigbors + id_flue*flue_size

        neighbours.extend(aux_neighbors)
        n_distances_computed += aux_n_distances

    # Once the complete index has been explored:
    # print(neighbours)
    return neighbours, n_distances_computed

def recursive_ANN_search_radius_pruning(punto, dataset, vector_training, n_flues, n_centroids, dist_function, radius, k_vecinos):

    # Establece el punto de búsqueda como un array de una sola fila
    punto_buscado = punto.reshape(1, -1)

    # Calcula el tamaño de las flues (excepto el último nodo que puede ser más pequeño):
    flue_size = int(np.ceil(len(vector_training) / n_flues))  # Número de flues (training_set_partitions) en el índice
    # print("Tamaño de cada flue:", tam_flue)

    # Creamos una lista para almacenar los vecinos encontrados
    neighbours = []

    # Creamos una variable para almacenar el número de distancias computadas
    n_distances_computed = 0

    # Empaquetamos todos los argumentos
    args = [
        (flue, punto_buscado, dataset, flue_size, n_centroids, dist_function, radius)
        for flue in np.arange(n_flues)
    ]

    # Aplicamos la función de búsqueda seleccionada a cada chimenea (flue) en paralelo
    with Pool() as pool:
        results = pool.starmap(recursive_ANN_search_flue, args)

    # Recolectamos los resultados
    for neighbours_flue, n_distances_computed_flue in results:
        n_distances_computed += n_distances_computed_flue
        neighbours.extend(neighbours_flue)

    # Una vez exploradas todas las chimeneas (flues), tenemos una lista de vecinos potenciales
    # print(f"Total number of neighbours found: {len(neighbours)}")

    # If no neighbours have been found:
    if not neighbours:

        # print("No neighbours have been found for this query point")

        # Pad the array of close points with None objects until it reaches the size of k neighbors
        # To avoid index out of bounds error
        vacio = np.empty(k_vecinos, dtype=int), np.empty([k_vecinos, vector_training.shape[0]], dtype=float), np.empty(k_vecinos, dtype=float), n_distances_computed

        print(f'vacio:{vacio}')
        return vacio

    # If any neighbour have been found:
    else:

        # print(f'{len(neighbours)} neighbours have been found for this query point')

        # The neighbours whose distance is already computed are those which are stored as tuples
        neighbours_with_d = [n for n in neighbours if isinstance(n, tuple)]
        # print(f'There are {len(neighbours_with_d)} which distance is already computed')

        # Separate tuple\_neighbours into two sublists: one for the ids and one for the distances to the query point
        id_neighbours_with_d = [n[0] for n in neighbours_with_d]
        # print(sorted(id_neighbours_with_d, reverse=True))
        distances_neighbours_with_d = [n[1] for n in neighbours_with_d]

        # By acceding the original dataset, we obtain its coordinates
        coords_neighbours_with_d = vector_training[id_neighbours_with_d]

        # For control, print the distances already computed
        # print(f'The distances computed until now are {len(distances_computed)}')

        # The neighbours whose distance is not computed yet are those which are not tuples
        id_neighbours_without_d = [n for n in neighbours if not isinstance(n, tuple)]
        # print(f'There are {len(id_neighbours_without_d)} which distance is not computed yet')

        # By acceding the original dataset, we obtain its coordinates and compute its distances to the query point
        coords_neighbours_without_d = vector_training[id_neighbours_without_d]

        if len(coords_neighbours_without_d) > 0:
            distances_neighbours_without_d = get_distances(np.array(punto_buscado), coords_neighbours_without_d,
                                                           dist_function)
        else:
            distances_neighbours_without_d = np.empty(0)
        # distances_neighbours_without_d = distance.cdist(np.array(punto_buscado), coords_neighbours_without_d, metric=metrica)[0]
        # distances_neighbours_without_d = pairwise_distances(np.array(punto_buscado), coords_neighbours_without_d, metric=metrica)[0]

        # And add the number of distances computed at this step to the n_distances_computed counter
        n_distances_computed += len(distances_neighbours_without_d)

        # We concatenate the neighbours whose distance is already computed and the neighbours whose distance is not computed yet
        k_neighbours_ids = np.concatenate((id_neighbours_with_d, id_neighbours_without_d))
        k_neighbours_coords = np.concatenate((coords_neighbours_with_d, coords_neighbours_without_d))
        k_neighbours_dists = np.concatenate((distances_neighbours_with_d, distances_neighbours_without_d))
        np.set_printoptions(threshold=np.inf)
        # print(f'Number of neighbours: {len(k_neighbours_ids)}')
        # print(f'Indices of the neighbours (sorted descending): {k_neighbours_ids[np.argsort(k_neighbours_ids)[::-1]]}')

        # And we store the info about each neighbour together into a single structure
        k_neighbours = np.empty((len(k_neighbours_ids), 3), object)

        k_neighbours[:, 0] = k_neighbours_ids
        k_neighbours[:, 1] = list(k_neighbours_coords)
        k_neighbours[:, 2] = k_neighbours_dists

        k_neighbours = np.vstack(k_neighbours)
        # print(k_neighbours)

        # To be able to find the k nearest

        # Drop the neighbors whose distance does not meet the condition (dist < radius)
        k_neighbours = k_neighbours[k_neighbours[:, 2] <= radius]

        # Sort by distance (column 2)
        sorted_neighbours = k_neighbours[np.argsort(k_neighbours[:, 2])]
        num_found = sorted_neighbours.shape[0]
        #print(f"Number of neighbours after radius pruning: {num_found}")

        # Select number of neighbors to retain
        minimum = min(k_vecinos, num_found)

        # Initialize outputs directly with fallback values (to avoid later conditional rewriting)
        indices_vecinos = np.full(k_vecinos, -1, dtype=int)
        coords_vecinos = np.full((k_vecinos, vector_training.shape[1]), np.nan)
        dists_vecinos = np.full(k_vecinos, np.nan)

        if minimum > 0:
            # Use vectorized slicing where possible
            selected = sorted_neighbours[:minimum]
            indices_vecinos[:minimum] = selected[:, 0].astype(int)
            dists_vecinos[:minimum] = selected[:, 2].astype(float)

            # If selected[:, 1] contains arrays/vectors, use stacking only if necessary
            coords_stack = np.array([v for v in selected[:, 1]])
            coords_vecinos[:minimum, :] = coords_stack


    # print(punto)
    # print(indices_vecinos, coords_vecinos, dists_vecinos, n_distances)

    return indices_vecinos, coords_vecinos, dists_vecinos, n_distances_computed

def distributed_ANN_search(vector_testing, vector_training, dataset, n_flues, n_centroids, dist_function, radius, k):

    # Update the metric name for compatibility with scipy
    if dist_function == 'manhattan':
        dist_function = 'cityblock'  # scipy cdist requires 'cityblock' instead of 'manhattan'


    """
    #### Process each query point in parallel using ThreadPoolExecutor ####
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda punto: recursive_ANN_search_radius_pruning(
                punto, dataset, vector_training, n_flues, n_centroids, dist_function, radius, k
            ),
            vector_testing
        ))
    """

    #### Process each query point secuentially ####
    # Create a list to store the results
    results = []

    # Iterate over each point in the testing set
    for punto in vector_testing:
        # Call the recursive knn search function for each point
        indices, coords, distances, n_distances = recursive_ANN_search_radius_pruning(
                punto, dataset, vector_training, n_flues, n_centroids, dist_function, radius, k
            )
        results.append((indices, coords, distances, n_distances))


    # Unzip the results into separate lists
    indices_vecinos, coords_vecinos, dists_vecinos, n_distances = zip(*results)

    return indices_vecinos, coords_vecinos, dists_vecinos, n_distances



