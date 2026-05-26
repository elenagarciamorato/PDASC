import logging

from pynndescent.distances import jaccard

from PDASC.utils import *
from PDASC.pdasc_ import create_tree
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import random
import joblib
import os

# Clustering methods to be used: k-means, k-medoids
# import sklearn.cluster  # k-means sklearn implementation
# from PDASC.clustering_algorithms import kmeans_kclust  # k-means k clust implementation
# import sklearn_extra.cluster  # k-medoids sklearn_extra implementation
import kmedoids as fast_kmedoids  # k-medoids fast_k-medoids (PAM) implementation

# Set up logging
logger = logging.getLogger(__name__)

# Set seed for reproducibility
#SEED = 10
#np.random.seed(SEED)
#random.seed(SEED)


#####    INDEX STORAGE AND LOADING FUNCTIONS    #####
def store_PDASC_index_flue(dataset, distance_function, tg, nc, n_flues, id_flue, index):

    # Set the path of the file where the index will be stored
    os.makedirs(f'ANN_Experiments/NearestNeighbors/{dataset}/indexes', exist_ok=True) # If the directory does not exist, create it
    file_path = f'ANN_Experiments/NearestNeighbors/{dataset}/indexes/{str(dataset)}_{str(distance_function)}_nc{str(nc)}_tg{str(tg)}_index_{n_flues}-{id_flue}.joblib'
    joblib.dump(index, file_path)
    print(f"Index for flue {id_flue} stored at {file_path}"),


def load_PDASC_index_flue(dataset, distance_function, tg, nc, n_flues=1, id_flue=0):

    if distance_function=='cityblock':
        distance_function = 'manhattan'  # scipy cdist requires 'cityblock' instead of 'manhattan'
    file_path = f'ANN_Experiments/NearestNeighbors/{dataset}/indexes/{str(dataset)}_{str(distance_function)}_nc{str(nc)}_tg{str(tg)}_index_{n_flues}-{id_flue}.joblib'
    print(f"Loading index for flue {id_flue} from {file_path}")
    return joblib.load(file_path)


#####    DISTRIBUTED INDEX BUILDING FUNCTIONS    #####

def simulate_flue_partitioning(training_set, n_flues, distance_function=None, n_centroids=None):


    # Total number of points in the training set
    n_total = training_set.shape[0]

    if (distance_function=='jaccard') & (n_flues>1):
        # For Jaccard distance, we want all partitions except the last one to have the same size,
        # and that size to be the largest multiple of n_centroids that fits in the division

        n_parts = n_flues - 1  # number of equal partitions (the last one may be smaller)
        avg_for_parts = n_total // n_parts
        base_size = (avg_for_parts // n_centroids) * n_centroids  # mayor múltiplo de n_centroids <= avg_for_k

        if base_size == 0:
            base_size = min(n_centroids, n_total)

        training_set_partitions = []
        start = 0
        for _ in range(n_parts):
            end = min(start + base_size, n_total)
            training_set_partitions.append(training_set[start:end])
            start = end
            if start >= n_total:
                break

        # Last partition takes the remaining elements
        training_set_partitions.append(training_set[start:])

    else:

        # The size of each node is calculated by dividing the total number of points by the number of nodes, with the last node possibly being smaller
        tam_nodos = int(np.ceil(n_total / n_flues))  # Size of each node (or flue) in the tree

        # We split the training set into partitions of size tam_nodos
        training_set_partitions = [np.array(training_set[i:i + tam_nodos]) for i in range(0, n_total, tam_nodos)]

    print(f"Total points: {n_total}, Nodes: {n_flues}, Size of each node: {[len(part) for part in training_set_partitions]}")
    return training_set_partitions

def create_index_flues(training_set, dataset, group_size, n_centroids, n_flues, dist_func, algorithm, implementation):

    # Simulate the partitioning of the training set into flues
    training_set_partitions = simulate_flue_partitioning(training_set, n_flues, dist_func, n_centroids)

    #print(f"Tamaños de cada partición en training_set_partitions: {[len(part) for part in training_set_partitions]}")

    # We build the lowest layer of the PDASC index

    # Create a pool of workers to build the index corresponding to each flue in parallel using the create_tree function
    with Pool() as pool:
        build_index_flue = partial(create_tree, tg=group_size,
                                        nc=n_centroids, distance_function=dist_func,algorithm=algorithm, implementation=implementation)

        indexes_flues = pool.map(build_index_flue, training_set_partitions)


    # Filter out None results
    indexes_flues = [ind_f for ind_f in indexes_flues if ind_f is not None]

    for i in range(len(indexes_flues)):
        # Store the index for each flue
        print(f"Storing index for flue {i} of {n_flues} with {indexes_flues[i][0]} layers")
        store_PDASC_index_flue(dataset, dist_func, group_size, n_centroids, n_flues, i, indexes_flues[i])

    return True


#####    DISTRIBUTED ANN SEARCH FUNCTIONS    #####
def recursive_ANN_search_flue(id_flue, n_flues, punto_buscado, dataset, flue_size, tg, n_centroids, dist_function, radius, pruning_strategy):

    # print(f"Processing flue {id_flue}")

    index_flue = load_PDASC_index_flue(dataset, dist_function, tg, n_centroids, n_flues, id_flue)  # Extraemos el DataFrame de la chimenea (flue) a procesar
    #print(len(index_flue))

    n_capas = index_flue[0]
    grupos_capa = index_flue[1]
    puntos_capa = index_flue[2]
    labels_capa = index_flue[3]
    promoted_points = index_flue[4]
    ordered_indices = index_flue[5]


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

    # distances_top_prototypes = distance.cdist(np.array(punto_buscado), coordinates_top_prototypes, metric=metrica)[0]
    # distances_top_prototypes = pairwise_distances(np.array(punto_buscado), coordinates_top_prototypes, metric=metrica)[0]

    # At the first lever, the radius to be used is the biggest one
    # first_layer_radius = max_radius
    # print(first_layer_radius)

    # We store the distances computed on the distances_computed lists
    #distances_computed = distances_top_prototypes.tolist()
    n_distances_computed = len(distances_top_prototypes)

    # At the top layer, we explore every prototype
    neighbours = []

    # Regarding the pruning strategy to be used:
    # If pruning_strategy is True, we use the dynamic radius with min_radius
    if pruning_strategy:
        resultados = [
            explore_centroid_dynamicradius_minradius(
                punto_buscado, n_capas, inheritage, prototype_id, prototype_distance,
                puntos_capa, labels_capa, grupos_capa, promoted_points, ordered_indices, tg, n_centroids,
                dist_function, [], 0, radius, id_flue, flue_size
            )
            for prototype_id, prototype_distance in enumerate(distances_top_prototypes)
        ]
    # If pruning_strategy is False, we use the static radius
    else:
        resultados = [
            explore_centroid_staticradius(
                punto_buscado, n_capas, inheritage, prototype_id, prototype_distance,
                puntos_capa, labels_capa, grupos_capa, promoted_points, ordered_indices, tg, n_centroids,
                dist_function, [], 0, radius, id_flue, flue_size
            )
            for prototype_id, prototype_distance in enumerate(distances_top_prototypes)
        ]

    # We collect the results
    for aux_neighbors, aux_n_distances in resultados:
        neighbours.extend(aux_neighbors)
        n_distances_computed += aux_n_distances

    # Once the complete index has been explored:
    return neighbours, n_distances_computed

def recursive_ann_search_coordinated(punto, dataset, vector_training, n_flues, tg, n_centroids, dist_function, radius, k_vecinos, pruning_strategy):

    # Establece el punto de búsqueda como un array de una sola fila
    punto_buscado = punto.reshape(1, -1)

    # Calcula el tamaño de las flues (particiones) según la métrica de distancia:
    if (dist_function == 'jaccard') & (n_flues > 1):
        # Para Jaccard, se busca que todas las particiones menos la última tengan el mismo tamaño,
        # y que ese tamaño sea el mayor múltiplo posible de n_centroids que quepa en la división.
        n_part = n_flues - 1  # número de particiones iguales (la última puede ser más pequeña)
        avg_for_part = len(vector_training) // n_part  # tamaño promedio entero por partición
        flue_size = (avg_for_part // n_centroids) * n_centroids  # mayor múltiplo de n_centroids <= avg_for_part

    else:
        # Para otras distancias, simplemente se reparte el total entre el número de flues,
        # redondeando hacia arriba para que todas tengan al menos un elemento.
        flue_size = int(np.ceil(len(vector_training) / n_flues))
    # print("Tamaño de cada flue:", flue_size)

    # Creamos una lista para almacenar los vecinos encontrados
    neighbours = []

    # Creamos una variable para almacenar el número de distancias computadas
    n_distances_computed = 0


    # Empaquetamos todos los argumentos
    args = [
        (flue, n_flues, punto_buscado, dataset, flue_size, tg, n_centroids, dist_function, radius, pruning_strategy)
        for flue in np.arange(n_flues)
    ]

    # Aplicamos la función de búsqueda seleccionada a cada chimenea (flue) en paralelo
    with Pool() as pool:
        results = pool.starmap(recursive_ANN_search_flue, args)

    # Imprime las estadísticas de cada flue
    #for flue_id, (neighs, n_dists) in enumerate(results):
    #    print(f"Flue {flue_id}: Found {len(neighs)} neighbours, computed {n_dists} distances")

    # Recolectamos los resultados
    for neighbours_flue, n_distances_computed_flue in results:
        n_distances_computed += n_distances_computed_flue
        neighbours.extend(neighbours_flue)

    # Una vez exploradas todas las chimeneas (flues), tenemos una lista de vecinos potenciales
    # print(f"Total number of neighbours found: {len(neighbours)}")

    # If no neighbours have been found:
    if not neighbours:

        print("No neighbours have been found for this query point")

        # Pad the array of close points with None objects until it reaches the size of k neighbors
        # To avoid index out of bounds error
        vacio = np.empty(k_vecinos, dtype=int), np.empty([k_vecinos, vector_training.shape[0]], dtype=float), np.empty(k_vecinos, dtype=float), n_distances_computed

        #print(f'vacio:{vacio}')
        return vacio

    # If any neighbour have been found:
    else:

        # print(f'{len(neighbours)} neighbours have been found for this query point')

        # The neighbours whose distance is already computed are those which are stored as tuples
        neighbours_with_d = [n for n in neighbours if isinstance(n, tuple)]
        #print(f'There are {len(neighbours_with_d)} which distance is already computed')

        # Separate tuple\_neighbours into two sublists: one for the ids and one for the distances to the query point
        id_neighbours_with_d = [n[0] for n in neighbours_with_d]
        # print(sorted(id_neighbours_with_d, reverse=True))
        distances_neighbours_with_d = [n[1] for n in neighbours_with_d]

        # By acceding the original dataset, we obtain its coordinates
        coords_neighbours_with_d = vector_training[id_neighbours_with_d]

        # For control, print the distances already computed
        #print(f'The distances computed until now are {n_distances_computed}')

        # The neighbours whose distance is not computed yet are those which are not tuples
        id_neighbours_without_d = [n for n in neighbours if not isinstance(n, tuple)]
        #print(f'There are {len(id_neighbours_without_d)} which distance is not computed yet')
        #print(sorted(id_neighbours_without_d))

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

def ANN_search(vector_testing, vector_training, dataset, n_flues, tg, n_centroids, dist_function, radius, k, pruning_strategy):

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
        indices, coords, distances, n_distances = recursive_ann_search_coordinated(
                punto, dataset, vector_training, n_flues, tg, n_centroids, dist_function, radius, k, pruning_strategy
            )

        results.append((indices, coords, distances, n_distances))


    # Unzip the results into separate lists
    indices_vecinos, coords_vecinos, dists_vecinos, n_distances = zip(*results)

    return indices_vecinos, coords_vecinos, dists_vecinos, n_distances


def PDASC_accepted_distances():
    return ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

