from copy import deepcopy, copy

# import numpy as np
from sklearn import preprocessing
from timeit import default_timer as timer
import logging
from GDASC.utils import *
# from sys import getsizeof

# Clustering methods to be used: k-means, k-medoids
# import sklearn.cluster  # k-means sklearn implementation
# from GDASC.clustering_algorithms import kmeans_kclust  # k-means k clust implementation
# import sklearn_extra.cluster  # k-medoids sklearn_extra implementation
import kmedoids as fast_kmedoids  # k-medoids fast_k-medoids (PAM) implementation

logger = logging.getLogger(__name__)


def create_tree(vector_original, tam_grupo, n_centroides, metric, algorithm, implementation):
    """
    Constructs a hierarchical tree structure using clustering algorithms.

    Parameters:
    vector_original : np.array
            Original data points.
    tam_grupo : int
            Size of the group to be clustered with centroids.
    n_centroides : int
            Number of centroids to be used in each clustering.
    metric   : str
            Distance function to be used for distance calculation in clustering.
    algorithm : str
            Clustering algorithm to be used ('kmedoids' or 'kmeans').
    implementation : str
            Specific implementation of the clustering algorithm to use.

    Returns:
    tuple: A tuple containing:
        - n_layers  : int
                Number of layers in the hierarchical tree.
        - grupos_capa : list
                Group structure for each layer.
        - puntos_capa : list
                Centroids for each group in each layer.
        - labels_capa : list
                Labels assigned to each point in each group for each layer.

    Note:
        To avoid future redundant distance computations, the structures are improved to contain nan values in the place of duplicated points.
    """

    normaliza = False

    # Starts the iterative process of construction-deconstruction of the index
    start_time_constr = timer()

    vector = vector_original
    if normaliza:
        vector = preprocessing.normalize(vector, axis=0, norm='l2')

    vector_length = vector_original.shape[0]
    vector_dimensionality = vector_original.shape[1]
    n_layers = calculate_numcapas(vector_length, tam_grupo, n_centroides)
    puntos_capa, labels_capa, grupos_capa, promoted_points = built_estructuras_capa_new(vector_length, tam_grupo,
                                                                                        n_centroides, n_layers,
                                                                                        vector_dimensionality)
    simplified_puntos_capa = copy(puntos_capa)
    duplicates = 0

    # Iterative process to apply the clustering algorithm to each layer
    for id_layer in range(n_layers):
        ngrupos = len(grupos_capa[id_layer])
        inicio = 0
        cont_ptos = 0  # Counts numer of points of each layer
        n_points = np.zeros(ngrupos, dtype=int)

        for id_group in range(ngrupos):
            fin = inicio + tam_grupo

            # We control that the last group has the correct number of points
            if fin > vector_length:
                fin = vector_length
            n_points[id_group] = fin - inicio

            if ((fin - inicio) >= n_centroides):
                if algorithm == 'kmedoids':

                    if implementation == 'sklearnextra':

                        print(f'implementation {algorithm} {implementation}')


                    elif implementation == 'fastkmedoids':

                        kmedoids = fast_kmedoids.KMedoids(n_clusters=n_centroides, method='fasterpam',
                                                          metric=metric).fit(vector[inicio:fin])

                        # print(f"cluster_centers {kmedoids.cluster_centers_}")  # coordenadas de los (nc=35) puntos que promocionan como medoides
                        # print(f"indices {kmedoids.medoid_indices_}")  # indices de los (nc=35) puntos que promocionan como medoides
                        # print(f"labels {kmedoids.labels_}")  # medoide al que se asociará cada uno de los tg=70 puntos en el nivel superior (cluster)

                        labels_capa[id_layer][id_group] = kmedoids.labels_

                        puntos_capa[id_layer][id_group] = kmedoids.cluster_centers_
                        simplified_puntos_capa[id_layer][id_group] = puntos_capa[id_layer][id_group]

                        prototype_points = kmedoids.medoid_indices_

                        # For the first layer, we store the promoted real points in an auxiliar structure
                        if id_layer == 0:
                            # Setting all points to False
                            promoted_points[id_group] = np.full(fin - inicio, False, dtype=bool)
                            # And setting to True the points that have been promoted as prototypes
                            promoted_points[id_group][prototype_points] = True

                        # For the intermediate layers, we set to NaN the corresponding point in the lower layer
                        elif 0 < id_layer < n_layers:

                            # For every point that has been promoted as prototype, we set to NaN the corresponding point in the lower layer
                            for i in prototype_points:
                                id_lower_group = id_group * 2 + (i >= n_centroides)
                                element = int(i % n_centroides)
                                simplified_puntos_capa[id_layer - 1][id_lower_group][element] = np.nan
                                duplicates += 1

                    cont_ptos += n_centroides

                elif algorithm == 'kmeans':

                    if implementation == 'sklearn':

                        print(f'implementation {algorithm} {implementation}')

                    elif implementation == 'kclust':

                        print(f'implementation {algorithm} {implementation}')


                else:  # Never accessed

                    print("A valid clustering algorithm is needed")

            # If the group has less points than centroids
            else:

                # We store the points in the group as they are
                puntos_capa[id_layer][id_group] = np.array(vector[inicio:fin])

                # Next layer for each group
                cont_ptos = cont_ptos + (fin - inicio)
                etiquetas = []
                for i in range((fin - inicio)):
                    etiquetas.append(i)

                labels_capa[id_layer][id_group] = np.array(etiquetas)
                prototype_points = list(range(fin - inicio))

                # The same process is done in order to simplify the structure
                # For the first layer, we store the promoted real points in an auxiliar structure
                if id_layer == 0:
                    promoted_points[id_group] = np.full(fin - inicio, False, dtype=bool)
                    promoted_points[id_group][prototype_points] = True

                # For the intermediate layers, we set to NaN the corresponding point in the lower layer
                elif 0 < id_layer < n_layers:

                    # For every point that has been promoted as prototype, we set to NaN the corresponding point in the lower layer
                    for i in prototype_points:
                        id_lower_group = id_group * 2 + (i >= n_centroides)
                        element = int(i % n_centroides)
                        simplified_puntos_capa[id_layer - 1][id_lower_group][element] = np.nan
                        duplicates += 1

            inicio = fin

        grupos_capa[id_layer] = n_points

        # We store the centroids of the layer to be able to do the inverse process
        vector = puntos_capa[id_layer]
        vector = np.concatenate(vector).ravel().tolist()
        vector = np.array(vector)
        vector = vector.reshape(cont_ptos, vector_dimensionality)

        # Calculate the number of groups for the next layer
        vector_length = cont_ptos  # Actualizamos vector_length con el número de puntos del siguiente nivel

    '''## Funciona pero solo detecta 139 coincidencias

    print(promoted_points)
    simplified_puntos_capa = copy(puntos_capa)

    duplicates = 0
    for capa in range(len(grupos_capa) - 2, -1, - 1):
        for grupo in range(len(grupos_capa[capa])):
            tam_grupo = len(puntos_capa[capa][grupo])
            tam_grupo_padre = len(puntos_capa[capa+1][grupo//2])
            #print(f'Visiting group {grupo} of layer {capa} with {tam_grupo} elements')
            #print(tam_grupo)

            for i in range(tam_grupo):
                #print(f'Checking element {i} of group {grupo} of layer {capa}')
                #print(len(puntos_capa[capa+1][grupo//2]))
                for j in range(tam_grupo_padre):
                    if np.array_equal(puntos_capa[capa][grupo][i], puntos_capa[capa+1][grupo//2][j]) and ((labels_capa[capa+1][grupo//2][i] == j)) : # Hay que revisar esta ultima , no se si es correcta (revisar incluir esta parte en la construcción)
                        #print(f'Element {i} of group {grupo} of layer {capa} is the same as element {j} of group {grupo//2} of layer {capa+1}')
                        duplicates = duplicates + 1
                        simplified_puntos_capa[capa][grupo][i] = np.nan
                        #print(simplified_puntos_capa[capa][grupo][i])
                    else:
                        simplified_puntos_capa[capa][grupo][i] = puntos_capa[capa][grupo][i]
                        #print(f'Group {grupo} of layer {capa} has {duplicates} duplicated elements')
    '''

    # Print the number of elements that have been substituted by NaN
    # print(f'Duplicates found = {duplicates}')

    end_time_constr = timer()

    logger.info('tree time=%s seconds', end_time_constr - start_time_constr)

    return n_layers, grupos_capa, simplified_puntos_capa, labels_capa, promoted_points


def knn_search(n_capas, n_centroides, seq_buscada, vector_original, vecinos, centroides_examinados,
               n, metrica, grupos_capa, puntos_capa, labels_capa):
    """
    Performs a k-nearest neighbors (KNN) search using a hierarchical tree structure.

    Parameters:
    n_capas         : int
            Number of layers in the hierarchical tree.
    n_centroides          : int
            Number of centroids in each group.
    seq_buscada           : numpy.ndarray
            The query point for which the nearest neighbors are to be found.
    vector_original       : numpy.ndarray
            The original dataset of points.
    vecinos               : list
            List to store the nearest neighbors found during the search.
    centroides_examinados : numpy.ndarray
            Array to track which centroids have been examined.
    n : int
            Current number of nearest neighbors found.
    metrica               : str
            The distance metric to use for calculating distances.
    grupos_capa           : list
            Number of points in each group at each layer.
    puntos_capa           : list
            Cluster centroids at each layer.
    labels_capa           : list
            Labels assigned to each point at each layer.

    Returns:
    bool                   : True if a new neighbor was stored, False otherwise.
    """

    # En principio, esta búsqueda sobre el árbol seria exactamente igual que con MASK (mask_search)
    print("******************** DECONSTRUCTION PROCESS *********************")
    logger.info('tree-depth=%s', n_capas)
    lista_pos = np.empty(100, int)
    # Reshape the query sequence
    seq_buscada = np.reshape(seq_buscada, (1, 2))
    # Iterate over each layer from the deepest to the root
    for id_capa in range(n_capas - 1, -1, -1):
        # Obtain centroids of the current layer
        centroides = puntos_capa[id_capa]
        centroides = np.concatenate(centroides)

        # Select points associated with the current centroid if not at the deepest layer
        if id_capa < (n_capas - 1):
            # seleccionamos solo los puntos que están asociados con ese centroide
            centroides = np.array(centroides[lista_pos])

        puntos_dist = np.concatenate([seq_buscada, centroides])
        D = pairwise_distances(puntos_dist, metric=metrica)  # euclidean, chebyshev, manhattan
        columna = busca_dist_menor(D)
        # Correct centroid index
        if id_capa != (n_capas - 1):
            pos_centroide = lista_pos[columna - 1]
            if pos_centroide >= n_centroides:
                id_grupo = int(pos_centroide / n_centroides)
                id_centroide = pos_centroide - (id_grupo * n_centroides)
            else:
                id_centroide = pos_centroide
                id_grupo = 0
        else:
            # Corrección para cuando la última capa del arbol tiene más de un grupo
            if len(grupos_capa[id_capa]) > 1:
                if (columna - 1) >= n_centroides:
                    id_grupo = int((columna - 1) / n_centroides)
                    id_centroide = (columna - 1) - (id_grupo * n_centroides)
                else:
                    id_centroide = columna - 1
                    id_grupo = 0
            else:
                id_centroide = columna - 1
                id_grupo = 0

        lista_pos_aux = np.argwhere(labels_capa[id_capa][id_grupo][:] == id_centroide)
        lista_pos = built_lista_pos(id_grupo, grupos_capa[id_capa][:], lista_pos_aux)
        lista_pos = lista_pos.ravel()

    # Select points at the data layer
    puntos_seleccionados = np.array(vector_original[lista_pos])
    puntos_dist = np.concatenate([seq_buscada, puntos_seleccionados])
    D = distance.pdist(puntos_dist, metric=metrica)  # scipy resulta más eficiente
    columna = busca_dist_menor(D)
    id_punto = lista_pos[columna - 1]

    # Control de los vecinos guardados (ciclados)
    # Si el id_punto encontrado ya lo teniamos guardado en vecinos, nos quedamos con el siguiente
    # mas cercano
    # Store the found neighbor or find another candidate
    vecino = np.empty(5, object)
    almacenado = False
    if n == 0:
        # Guardamos directamente el vecino encontrado (es el primero)
        vecino[0] = id_punto
        vecino[1] = D[0, columna]
        vecino[2] = vector_original[id_punto]
        vecino[3] = id_grupo
        vecino[4] = id_centroide
        vecinos[n] = vecino
        almacenado = True
        if len(lista_pos) == 1:
            # No hay más puntos asociados con ese centroide, por lo que lo marcamos (ponemos su posición
            # a 1) como ya examinado
            centroides_examinados[id_grupo][id_centroide] = 1
    else:
        # Buscamos si el nuevo vecino esta ya guardado
        ids_vecinos = np.zeros(n, dtype=int)
        for x in range(n):
            ids_vecinos[x] = vecinos[x][0]
        index = np.ravel(np.asarray(ids_vecinos == id_punto).nonzero())

        if len(index) == 0:
            # No lo tenemos guardado, por lo tanto, lo guardamos
            vecino[0] = id_punto
            vecino[1] = D[0, columna]
            vecino[2] = vector_original[id_punto]
            vecino[3] = id_grupo
            vecino[4] = id_centroide
            vecinos[n] = vecino
            almacenado = True
            if len(lista_pos) == 1:
                # No hay más puntos asociados con ese centroide, por lo que lo marcamos (ponemos su posición
                # a 1) como ya examinado
                centroides_examinados[id_grupo][id_centroide] = 1
        else:
            # Si lo tenemos guardado. Buscamos otro candidato
            if len(lista_pos) == 1:
                # No tengo más candidatos asociados a ese centriode. Hay que buscar un nuevo centroide y examinar
                # sus candidatos
                id_ult_vecino = vecinos[n - 1][0]
                id_punto, dist, id_grupo, new_id_centroide = search_near_centroid(id_grupo, id_centroide,
                                                                                  id_ult_vecino, centroides_examinados,
                                                                                  puntos_capa,
                                                                                  labels_capa, grupos_capa, ids_vecinos,
                                                                                  vector_original, metrica)
                vecino[0] = id_punto
                vecino[1] = dist
                vecino[2] = vector_original[id_punto]
                vecino[3] = id_grupo
                vecino[4] = new_id_centroide
                vecinos[n] = vecino
                almacenado = True
            else:
                # Tenemos más candidatos asociados a ese centroide. Buscamos el siguiente punto más cercano
                new_lista_pos = np.setdiff1d(lista_pos, ids_vecinos)
                if len(new_lista_pos) == 0:
                    id_ult_vecino = vecinos[n - 1][0]
                    id_punto, dist, id_grupo, new_id_centroide = search_near_centroid(id_grupo, id_centroide,
                                                                                      id_ult_vecino,
                                                                                      centroides_examinados,
                                                                                      puntos_capa, labels_capa,
                                                                                      grupos_capa, ids_vecinos,
                                                                                      vector_original, metrica)
                    vecino[0] = id_punto
                    vecino[1] = dist
                    vecino[2] = vector_original[id_punto]
                    vecino[3] = id_grupo
                    vecino[4] = new_id_centroide
                    vecinos[n] = vecino
                    almacenado = True
                else:
                    puntos_seleccionados = np.array(vector_original[new_lista_pos])
                    vecino_guardado = (np.array(vector_original[id_punto])).reshape(1, 2)
                    puntos_dist = np.concatenate([vecino_guardado, puntos_seleccionados])
                    D = pairwise_distances(puntos_dist, metric=metrica)
                    new_colum = busca_dist_menor(D)
                    id_punto = new_lista_pos[new_colum - 1]

                    vecino[0] = id_punto
                    vecino[1] = D[0, new_colum]
                    vecino[2] = vector_original[id_punto]
                    vecino[3] = id_grupo
                    vecino[4] = id_centroide
                    vecinos[n] = vecino
                    almacenado = True
                    if len(new_lista_pos) == 1:
                        # No hay más puntos asociados con ese centroide, por lo que lo marcamos (ponemos su posición
                        # a 1) como ya examinado
                        centroides_examinados[id_grupo][id_centroide] = 1

    print("END OF DECONSTRUCTION PROCESS")

    return almacenado


def knn_approximate_search(n_centroides, punto_buscado, vector_original, k_vecinos, metrica,
                           grupos_capa, puntos_capa, labels_capa, dimensiones, radio):
    """
    Performs an approximate k-nearest neighbors (KNN) search using a hierarchical tree structure
    and a search radius.

    Parameters:
    n_centroides : int
            Number of centroids in each group.
    punto_buscado : numpy.ndarray
                The query point for which the nearest neighbors are to be found.
    vector_original : numpy.ndarray
                The original dataset of points.
    k_vecinos : int
            Number of nearest neighbors to find.
    metrica : str
            The distance metric to use for calculating distances.
    grupos_capa :  list
             Number of points in each group at each layer.
    puntos_capa : list
            Cluster centroids at each layer.
    labels_capa : list
             Labels assigned to each point at each layer.
    dimensiones : int
            Number of dimensions of the points.
    radio : float
        Search radius for the approximate search.

    Returns:
    list: A list containing three elements:
        - numpy.ndarray: Indices of the k nearest neighbors.
        - numpy.ndarray: Coordinates of the k nearest neighbors.
        - numpy.ndarray: Distances to the k nearest neighbors.
    """
    #  A la hora de buscar los mas vecinos más cercanos, se utiliza tambien la métrica
    # que se pasa como argumento y que teoricamente debe ser la misma con la que se construyó el arbol,
    # calculando las distancias a traves de la función de scipy cdist(punto, puntos, metrica)

    # La búsqueda es aproximada porque se limita a un radio

    # Update the metric name for compatibility with scipy
    if metrica == 'manhattan':
        metrica = 'cityblock'  # scipy cdist requires 'cityblock' instead of 'manhattan'

    punto_buscado = np.reshape(punto_buscado, (1, dimensiones))
    # Directly move to layer 1 (the one above the data layer) and set the search radius( we set as radius 3 times the smallest distance)

    id_capa = 0
    centroides = puntos_capa[id_capa]
    centroides = np.concatenate(centroides)
    puntos_dist = np.concatenate([punto_buscado, centroides])

    D = distance.cdist(punto_buscado, centroides, metric=metrica)[0]

    radius = radio  # 3 * dist_nearest_centroid (1.15 glove completo, 3 glove100000, 5 MNIST)

    # Para cada uno de los centroides con distancia menor a radius nos quedamos con k vecinos más cercanos
    # Primero almacenamos los índices de los centroides que cumplen la condición
    # filad = D[0, 1:]

    # Find centroids within the search radius
    selec_centroides = np.flatnonzero(D <= radius)
    # Preallocate the array to store centroid group and ID for each of them
    ids_selec_centroides = np.empty(len(selec_centroides), dtype=object)

    # 29/09/24 CODE IMPROVEMENT: Obtain group and centroid IDs in a vectorized manner

    for i, sc in enumerate(selec_centroides):
        id_grupo = sc // n_centroides  # Integer division to get the group ID
        id_centroide = sc % n_centroides  # Modulo operation to get the centroid ID
        ids_selec_centroides[i] = (id_grupo, id_centroide)

    # Compute the total size of selected centroids based on labels
    tam = sum(np.count_nonzero(labels_capa[id_capa][id_grupo][:] == id_centroide)
              for id_grupo, id_centroide in ids_selec_centroides)

    # Find the points associated with those centroids
    ids_selec_points = np.empty(tam, int)
    ini = 0
    fin = 0
    for idg, idc in ids_selec_centroides:
        lista_pos_aux = np.argwhere(labels_capa[id_capa][idg][:] == idc)
        lista_pos = built_lista_pos(idg, grupos_capa[id_capa][:], lista_pos_aux)
        lista_pos = np.reshape(lista_pos, len(lista_pos))
        fin += len(lista_pos)
        ids_selec_points[ini:fin] = lista_pos
        ini = fin

    # Within all the selected points, we would only store those who meet the condition (dist<radius)
    puntos_seleccionados = np.array(vector_original[ids_selec_points])
    dist = distance.cdist(np.array(punto_buscado), np.array(puntos_seleccionados), metric=metrica)

    aux_ids_points = np.array(np.nonzero(dist <= radius))  # +1
    aux_ids_points = aux_ids_points[1]
    ids_points = ids_selec_points[aux_ids_points]
    dist_points = dist[dist <= radius]

    puntos_cercanos = np.empty((len(ids_points), 3), object)
    for i in range(len(ids_points)):
        puntos_cercanos[i][0] = ids_points[i]
        puntos_cercanos[i][1] = dist_points[i]
        puntos_cercanos[i][2] = vector_original[ids_points[i]]

    # Creamos las estructuras para almacenar los datos relativos a los vecinos
    # Structures to store the nearest neighbors
    indices_k_vecinos = np.empty(k_vecinos, dtype=int)
    coords_k_vecinos = np.empty([k_vecinos, vector_original.shape[1]], dtype=float)
    dists_k_vecinos = np.empty(k_vecinos, dtype=float)

    # Completar el array de puntos cercanos  con None s hasta llegar al tamaño de vecinos deseado (k_vecinos)
    # Esto evita el error index out of bounds

    # Pad the array of close points with None until it reaches the size of k neighbors
    if len(puntos_cercanos) < k_vecinos:
        puntos_cercanos = np.append(puntos_cercanos, np.full((k_vecinos - len(puntos_cercanos), 3), None), axis=0)

    # Sort points by distance to the query point
    # Ordenamos los puntos en base a su distancia con el punto de query
    idx = np.argsort(puntos_cercanos[:, 1])

    # Select the k closest points as neighbors
    # Designamos los k_vecinos puntos con menor distancia al punto de consulta como vecinos
    for i in range(k_vecinos):
        indices_k_vecinos[i] = puntos_cercanos[idx[i]][0]
        coords_k_vecinos[i, :] = puntos_cercanos[idx[i]][2]
        dists_k_vecinos[i] = puntos_cercanos[idx[i]][1]

    return [indices_k_vecinos, coords_k_vecinos, dists_k_vecinos]


def recursive_approximate_knn_search(n_capas, n_centroides, vector_testing, vector_original, k_vecinos, metrica,
                           grupos_capa, puntos_capa, labels_capa, promoted_points, initial_radius, dataset):
    """
    Performs an approximate k-nearest neighbors (A-KNN) search using a hierarchical tree structure
    and a search radius.

    It is prepared to manage the new version of explore_centroid_optimised() that may return a list of
    candidate neighbours whose distance to the query point is known or must be computed, depending on the situation.


    Parameters:
    n_capas : int
        Number of layers in the hierarchical tree.
    n_centroides : int
        Number of centroids in each group.
    punto_buscado : numpy.ndarray
        The query point for which the nearest neighbors are to be found.
    vector_original : numpy.ndarray
        The original dataset of points.
    k_vecinos : int
        Number of nearest neighbors to find.
    metrica : str
        The distance metric to use for calculating distances.
    grupos_capa : list
        Number of points in each group at each layer.
    puntos_capa : list
        Cluster centroids at each layer.
    labels_capa : list
        Labels assigned to each point at each layer.
    promoted_points: list
        List of boolean arrays to track which points from the original dataset have been promoted as prototypes
    radio : float
        Search radius for the approximate search.

    Returns:
    list: A list containing three elements:
        - numpy.ndarray: Indices of the k nearest neighbors.
        - numpy.ndarray: Coordinates of the k nearest neighbors.
        - numpy.ndarray: Distances to the k nearest neighbors.
"""
    # Update the metric name for compatibility with scipy
    if metrica == 'manhattan':
        metrica = 'cityblock'  # scipy cdist requires 'cityblock' instead of 'manhattan'


    # Creamos las estructuras para almacenar los futuros vecinos
    indices_vecinos = np.empty([len(vector_testing), k_vecinos], dtype=int)
    coords_vecinos = np.empty([len(vector_testing), k_vecinos, vector_testing.shape[1]], dtype=float)
    dists_vecinos = np.empty([len(vector_testing), k_vecinos], dtype=float)

    # Y el número de distancias calculadas en cada ejecución
    n_distances = np.empty([len(vector_testing)], dtype=int)


    # For every point in the testing set, find its k nearest neighbors
    for punto in range(len(vector_testing)):

        #print(f"Punto: {punto}")

        # We obtain the list of adaptative radius to be used
        dynamic_radius_list = get_dynamic_radius_list(n_capas, initial_radius, dataset)

        # Create an array of k_vecinos * 2 elements where the value of them are the initial radius
        # candidates = np.full(int(k_vecinos * 10), initial_radius, dtype=float)

        # Establish the query point
        punto_buscado = vector_testing[punto].reshape(1, -1)
        #print("El punto de query es: ", punto_buscado)

        # (At the first level, current layer=n_capas-1 and current_group = grupos_capa[n_layer].size[0]-1 = 0)
        inheritage = [0]

        # At the first lever, the radius to be used is the first one stored in the dynamic_radius_list
        current_layer_radius = dynamic_radius_list[-1]
        # current_layer_radius = candidates[-1]
        # print(current_layer_radius)

        # We take the top-layer prototypes, including its coordinates and distances to the query point
        coordinates_top_prototypes = np.vstack(puntos_capa[n_capas-1][:])
        distances_top_prototypes = distance.cdist(np.array(punto_buscado), coordinates_top_prototypes, metric=metrica)[0]

        # We store the distances computed on the distances_computed lists
        distances_computed = distances_top_prototypes.tolist()

        # We would only explore those prototypes which meets the condition / are within a radius (dist<radius)
        explorable_prototypes = np.where(distances_top_prototypes <= current_layer_radius)[0]


        # We search for every neighbour by exploring each top-layer prototype that meets the radius condition (established for each layer) recursively
        neighbours = []

        for prototype_id in explorable_prototypes:
            prototype_distance = distances_top_prototypes[prototype_id]
            explore_centroid_optimised(punto_buscado, n_capas, inheritage, prototype_id, prototype_distance, puntos_capa, labels_capa, grupos_capa, promoted_points, n_centroides, metrica, neighbours, distances_computed, dynamic_radius_list)

        # Once the complete index has been explored:
        # Get the number of total distances computed

        # print(neighbours)

        # If no neighbours have been found:
        if not neighbours:

            #print("No neighbours have been found for this query point")

            # Pad the array of close points with None objects until it reaches the size of k neighbors
            # To avoid index out of bounds error
            return [np.empty(k_vecinos, dtype=int), np.empty([k_vecinos, vector_original.shape[1]], dtype=float), np.empty(k_vecinos, dtype=float)], len(distances_computed)

        # If any neighbour have been found:
        else:

            # print(f'{len(neighbours)} neighbours have been found for this query point')

            # The neighbours whose distance is already computed are those which are stored as tuples
            neighbours_with_d = [n for n in neighbours if isinstance(n, tuple)]
            # print(f'There are {len(neighbours_with_d)} which distance is already computed')

            # Separate tuple\_neighbours into two sublists: one for the ids and one for the distances to the query point
            id_neighbours_with_d = [n[0] for n in neighbours_with_d]
            distances_neighbours_with_d = [n[1] for n in neighbours_with_d]

            # By acceding the original dataset, we obtain its coordinates
            coords_neighbours_with_d = vector_original[id_neighbours_with_d]

            # For control, print the distances already computed
            # print(f'The distances computed until now are {len(distances_computed)}')

            # The neighbours whose distance is not computed yet are those which are not tuples
            id_neighbours_without_d = [n for n in neighbours if not isinstance(n, tuple)]
            # print(f'There are {len(id_neighbours_without_d)} which distance is not computed yet')

            # By acceding the original dataset, we obtain its coordinates and compute its distances to the query point
            coords_neighbours_without_d = vector_original[id_neighbours_without_d]
            distances_neighbours_without_d = distance.cdist(np.array(punto_buscado), coords_neighbours_without_d, metric=metrica)[0]

            # And add the distances computed on the distances_computed list
            distances_computed.extend(distances_neighbours_without_d.tolist())

            # We concatenate the neighbours whose distance is already computed and the neighbours whose distance is not computed yet
            neighbours_ids = np.concatenate((id_neighbours_with_d, id_neighbours_without_d))
            neighbours_coords = np.concatenate((coords_neighbours_with_d, coords_neighbours_without_d))
            neighbours_dists = np.concatenate((distances_neighbours_with_d, distances_neighbours_without_d))

            # And we store the info about each neighbour together into a single structure
            neighbours = np.empty((len(neighbours_ids), 3), object)

            neighbours[:, 0] = neighbours_ids
            neighbours[:, 1] = list(neighbours_coords)
            neighbours[:, 2] = neighbours_dists

            neighbours = np.vstack(neighbours)

            # To be able to find the k nearest

            # Create the structures to store the data related to the neighbors
            indices_k_vecinos = np.empty(k_vecinos, dtype=int)
            coords_k_vecinos = np.empty([k_vecinos, vector_original.shape[1]], dtype=float)
            dists_k_vecinos = np.empty(k_vecinos, dtype=float)

            # Sort them according to their distance to the query point
            sorted_neighbours = neighbours[neighbours[:, 2].argsort()]

            # Select the minimum value between k_vecinos and the number of neighbours founded
            minimum = min(k_vecinos, sorted_neighbours.shape[0])

            # Select the k closest points as neighbors (using vectorised operations and avoiding the loop
            indices_k_vecinos[:minimum] = sorted_neighbours[:minimum, 0]
            coords_k_vecinos[:minimum, :] = np.vstack(sorted_neighbours[:minimum, 1])
            dists_k_vecinos[:minimum] = sorted_neighbours[:minimum, 2]

            # Print them
            # print(f"The neighbours are: {indices_k_vecinos} with distances {dists_k_vecinos}")

            # And return the results
            # print(f"The search process computes a total of {len(distances_computed)} distances")

            indices_vecinos[punto] = indices_k_vecinos
            coords_vecinos[punto] = coords_k_vecinos
            dists_vecinos[punto] = dists_k_vecinos
            n_distances[punto] = len(distances_computed)

    return indices_vecinos, coords_vecinos, dists_vecinos, n_distances
