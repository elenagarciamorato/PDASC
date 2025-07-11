import dask
from dask import delayed, compute
from dask.distributed import Client
import dask.dataframe as dd
import dask.array as da
import kmedoids as fast_kmedoids
from timeit import default_timer as timer
import pandas
import pandas as pd
from multiprocessing import Pool
from functools import partial
from sklearn.preprocessing import normalize
import random

from PDASC.utils import *
from benchmarks.neighbors_utils import save_neighbors_and_performance
from data.load_train_test_set import *

SEED = None
np.random.seed(SEED)
random.seed(SEED)



@dask.delayed
def procesa_particion(cluster, capa, grupo, n_centroides, dist_func, random_state=SEED):

    #print(f"Procesando capa {capa}, grupo {grupo}")
    # print(particion)
    # Luego conviértelo a NumPy
    if type(cluster) is not np.ndarray:
        cluster = cluster.to_numpy()
    if cluster.shape[0] == 0:
        print("La partición no tiene ninguna dimensión, ERROR")
        return None
    else:

        # print(particion)
        # Esta función se aplicará a cada partición de forma paralela
        kmedoids = fast_kmedoids.KMedoids(n_clusters=n_centroides, method='fasterpam',
                                          metric=dist_func, random_state=random_state).fit(cluster)

        # Store the labels assigned to each point in the current group
        labels = kmedoids.labels_
        #print(labels)

        # Agrupar los índices para cada clase (prueba)
        #new_labels = [np.where(labels == i)[0].tolist() for i in range(n_centroides)]
        #print(new_labels)

        # Store the cluster centers (medoids) for the current group
        coords_prototipos = kmedoids.cluster_centers_

        # Store the indices of the points promoted as medoids
        indices_prototipos = kmedoids.medoid_indices_

        """
        output_listas = [None] * (n_centroides * 2)
        output_coords = [None] * (n_centroides * 2)
        for idx, lista, coord in zip(indices_prototipos, new_labels, coords_prototipos):
            if 0 <= idx < n_centroides * 2:
                output_listas[idx] = lista
                output_coords[idx] = coord

        """
        #print(new_labels)
        #print(indices_prototipos)
        #print(output_listas)
        #print(output_coords)


        return pd.DataFrame([{
            "capa": capa,
            "grupo": grupo,
            "labels": labels,
            "coords_prototipos": coords_prototipos,
            "indices_prototipos": indices_prototipos,
            #"simplified_puntos": simplified_puntos_capa
        }])


@dask.delayed
def procesa_grupos(df_completo, capa, grupos, n_centroides, dist_func):
    # Esto te da una Series de listas (o arrays)
    df_porcion = df_completo[
        (df_completo["capa"] == capa - 1) & (df_completo["grupo"].isin(grupos))
        ]["coords_prototipos"]

    # Concatenar todos los valores de 'coords_prototipos' (ignorando None si los hay)
    # Primero, aplanamos la lista de listas (o arrays)
    coords_concat = []
    for sublist in df_porcion:
        if sublist is not None:
            coords_concat.extend([x for x in sublist if x is not None])

    # Si quieres un np.array:
    coordenadas_array = np.array(coords_concat)

    #print(len(coordenadas_array))

    nuevo_grupo = grupos[0]//2

    cluster = procesa_particion(coordenadas_array, capa, nuevo_grupo, n_centroides, dist_func)

    return cluster


def create_tree(training_set, tam_grupo, n_centroides, dist_func, algorithm, implementation):

    n_total = training_set.shape[0] # Número total de puntos
    columnas = [f"dim_{i}" for i in range(training_set.shape[1])] # Nombres de columnas para el DataFrame

    # Paso 1: Crear Dask Array con particiones consecutivas
    dask_array = da.from_array(training_set, chunks=(tam_grupo, training_set.shape[1]))

    # Paso 2: Convertir a Dask DataFrame
    training_set_ddf = dd.from_dask_array(dask_array, columns=columnas)

    # Paso 3: crear un índice artificial y usarlo directamente como índice
    indice = da.arange(n_total, chunks=tam_grupo)
    training_set_ddf["__row__"] = indice
    training_set_ddf = training_set_ddf.set_index("__row__", sorted=True)

    # (opcional) Comprobar divisiones para verificar particiones consecutivas
    # print(training_set_ddf.divisions)

    # (Opcional, para verificar divisiones)
    #print("Divisiones conocidas:", training_set_ddf.known_divisions)
    #print("Divisions:", training_set_ddf.divisions)
    #training_set = training_set.repartition(divisions=training_set.divisions)

    # Obtener particiones individuales (como DataFrames Dask)
    n_partitions=len(training_set_ddf.divisions) - 1
    print(f"Number of groups (partitions) of the lowest level: {n_partitions}")

    particiones = [training_set_ddf.get_partition(i) for i in range(n_partitions)]

    # Componemos la capa más baja del indice PDASC
    capa = 0

    # Lista de objetos delayed
    tareas = [
        procesa_particion(particiones[i], capa, i, n_centroides=n_centroides, dist_func=dist_func)
        for i in range(n_partitions)
    ]

    # Ejecutar en paralelo y combinar los resultados
    resultados = dask.compute(*tareas)
    df_completo = pd.concat(resultados)

    #print(df_completo)

    # Agrupamos por capa y grupo para obtener los puntos de cada grupo
    grupos_capa=n_partitions
    while grupos_capa != 1:

        n_obj = df_completo[df_completo["capa"] == capa].shape[0]

        capa = capa + 1

        tareas = []
        for i in range(0, n_obj, 2):
            tareas.append(procesa_grupos(df_completo, capa, [i, i + 1], n_centroides=n_centroides, dist_func=dist_func))

        # Ejecutar en paralelo y combinar los resultados
        resultados = dask.compute(*tareas)
        df_capa = pd.concat(dask.compute(*resultados))
        # Y combinar los resultados de esta capa con el DataFrame final
        df_completo = pd.concat([df_completo,df_capa],ignore_index=True)

        # Calcular el número de grupos en la capa actual
        grupos_capa = len(df_capa["grupo"].unique())

    index=df_completo
    #print(f'Length Labels = {len(index[(index["capa"] == index["capa"].max()) & ((index["grupo"] == 0))]["labels"].values[0])}')
    # print(f'Length Coords = {len(index[index["capa"] == index["capa"].max() & (index["grupo"] == 0)]["coords_prototipos"].values[0])}')
    #print(f'Length Indices prototipos = {len(index[index["capa"] == index["capa"].max() & (index["grupo"] == 0)]["indices_prototipos"].values[0])}')

    # Print the resulting index (DataFrame)
    print(index)


    """
    # Mostrar el número de grupos en cada capa
    for capa, subdf in df_completo.groupby('capa'):
        num_grupos = subdf['grupo'].nunique()
        print(f"La capa {capa} tiene {num_grupos} grupos.")
    """
    return df_completo


def explore_centroid_dynamicradius_minradius(punto_buscado, index, n_centroides, prototype_id, prototype_distance, capa, grupo, rama, dist_function, min_radius):


    # print(f"Exploring prototype: capa={capa}, grupo={grupo}, prototype_id={prototype_id}, prototype_distance={prototype_distance}")

    lower_layer = capa - 1
    lower_group = grupo * 2 + rama
    neighbours = []
    distances_computed = 0

    # print(f'Estamos explorando el prototipo {prototype_id} que mapea la capa {lower_layer}, grupo {lower_group}')

    if lower_layer == 0:

        # print(f'Se ha llegado a la capa 0, vamos a explorar el grupo {lower_group} del dataset original')
        # Obtain the IDs of prototypes from the layer below
        childs = index[(index["capa"] == lower_layer) & (index["grupo"] == lower_group)]["labels"].values[0]
        #print(f"Mapping of prototypes in layer {origin_layer}, group {origin_group}: {id_childs}")

        # Obtain the prototypes of the layer below which are mapped by this prototype
        id_associated_childs = np.where(childs == prototype_id)[0]
        #print(f"Associated prototypes in layer {lower_layer} for prototype {prototype_id}: {id_associated_childs}")

        #print(prototype_id,prototype_distance, lower_layer, lower_group)
        # Lets take into account that at this point we do not restrict by radius, but explore all the points mapped by the current prototype
        for id_child in id_associated_childs:
            neighbour_id = n_centroides * 2 * lower_group + id_child

            #print(index[(index["capa"] == lower_layer) & (index["grupo"] == lower_group)]["indices_prototipos"].values[0])
            if id_child in index[(index["capa"] == lower_layer) & (index["grupo"] == lower_group)]["indices_prototipos"].values[0]:
                #print(f'Se simplifica porque el prototipo {id_child} figura como indice en {index[(index["capa"] == lower_layer) & (index["grupo"] == lower_group)]["indices_prototipos"].values[0]}')
                neighbours.append((neighbour_id, prototype_distance))
                #print("SIMPLIFICACION FINAL")
            else:
                neighbours.append(neighbour_id)

            # print(f'Neighbour found: {neighbour_id} with prototype distance: {associated_childs[i, 3]}')
            #neighbours.append(neighbour_id)

        return neighbours, distances_computed

    # If we are not in the bottom layer, we explore the prototypes of the layer below
    else:

        # Obtain the IDs of prototypes from the layer below
        # print(index[(index["capa"] == lower_layer) & ((index["grupo"] == lower_group))]["labels"])
        id_childs = index[(index["capa"] == lower_layer) & (index["grupo"] == lower_group)]["labels"].values[0]
        #print(f"Mapping of prototypes in layer {origin_layer}, group {origin_group}: {id_childs}")

        # Obtain the prototypes of the layer below which are mapped by this prototype
        position_associated_childs = np.where(id_childs == prototype_id)[0]

        #print(position_associated_childs)
        #print(f"Associated prototypes in layer {lower_layer} for prototype {prototype_id}: {position_associated_childs}")

        # Explore each associated prototype in the layer below and store it into a list
        associated_childs = np.empty((len(position_associated_childs), 5), dtype=object)

        for i in range(len(position_associated_childs)):
            associated_childs[i, 0] = position_associated_childs[i] % n_centroides
            #print(f'Prototype ID in the lower layer: {associated_childs[i, 0]}')
            associated_childs[i, 1] = lower_group   # Group ID in the lower layer
            #print(f'Group ID in the lower layer: {associated_childs[i, 1]}')
            #print(f'Position in the layer even below: {position_associated_childs[i]}')
            #print(f'Coordinates in the layer even below: {lower_layer-1}, grupo={(lower_group * 2 + position_associated_childs[i] // n_centroides)}')
            #associated_childs[i, 2] = index[(index["capa"] == lower_layer) & (index["grupo"] == lower_group)]["coords_prototipos"].values[0][i] # Coordinates of the prototype in the lower layer
            #print(index[(index["capa"] == lower_layer-1) & (index["grupo"] == (lower_group * 2 + position_associated_childs[i] // n_centroides))]["coords_prototipos"].values)
            associated_childs[i, 2] = index[(index["capa"] == lower_layer-1) & (index["grupo"] == (lower_group * 2 + position_associated_childs[i] // n_centroides))]["coords_prototipos"].values[0][associated_childs[i, 0]]
            # print(index[(index["capa"] == lower_layer-1) & (index["grupo"] == (lower_group + position_associated_childs[i] // n_centroides))]["coords_prototipos"].values[0][i])
            #print(f'Coordinates of the child {associated_childs[i, 0]} in layer {lower_layer-1}, grupo {lower_group + position_associated_childs[i] // n_centroides} son {associated_childs[i, 2]}')
            associated_childs[i, 3] = None # Distance to the query point, to be computed later
            associated_childs[i, 4] = position_associated_childs[i] // n_centroides # Branch ID in the lower even layer (0 or 1)

             # Se cualula la distancia del prototipo asociado al punto buscado
            if associated_childs[i, 0] + n_centroides * associated_childs[i, 4] in index[(index["capa"] == lower_layer) & (index["grupo"] == lower_group)]["indices_prototipos"].values[0]:
                #print(f'Se simplifica porque el prototipo {associated_childs[i, 0] + n_centroides * associated_childs[i, 4]}, perteneciente al grupo{lower_group} de la capa {lower_layer} figura como indice en {index[(index["capa"] == lower_layer) & (index["grupo"] == lower_group)]["indices_prototipos"].values[0]}')
                #print('SIMPLIFICACION')
                associated_childs[i, 3] = prototype_distance

            else:

                coordinates_child = np.vstack(associated_childs[i, 2])

                #if np.isnan(coordinates_bottomed_prototypes[i]).any():
                #    associated_childs[i, 3] = prototype_distance
                #else:
                associated_childs[i, 3] = get_distances(punto_buscado, coordinates_child.reshape(1, -1), dist_function)[0]
                # associated_childs[i, 3] = distance.pdist([punto_buscado[0], coordinates_bottomed_prototypes[i]], metric=metrica)[0]
                # associated_childs[i, 3] = pairwise_distances(punto_buscado, coordinates_bottomed_prototypes[i].reshape(1, -1), metric=metrica)[0][0]
                # distances_computed.append(associated_childs[i, 3])
                distances_computed += 1

        # We update the max_radius to be used according to the nearest prototype distance
        nearest_associated_prototype = np.min(associated_childs[:, 3])

        # Asi es segun la definición del paper:
        radius = np.maximum(min_radius, nearest_associated_prototype)

        # print(f'Radius value at this layer: {radius}')

        explorable_prototypes_indices = np.where(associated_childs[:, 3] <= radius)[0]
        explorable_prototypes = associated_childs[explorable_prototypes_indices]

        for i in range(len(explorable_prototypes)):
            centroid = explorable_prototypes[i]
            new_neighbours, new_distances_computed = explore_centroid_dynamicradius_minradius(punto_buscado, index, n_centroides, centroid[0], centroid[3], lower_layer, lower_group, centroid[4], dist_function, min_radius)
            neighbours.extend(new_neighbours)
            distances_computed += new_distances_computed

    #return None, None
    return neighbours, distances_computed

def recursive_approximate_knn_search_radius_pruning(punto, dataset, vector_training, index, k_vecinos, dist_function, radius):

    # Update the metric name for compatibility with scipy
    if dist_function == 'manhattan':
        dist_function = 'cityblock'  # scipy cdist requires 'cityblock' instead of 'manhattan'

    # Creamos las estructuras para almacenar los futuros vecinos
    #indices_vecinos = np.empty(k_vecinos, dtype=int)
    #coords_vecinos = np.empty([k_vecinos, punto.shape[1]], dtype=float)
    #dists_vecinos = np.empty(k_vecinos, dtype=float)
    # Creamos una estructura para almacenar el número de distancias computadas
    #n_distances = np.empty(1, dtype=int)

    # Y el número de distancias calculadas en cada ejecución
    #n_distances = np.empty([1], dtype=int)


    # Establish the query point
    punto_buscado = punto.reshape(1, -1)
    n_centroids = len(index[index["capa"] == index["capa"].max() & (index["grupo"] == 0)]["coords_prototipos"].values[0])


    # We take the top-layer prototypes, including its coordinates and distances to the query point
    coordinates_top_prototypes = np.vstack(index[(index["capa"] == index["capa"].max()) & (index["grupo"] == 0)]["coords_prototipos"].values[0])
    #print(f"Coordinates of top prototypes: {coordinates_top_prototypes}")
    distances_top_prototypes = get_distances(np.array(punto_buscado), coordinates_top_prototypes, dist_function)
    # print(f"Distances to top prototypes: {distances_top_prototypes}")
    n_distances_computed = len(distances_top_prototypes)

    # Number of top prototypes
    n_top_prototypes = len(distances_top_prototypes)
    #print(f"Number of top prototypes: {n_top_prototypes}")
    # Top prototypes ids=
    top_prototypes_id = np.arange(n_top_prototypes)
    # print(f"Top prototypes ids: {top_prototypes_id}")

    neighbours = []
    capa_max= index["capa"].max() + 1

    tareas = [
        dask.delayed(explore_centroid_dynamicradius_minradius(
            punto_buscado,
            index,
            n_centroids,
            prototype_id,
            prototype_distance,
            capa_max,
            0,
            0,
            dist_function,
            radius)
        )
        for prototype_id, prototype_distance in zip(top_prototypes_id, distances_top_prototypes)
    ]
    resultados = dask.compute(*tareas)

    # resultados: [(indices1, coords1, vec1), (indices2, coords2, vec2), ...]
    aux_neighbors, aux_n_distances = zip(*resultados)

    # Flatten the list of neighbours and count the distances computed
    aux_neighbors = [item for sublist in aux_neighbors for item in sublist]
    aux_n_distances= sum(aux_n_distances)
    #print(aux_neighbors)

    # print("found neighbours:", len(aux_neighbors))
    neighbours.extend(aux_neighbors)

    # print(f"Total neighbours found so far: {len(neighbours)}")
    n_distances_computed += aux_n_distances
    # print(f"Total distances computed so far: {n_distances_computed}")

    #print(aux_neighbors)



    ######## Version antigua funcional
    """
    for prototype_id in top_prototypes_id:
        #print(f"Exploring TOP prototype {prototype_id}")
        # aux_neighbors, aux_n_distances = explore_centroid_CDFradius1(punto_buscado, n_capas, inheritage, prototype_id, prototype_coords, puntos_capa, labels_capa, grupos_capa, promoted_points,n_centroides, metrica, [], 0, min_radius)
        # print(distances_top_prototypes[prototype_id])
        aux_neighbors, aux_n_distances = explore_centroid_dynamicradius_minradius(punto_buscado, index, n_centroids, prototype_id, index["capa"].max() + 1, 0, 0, distances_top_prototypes[prototype_id], dist_function, radius)
        #print("found neighbours:", len(aux_neighbors))
        neighbours.extend(aux_neighbors)
        #print(f"Total neighbours found so far: {len(neighbours)}")
        n_distances_computed += aux_n_distances
        #print(f"Total distances computed so far: {n_distances_computed}")

        #print(aux_neighbors)
        #print("")
    """
    ###################################

    # If no neighbours have been found:
    if not neighbours:

        # print("No neighbours have been found for this query point")

        # Pad the array of close points with None objects until it reaches the size of k neighbors
        # To avoid index out of bounds error
        vacio = np.empty(k_vecinos, dtype=int), np.empty([k_vecinos, vector_training.shape[0]],
                                                        dtype=float), np.empty(k_vecinos, dtype=float), n_distances_computed

        print(f'vacio:{vacio}')
        return vacio
    # If any neighbour have been found:
    else:

        # print(f'{len(neighbours)} neighbours have been found for this query point')

        # The neighbours whose distance is already computed are those which are stored as tuples
        neighbours_with_d = [n for n in neighbours if isinstance(n, tuple)]
        #print(f'There are {len(neighbours_with_d)} which distance is already computed')

        # Separate tuple\_neighbours into two sublists: one for the ids and one for the distances to the query point
        id_neighbours_with_d = [n[0] for n in neighbours_with_d]
        distances_neighbours_with_d = [n[1] for n in neighbours_with_d]

        # By acceding the original dataset, we obtain its coordinates
        coords_neighbours_with_d = vector_training[id_neighbours_with_d]

        # For control, print the distances already computed
        # print(f'The distances computed until now are {len(distances_computed)}')

        # The neighbours whose distance is not computed yet are those which are not tuples
        id_neighbours_without_d = [n for n in neighbours if not isinstance(n, tuple)]
        #print(f'There are {len(id_neighbours_without_d)} which distance is not computed yet')

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

        # And we store the info about each neighbour together into a single structure
        k_neighbours = np.empty((len(k_neighbours_ids), 3), object)

        k_neighbours[:, 0] = k_neighbours_ids
        k_neighbours[:, 1] = list(k_neighbours_coords)
        k_neighbours[:, 2] = k_neighbours_dists

        k_neighbours = np.vstack(k_neighbours)
        #print(k_neighbours)

        # To be able to find the k nearest

        # Create the structures to store the data related to the neighbors
        indices_k_vecinos = np.empty(k_vecinos, dtype=int)
        coords_k_vecinos = np.empty([k_vecinos, vector_training.shape[1]], dtype=float)
        dists_k_vecinos = np.empty(k_vecinos, dtype=float)

        # Drop the neighbors whose distance does not meet the condition (dist<radius)
        k_neighbours = k_neighbours[k_neighbours[:, 2] <= radius]

        # Sort them according to their distance to the query point
        sorted_neighbours = k_neighbours[k_neighbours[:, 2].argsort()]

        # Select the minimum value between k_vecinos and the number of neighbours founded
        minimum = min(k_vecinos, sorted_neighbours.shape[0])

        # Select the k closest points as neighbors (using vectorised operations and avoiding the loop
        indices_k_vecinos[:minimum] = sorted_neighbours[:minimum, 0]

        if minimum == 0:
            # fill every element in the arry with a vector_original.shape[1] vector of None
            coords_k_vecinos = np.full([k_vecinos, vector_training.shape[1]], None)
            dists_k_vecinos = np.full(k_vecinos, None)
        else:

            #print(sorted_neighbours[:minimum, 1])
            coords_k_vecinos[:minimum, :] = np.vstack(sorted_neighbours[:minimum, 1])
            dists_k_vecinos[:minimum] = sorted_neighbours[:minimum, 2]

        # Print them
        # print(f"The neighbours are: {indices_k_vecinos} with distances {dists_k_vecinos}")

        # And return the results
        # print(f"The search process computes a total of {n_distances_computed} distances")
        indices_vecinos = indices_k_vecinos
        coords_vecinos = coords_k_vecinos
        dists_vecinos = dists_k_vecinos
        n_distances = n_distances_computed

    #print(punto)
    #print(indices_vecinos, coords_vecinos, dists_vecinos, n_distances)

    return indices_vecinos, coords_vecinos, dists_vecinos, n_distances

def distributed_knn_search(vector_testing, dataset, vector_training, index, k, dist_function, radius):

    # Create a list to store the results
    results = []

    # Iterate over each point in the testing set
    for punto in vector_testing:
        # Call the recursive knn search function for each point
        indices, coords, distances, n_distances = recursive_approximate_knn_search_radius_pruning(
            punto, dataset, vector_training, index, k, dist_function, radius)
        results.append((indices, coords, distances, n_distances))

    # Unzip the results into separate lists
    indices_vecinos, coords_vecinos, dists_vecinos, n_distances = zip(*results)

    return indices_vecinos, coords_vecinos, dists_vecinos, n_distances


"""
if __name__ == "__main__":

    dataset=("municipios")
    dist_function = "haversine"
    k = 10

    tg = 60
    nc = 30
    r = 0.11


    dataset="MNIST"
    dist_function = "euclidean"
    k = 10

    tg = 1000
    nc = 500
    r = 0.99



    algorithm = 'kmedoids'
    implementation = 'fastkmedoids'

    # Load train and test datasets
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"
    vector_training, vector_testing = load_train_test_h5py(file_name)

    # If distance is haversine, convert data to radians
    if dist_function == 'haversine':
        vector_training = np.radians(vector_training)
        vector_testing = np.radians(vector_testing)

    # If distance is cosine, normalize the vectors
    elif dist_function == 'cosine':
        vector_training = normalize(vector_training, axis=1, norm='l2')
        vector_testing = normalize(vector_testing, axis=1, norm='l2')

    # And generate the index
    index = create_tree(vector_training, tg, nc, dist_function, algorithm, implementation)

    # Print the resulting index (DataFrame)
    #print(index)

    # 3rd - We search the k neighbors of the testing points
    # while measuring the time spent
    #start_time_s = timer()
    #print(f"Solo vamos a buscar el punto {vector_testing[:1]}")
    for i in vector_testing[:1]:
        indices, coords, distances, n_dist = recursive_approximate_knn_search_radius_pruning(i, dataset, vector_training, index, k, dist_function, r)

    #end_time_s = timer()
    #search_time = end_time_s - start_time_s

    # indices, coords, distances, n_dist = recursive_approximate_knn_search_radius_pruning(vector_testing[:1], dataset, vector_training, index, k, dist_function, r)

    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    file_name = "./benchmarks/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(k) + "_" + dist_function+ "_PDASC_tg" + str(tg) + "_nc" + str(nc) + "_r" + str(r) + "_kmedoids_fastkmedoids.hdf5"

    # Store indices, coords and dist into a hdf5 file
    save_neighbors_and_performance(indices, coords, distances, n_dist, 0, file_name)

"""



