import GDASC.gdasc_ as gdasc
from timeit import default_timer as timer
import data.load_train_test_set as lts
from benchmarks.neighbors_utils import *
from sys import getsizeof


def GDASC(config_file):

    # Read config file containing experiment's parameters
    dataset, k, distance, method, tam_grupo, n_centroides, initial_radius, algorithm, implementation = read_config_file(config_file)

    # Print information about the experiment in the log file
    logging.info('------------------------------------------------------------------------')
    logging.info("---- Searching the " + str(k) + " nearest neighbors within " + method + " over " + str(
        dataset) + " dataset using " + str(distance) + " distance. ----")
    logging.info("")
    logging.info('---- GDASC Parameters - tam_grupo=%s - n_centroids=%s - radius=%s - algorithm=%s - implementation=%s ----', tam_grupo, n_centroides, initial_radius, algorithm, implementation)
    logging.info('------------------------------------------------------------------------\n')

    # Regarding the dataset name, set the file name to load the train and test set
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"


    # 1º Leemos los datos

    # Read train and test set from preprocesed h5py file
    vector_training, vector_testing = lts.load_train_test_h5py(file_name)

    # Read train and test set from original file
    # vector_training, vector_testing = lts.load_train_test(str(dataset))

    # 2º Generamos el árbol

    # Updated implementation
    n_capas, grupos_capa, puntos_capa, labels_capa, promoted_points = gdasc.create_tree(vector_training, tam_grupo, n_centroides, distance, algorithm, implementation)

    # Print number of layers and a brief comment
    #print(f'Number of layers = {n_capas}')

    # Size of the index
    #print(getsizeof(puntos_capa) + getsizeof(labels_capa) + getsizeof(grupos_capa) + getsizeof(n_capas))

    # Store index in a file
    # with open("./algorithms/GDASC_MNIST" + str(k) +".pickle", 'wb') as handle:
    #    dump((puntos_capa, labels_capa), handle)

    # 3º Buscamos los k vecinos de los puntos de testing
    start_time_s = timer()

    # Creamos las estructuras para almacenar los futuros vecinos
    indices_vecinos = np.empty([len(vector_testing), k], dtype=int)
    coords_vecinos = np.empty([len(vector_testing), k, vector_testing.shape[1]], dtype=float)
    dists_vecinos = np.empty([len(vector_testing), k], dtype=float)

    # Y el número de distancias calculadas en cada ejecución
    n_distances = np.empty([len(vector_testing)], dtype=int)


    # For every point in the testing set, find its k nearest neighbors
    for i in range(len(vector_testing)):

        punto = vector_testing[i]
        #logging.info('punto %s =%s', i,  punto)

        start_time_iter = timer()

        vecinos_i, n_distances_i = gdasc.recursive_approximate_knn_search(n_capas, n_centroides, punto, vector_training,
                                                                          k, distance, grupos_capa, puntos_capa,
                                                                          labels_capa, promoted_points, float(initial_radius), dataset)

        # Legacy implementation
        #vecinos_i = gdasc.knn_approximate_search(n_centroides, punto, vector_training, k, distance, grupos_capa, puntos_capa, labels_capa, dimensionalidad, float(radio))

        end_time_iter = timer()
        #logging.info('Index time= %s seconds', end_time_iter - start_time_iter)
        #logging.info('punto %s - time= %s seconds', i, end_time_iter - start_time_iter)

        indices_vecinos[i] = vecinos_i[0]
        coords_vecinos[i] = vecinos_i[1]
        dists_vecinos[i] = vecinos_i[2]

        n_distances[i] = n_distances_i


    end_time_s = timer()
    search_time = end_time_s - start_time_s
    logging.info('Search time = %s seconds\n', search_time)
    logging.info('Average time spent in searching a single point = %s', search_time/vector_testing.shape[0])
    logging.info('Speed (points/s) = %s\n', vector_testing.shape[0]/search_time)

    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    file_name = "./benchmarks/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(k) + "_" + distance + "_" + method + "_tg" + str(tam_grupo) + "_nc" + str(n_centroides) + "_r" + str(initial_radius) + "_" + str(algorithm) + "_" + str(implementation) + ".hdf5"

    # Store indices, coords and dist into a hdf5 file
    save_neighbors_and_performance(indices_vecinos, coords_vecinos, dists_vecinos, n_distances, search_time, file_name)

    logging.info("\n")