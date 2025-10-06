#import PDASC.pdasc_ as pdasc
from timeit import default_timer as timer
import data.load_train_test_set as lts
from PDASC import pdasc_
from PDASC import pdasc_flues_
# from PDASC import pdasc_DataFrames_
from ANN_Experiments.neighbors_utils import *
from sklearn.preprocessing import normalize


def PDASC(config_file):

    # Read config file containing experiment's parameters
    dataset, k, distance, method, group_size, n_centroids, n_nodes, radius, algorithm, implementation = read_config_file(config_file)


    # Check if the method choosen are valid:
    if algorithm not in pdasc_.PDASC_accepted_algorithms():
        print("The algorithm choosen is not valid. Please, check the PDASC documentation and try again.")
        exit(2)

    # Print information about the experiment in the log file
    logging.info('------------------------------------------------------------------------')
    logging.info("---- Searching the " + str(k) + " nearest neighbors within " + method + " over " + str(
        dataset) + " dataset using " + str(distance) + " distance. ----")
    logging.info("")
    logging.info('---- PDASC Parameters - group_size=%s - n_centroids=%s - radius=%s - algorithm=%s - implementation=%s ----', group_size, n_centroids, radius, algorithm, implementation)
    logging.info('------------------------------------------------------------------------\n')

    # Regarding the dataset name, set the file name to load the train and test set
    file_name = f"./data/{dataset}_train_test_set.hdf5"


    # 1st - We read the dataset to be used

    # Read train and test set from preprocesed h5py file
    vector_training, vector_testing = lts.load_train_test_h5py(file_name)

    # If distance is haversine, convert data to radians
    if distance == 'haversine':
        vector_training = np.radians(vector_training)
        vector_testing = np.radians(vector_testing)

    # If distance is cosine, normalize the vectors
    elif distance == 'cosine':
        vector_training = normalize(vector_training, axis=1, norm='l2')
        vector_testing = normalize(vector_testing, axis=1, norm='l2')

    # Read train and test set from original file
    # vector_training, vector_testing = lts.load_train_test(str(dataset))

    # Make a np array considering the first 10 elements of vector_testing
    #vector_testing = np.array(vector_testing[:1])

    # Selecciona aleatoriamente el 10% de los puntos de vector_training
    # vector_training = vector_training[np.random.choice(vector_training.shape[0], size=int(vector_training.shape[0]*0.1), replace=False)]
    # Selecciona aleatoriamente el 10% de los puntos de vector_testing
    # vector_testing = vector_testing[np.random.choice(vector_testing.shape[0], size=int(vector_testing.shape[0]*0.01), replace=False)]

    #Selecciona el primer punto de vector_testing
    #vector_testing = vector_testing[:1]

    # 2nd - We build the tree

    # By using the updated implementation
    # n_capas, grupos_capa, puntos_capa, labels_capa, promoted_points = pdasc_.create_tree(vector_training, group_size, n_centroids, distance, algorithm, implementation)

    # And store the index built by PDASC in a file
    # store_PDASC_index(dataset, distance, grupos_capa, puntos_capa, labels_capa)

    # By using the distributed implementation

    # Check if all index files already exist
    all_exist = True
    index_time = 0
    path = f"./ANN_Experiments/NearestNeighbors/{dataset}/indexes/"
    for node in range(n_nodes):
        filename = f"{dataset}_{distance}_index_{n_nodes}-{node}.joblib"
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            all_exist = False
            break  # no need to check further if one is missing

    # If they don't exist, create them
    if not all_exist:

        start_time_i = timer()
        print("[INFO] Some index files are missing, creating all with create_index...")
        index = pdasc_flues_.create_index_flues(vector_training, dataset, group_size, n_centroids, n_nodes, distance,
                                                algorithm, implementation)
        end_time_i = timer()
        index_time = end_time_i - start_time_i
    # By using the DataFrame experimental implementation
    #index = pdasc_DataFrames_.create_tree(vector_training, dataset, group_size, n_centroids, n_nodes, distance, algorithm, implementation)
    # print(index)
    # Print number of layers
    # print(f'Number of layers = {n_capas}')

    # 3rd - We search the k neighbors of the testing points
    # while measuring the time spent
    start_time_s = timer()

    # print(f"Solo vamos a buscar el punto {vector_testing[:1]}")
    # vector_testing = vector_testing[:1]  # Uncomment this line to test with only the first point
    # indices_vecinos, coords_vecinos, dists_vecinos, n_distances = pdasc.recursive_approximate_knn_search(n_capas, n_centroids, vector_testing, vector_training, k, distance, grupos_capa, puntos_capa, labels_capa, promoted_points, float(initial_radius), dataset)
    # indices_vecinos, coords_vecinos, dists_vecinos, n_distances = pdasc.recursive_approximate_knn_search_classical_pruning(n_capas, n_centroids, vector_testing, vector_training, k, distance, grupos_capa, puntos_capa, labels_capa, promoted_points, float(initial_radius), dataset)

    # By using the updated implementation
    #indices_vecinos, coords_vecinos, dists_vecinos, n_distances = pdasc_.recursive_approximate_knn_search_radius_pruning(n_capas, n_centroids, vector_testing, vector_training, k, distance, grupos_capa, puntos_capa, labels_capa, promoted_points, float(radius), dataset)

    # By using the distributed (flues) implementation
    indices_vecinos, coords_vecinos, dists_vecinos, n_distances = pdasc_flues_.distributed_ANN_search(vector_testing, vector_training, dataset, n_nodes, n_centroids, distance, radius, k)

    # By using the experimental DataFrame implementation
    #indices_vecinos, coords_vecinos, dists_vecinos, n_distances = pdasc_DataFrames_.distributed_ANN_search(vector_testing, dataset, vector_training, n_nodes, distance, float(radius), k)
    end_time_s = timer()

    # Obtain search time and print information about it in the log file
    search_time = end_time_s - start_time_s

    # Get the size of the index in MB
    index_size = get_index_size(dataset, 'PDASC', distance, {'n_nodes': n_nodes})

    logging.info('Search time = %s seconds\n', search_time)
    logging.info('Average time spent in searching a single point = %s', search_time/vector_testing.shape[0])
    logging.info('Speed (points/s) = %s\n', vector_testing.shape[0]/search_time)

    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    file_name = f"./ANN_Experiments/NearestNeighbors/{dataset}/knn_{dataset}_{k}_{distance}_{method}_tg{group_size}_nc{n_centroids}_r{radius}_n{n_nodes}.hdf5"

    # Store indices, coords and dist into a hdf5 file
    save_neighbors_and_performance(indices_vecinos, coords_vecinos, dists_vecinos, n_distances, index_size, index_time, search_time, file_name)


    logging.info("\n")

