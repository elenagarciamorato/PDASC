from data.load_train_test_set import *
from benchmarks.neighbors_utils import *
from benchmarks.algorithms.Pynndescent.module import PYNN_nn_index, PYNN_nn_search, PYNN_accepted_distances
from timeit import default_timer as timer

# PYNN algorithm admits the following distances:
# 'euclidean', 'l2', 'sqeuclidean', 'manhattan', 'taxicab', 'l1', 'chebyshev', 'linfinity', 'linfty', 'linf', 'minkowski', 'seuclidean', 'standardised_euclidean', 'wminkowski', 'weighted_minkowski', 'mahalanobis', 'canberra', 'cosine', 'dot', 'correlation', 'hellinger', 'haversine', 'braycurtis', 'spearmanr', 'kantorovich', 'wasserstein', 'tsss', 'true_angular', 'hamming', 'jaccard', 'dice', 'matching', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalsneath', 'sokalmichener', 'yule'

def PYNN(config_file):

    # Read config file containing experiment's parameters
    dataset, k, distance, method = read_config_file(config_file)

    # Check if the distance and choosen is valid:
    if distance not in PYNN_accepted_distances() :
        print("The distance is not valid. Please, check the PYNN documentation and try again.")
        exit(2)

    # Print information about the experiment in the log file
    logging.info('------------------------------------------------------------------------')
    logging.info("---- Searching the " + str(k) + " nearest neighbors within " + method + " over " + str(
        dataset) + " dataset using " + str(distance) + " distance. ----")
    logging.info('------------------------------------------------------------------------\n')

    # Regarding the dataset name, set the file name to load the train and test set
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"

    # Load the train and test sets to carry on the benchmarks
    # train_set, test_set = load_train_test(str(dataset))
    train_set, test_set = load_train_test_h5py(file_name)

    # If distance is haversine, convert data to radians
    if distance == 'haversine':
        train_set = np.radians(train_set)
        test_set = np.radians(test_set)

    # GENERATE INDEX AND CENTROIDS
    # AND FIND THE plotting FROM THE train_set OF THE ELEMENTS CONTAINED IN THE test_set, USING DISTANCE CHOOSEN

    # Using PYNN, build the index tree and generate the num_centroids describing the data
    start_time_i = timer()
    pynn_index = PYNN_nn_index(train_set, distance)
    end_time_i = timer()
    logging.info('Index time= %s seconds', end_time_i - start_time_i)

    # Store index on disk to obtain its size
    # with open("./algorithms/Pynndescent/" + dataset + str(k) +".pickle", 'wb') as handle:
    #    dump(pynn_index, handle)

    # Using PYNN and the index built, search for the knn nearest neighbors
    start_time_s = timer()
    indices, coords, dists, n_distances = PYNN_nn_search(train_set, test_set, k, distance, pynn_index)
    end_time_s = timer()

    search_time = end_time_s - start_time_s

    logging.info('Search time = %s seconds\n', search_time)
    logging.info('Average time spent in searching a single point = %s', search_time/test_set.shape[0])
    logging.info('Speed (points/s) = %s\n', test_set.shape[0]/search_time)

    # Store indices, coords and dist into a tridimensional matrix of size vector.size() x 3 x knn
    # knn = zip(indices, coords, dists)

    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    file_name = "./benchmarks/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(
        k) + "_" + distance + "_" + method + ".hdf5"

    # Store indices, coords and dist into a hdf5 file
    save_neighbors_and_performance(indices, coords, dists, n_distances, search_time, file_name)

    # Print
    # print_knn(train_set, test_set, coords, dataset_name, d, "PYNN", k)

    '''
    # Obtain error rate of the K Nearest Neighbors found
    file_name_le = "./benchmarks/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(k) + "_" + distance + "Exact.hdf5"
    file_name = "./benchmarks/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(k) + "_" + distance + "_" + method + ".hdf5"
    
    error_rate(dataset, distance, 'FLANN', k, False, file_name_le, file_name)
    '''

    logging.info("\n")
