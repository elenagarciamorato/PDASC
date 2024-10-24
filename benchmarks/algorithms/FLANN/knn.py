from data.load_train_test_set import *
from benchmarks.neighbors_utils import *
from benchmarks.algorithms.FLANN.module import FLANN_nn_index, FLANN_nn_search
import logging
from timeit import default_timer as timer

def FLANN(config_file):

    # Read config file containing experiment's parameters
    dataset, k, distance, method, n_centroides, algorithm = read_config_file(config_file)

    # Set log configuration
    logging.basicConfig(filename="./benchmarks/logs/" + dataset + "/test_knn_"  + dataset + "_" + str(k) + "_" + distance + "_" + method + ".log",
                        filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
    logging.info('------------------------------------------------------------------------')
    logging.info('                          plotting Searching')
    logging.info('------------------------------------------------------------------------\n')
    logging.info("")
    logging.info("---- Searching the " + str(k) + " nearest neighbors within " + method + " over " + str(
        dataset) + " dataset using " + str(distance) + " distance. ----")



    # Regarding the dataset name, set the file name to load the train and test set
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"

    # Load the train and test sets of each dataset to carry on the benchmarks
    # train_set, test_set = load_train_test(str(dataset))
    train_set, test_set = load_train_test_h5py(file_name)


    # GENERATE INDEX AND CENTROIDS
    # AND FIND THE plotting FROM THE train_set OF THE ELEMENTS CONTAINED IN THE test_set, USING DISTANCE CHOOSEN


    # Using FLANN, build the index tree and generate the num_centroids describing the data
    start_time_i = timer()
    centroids = FLANN_nn_index(train_set, n_centroides, distance, algorithm)
    end_time_i = timer()
    logging.info('Index time= %s seconds',  end_time_i - start_time_i)

    # Using FLANN and the index built, search for the knn nearest neighbors
    start_time_s = timer()
    indices, coords, dists, n_distances = FLANN_nn_search(train_set, test_set, k, distance, algorithm)
    end_time_s = timer()

    search_time = end_time_s - start_time_s

    logging.info('Search time = %s seconds\n', search_time)
    logging.info('Average time spent in searching a single point = %s', search_time/test_set.shape[0])
    logging.info('Speed (points/s) = %s\n', test_set.shape[0]/search_time)

    # Store indices, coords and dist into a tridimensional matrix of size vector.size() x 3 x knn
    # knn = zip(indices, coords, dists)

    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    file_name = "./benchmarks/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(k) + "_" + distance + "_" + method + ".hdf5"

    # Store indices, coords and dist into a hdf5 file
    save_neighbors_and_performance(indices, coords, dists, n_distances, search_time, file_name)


    # Print
    # print_knn(train_set, test_set, coords, dataset_name, d, "FLANN", k)

    '''
    # Obtain error rate of the K Nearest Neighbors found
    file_name_le = "./benchmarks/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(k) + "_" + distance + "_BruteForce.hdf5"
    file_name = "./benchmarks/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(k) + "_" + distance + "_" + method + ".hdf5"
    
    error_rate(dataset, distance, 'FLANN', k, False, file_name_le, file_name)
    '''

    logging.info('------------------------------------------------------------------------\n')
