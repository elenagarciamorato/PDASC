from data.load_train_test_set import *
from benchmarks.neighbors_utils import *
from benchmarks.algorithms.Exact.module import Exact_nn_index, Exact_nn_search
from timeit import default_timer as timer


def Exact(config_file):

    # Read config file containing experiment's parameters
    dataset, k, distance, method, exact_algorithm = read_config_file(config_file)

    logging.info('------------------------------------------------------------------------')
    logging.info(" Searching the " + str(k) + " nearest neighbors within " + method + " - " + exact_algorithm
                 + " algorithm over " + str(dataset) + " dataset using " + str(distance) + " distance. ----")
    logging.info('------------------------------------------------------------------------\n')

    # Regarding the dataset name, set the file name to load the train and test set
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"

    # Load the train and test sets to carry on the benchmarks
    # train_set, test_set = load_train_test(str(dataset))
    vector_training, vector_testing = load_train_test_h5py(file_name)

    # GENERATE INDEX AND CENTROIDS
    # AND FIND THE plotting FROM THE train_set OF THE ELEMENTS CONTAINED IN THE test_set, USING DISTANCE CHOOSEN

    # Using Exact Algorithm, build the index of nearest neighbors
    start_time_i = timer()
    knn_index = Exact_nn_index(vector_training, distance, exact_algorithm)
    end_time_i = timer()
    logging.info('Index time= %s seconds', end_time_i - start_time_i)

    # Store index on disk to obtain its size
    # with open("./algorithms/Exact/" + dataset + str(k) +".pickle", 'wb') as handle:
    #    dump(knn_index, handle)

    # Using Exact Algorithm and the index built, find the k nearest neighbors
    start_time_s = timer()

    # Build the arrays to store the indices, coordinates and distances of the k nearest neighbors
    indices_vecinos = np.empty([len(vector_testing), k], dtype=int)
    coords_vecinos = np.empty([len(vector_testing), k, vector_testing.shape[1]], dtype=float)
    dists_vecinos = np.empty([len(vector_testing), k], dtype=float)

    # And number of distances computed
    n_distances = np.empty(len(vector_testing))

    # For every point in the testing set, find its k nearest neighbors
    for i in range(len(vector_testing)):
        punto = vector_testing[i].reshape(1, -1)
        # logging.info('punto %s =%s', i,  punto)

        indices, coords, dists, n_dist = Exact_nn_search(vector_training, punto, k, distance, knn_index)

        indices_vecinos[i] = indices[0]
        coords_vecinos[i] = coords[0]
        dists_vecinos[i] = dists[0]
        n_distances[i] = n_dist


    end_time_s = timer()

    search_time = end_time_s - start_time_s

    logging.info('Search time = %s seconds\n', search_time)
    logging.info('Average time spended in searching a single point = %s',
                 search_time / vector_testing.shape[0])
    logging.info('Speed (points/s) = %s\n', vector_testing.shape[0] / search_time)

    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    file_name = "./benchmarks/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(k) + "_" + distance + "_" + method + "_" + exact_algorithm + ".hdf5"

    # Store indices, coords and dist into a hdf5 file
    save_neighbors_and_performance(indices_vecinos, coords_vecinos, dists_vecinos, n_distances, search_time, file_name)

    # Print
    # print_knn(train_set, test_set, coords, dataset_name, d, "Exact", k)

    logging.info("\n")
