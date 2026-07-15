from data.load_train_test_set import load_train_test_h5py, load_hdf5
from ANN_Experiments.neighbors_utils import *
from ANN_Experiments.algorithms.NMSLIB_HNSW.module import NmslibHNSW
from timeit import default_timer as timer

def NMSLIB_HNSW(exp_parameters):

    # Process experiment's parameters
    dataset = exp_parameters["dataset"]
    method = exp_parameters["method"]
    k = exp_parameters["k"]
    metric = exp_parameters["distance"]

    # NMSLIB HNSW parameters
    M = exp_parameters["M"]
    efConstruction = exp_parameters["efConstruction"]
    efSearch = exp_parameters["efSearch"]
    post = exp_parameters["post"]
    coords_in_degrees = exp_parameters["coords_in_degrees"]

    # Print information about the experiment in the log file
    logging.info('------------------------------------------------------------------------')
    logging.info("---- Searching the " + str(k) + " nearest neighbors within " + method + " over " + str(
        dataset) + " dataset using " + str(metric) + " distance. ----")
    logging.info('------------------------------------------------------------------------\n')

    # Regarding the dataset name, set the file name to load the train and test set
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"

    # Load the train and test sets of each dataset to carry on the benchmarks
    if metric == "jaccard":
        train_set, test_set = load_hdf5(file_name)
    else:
        # Load the train and test sets of each dataset to carry on the benchmarks
        train_set, test_set = load_train_test_h5py(file_name)

    # Fit the index with training data
    start_time_fit = timer()
    nmslib_index = NmslibHNSW(metric, M, efConstruction, efSearch, post, coords_in_degrees).HNSW_nn_index(train_set)
    end_time_fit = timer()

    index_time = end_time_fit - start_time_fit

    # Save the index to a file
    os.makedirs(f'ANN_Experiments/NearestNeighbors/{dataset}/indexes', exist_ok=True)  # If the directory does not exist, create it
    path = f"./ANN_Experiments/NearestNeighbors/{dataset}/indexes/NMSLIBHNSW_{dataset}_{metric}_M{M}_efC{efConstruction}_index.joblib"
    path_aux= f"./ANN_Experiments/NearestNeighbors/{dataset}/indexes/NMSLIBHNSW_{dataset}_{metric}_M{M}_efC{efConstruction}_index.joblib.dat"

    # Store the index using NMSLIB built-in function (in a binary format)
    nmslib_index.index.saveIndex(path, save_data=True)
    
    
    logging.info('Indexing time= %s seconds', end_time_fit - start_time_fit)

    # Search for k nearest neighbors in test set
    start_time_search = timer()
    indices, coords, dists, n_distances = NmslibHNSW.HNSW_nn_search(nmslib_index, metric, train_set, test_set, k)
    end_time_search = timer()

    search_time = end_time_search - start_time_search

    # Get index size
    index_size = get_index_size(dataset, 'NMSLIBHNSW', metric, {'M': M, 'efConstruction': efConstruction})

    # Drop from disk the file located in path containing the index to save space
    os.remove(path)
    os.remove(path_aux)

    # Get coordinates of the neighbors
    #coords = train_set[indices]
    print(f"Search time for {search_time} seconds")
    logging.info('Search time = %s seconds\n', search_time)
    logging.info('Average time spent in searching a single point = %s', search_time / test_set.shape[0])
    logging.info('Speed (points/s) = %s\n', test_set.shape[0] / search_time)

    # Store indices, coords and dist into a hdf5 file
    file_name = f"./ANN_Experiments/NearestNeighbors/{dataset}/knn_{dataset}_{k}_{metric}_{method}_M{M}_efC{efConstruction}_efS{efSearch}.hdf5"
    save_neighbors_and_performance(indices, coords, dists, n_distances, index_size, index_time, search_time, file_name)
