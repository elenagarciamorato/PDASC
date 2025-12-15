from ANN_Experiments.neighbors_utils import *
from data.load_train_test_set import load_train_test_h5py
from ANN_Experiments.algorithms.Annoy.module import annoy_knn
from timeit import default_timer as timer
#from benchmarks.plotting.performance_utils import *


def Annoy(config_file):
    # Read config file containing experiment's parameters
    
    dataset, k, metric, method, n_trees, k_search = read_config_file(config_file)

    # Check if metric is valid
    valid_metrics = {"angular", "euclidean", "cosine","manhattan", "hamming", "dot"}

    if metric not in valid_metrics:
        raise ValueError(
            f"Metric not accepted: '{metric}'. "
            f"Accepted metrics are: {', '.join(sorted(valid_metrics))}."
        )
    # Print information about the experiment in the log file
    logging.info('------------------------------------------------------------------------')
    logging.info("---- Searching the " + str(k) + " nearest neighbors within " + method + " over " + str(
        dataset) + " dataset using " + str(metric) + " distance. ----")
    logging.info('------------------------------------------------------------------------\n')

    # Regarding the dataset name, set the file name to load the train and test set
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"

    # Load the train and test sets of each dataset to carry on the benchmarks
    train_set, test_set = load_train_test_h5py(file_name)

    
    # Fit the index with training data
    start_time_i = timer()
    dim = train_set.shape[1]
    annoy_index = annoy_knn(n_trees, k_search, metric).annoy_nn_index(train_set)
    
    # (train_set.shape[1], n_trees=n_trees, metric=metric).annoy_nn_index(train_set)
    end_time_i = timer()
    index_time = end_time_i - start_time_i
    
    logging.info('Indexing time= %s seconds', end_time_i - start_time_i)

    # Store index on disk
    os.makedirs(f'ANN_Experiments/NearestNeighbors/{dataset}/indexes', exist_ok=True) # If the directory does not exist, create it
    path = f"./ANN_Experiments/NearestNeighbors/{dataset}/indexes/ANNOY_{dataset}_{metric}_nn{k}_ntrees{n_trees}_ksearch{k_search}_index.joblib"
    annoy_index_pickeable = annoy_index.pickleable_index(dataset)
    store_index(annoy_index_pickeable, path)

    # Search for k nearest neighbors in test set
    start_time_search = timer()
    indices, dists, n_distances = (annoy_index.annoy_nn_search(test_set, k))
    end_time_search = timer()

    search_time = end_time_search - start_time_search

    # Get index size
    index_size = get_index_size(dataset, 'ANNOY', metric, {'n_neighbors': k, 'n_trees': n_trees, 'k_search': k_search})
    # Drop from disk the file located in path containing the index to save space
    os.remove(path)

    # Get coordinates of the neighbors
    coords = train_set[indices]
    logging.info('Search time = %s seconds\n', search_time)
    logging.info('Average time spent in searching a single point = %s', search_time / test_set.shape[0])
    logging.info('Speed (points/s) = %s\n', test_set.shape[0] / search_time)

    
    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    file_name = f"./ANN_Experiments/NearestNeighbors/{dataset}/knn_{dataset}_{k}_{metric}_{method}_ntrees{n_trees}_ksearch{k_search}.hdf5"

    # Store indices, coords and dist into a hdf5 file
    save_neighbors_and_performance(indices, coords, dists, n_distances, index_size, index_time, search_time, file_name)

    
    