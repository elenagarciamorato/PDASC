from data.load_train_test_set import *
from ANN_Experiments.neighbors_utils import *
from ANN_Experiments.algorithms.FAISS_LSH.module import FaissLSH
from timeit import default_timer as timer
import joblib
import faiss

def FAISS_LSH(config_file):
    # Read config file containing experiment's parameters
    dataset, k, metric, method, nbits = read_config_file(config_file)

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
    start_time_fit = timer()
    faiss_lsh = FaissLSH(metric, nbits).LSH_nn_index(train_set)
    end_time_fit = timer()

    logging.info('Indexing time= %s seconds', end_time_fit - start_time_fit)

    # Save the index to a file
    os.makedirs(f'ANN_Experiments/NearestNeighbors/{dataset}/indexes', exist_ok=True) # If the directory does not exist, create it
    path = f"./ANN_Experiments/NearestNeighbors/{dataset}/indexes/LSH_{dataset}_{metric}_index.joblib"

    # Store the index using the customised function (in a joblib format))
    # store_index(faiss_lsh.index, path)

    # Store the index using FAISS built-in function (in a binary format)
    # print(type(faiss_lsh.index))
    # print(faiss_lsh.index.__class__)
    faiss.write_index(faiss_lsh.index, path)

    # Search for k nearest neighbors in test set
    start_time_search = timer()
    dists, indices, n_distances = faiss_lsh.LSH_nn_search(test_set, k)
    end_time_search = timer()

    search_time = end_time_search - start_time_search

    # Get index size
    index_size = get_index_size(dataset, 'LSH', metric, np.nan)
    # Drop from disk the file located in path containing the index to save space
    os.remove(path)

    # Get coordinates of the neighbors
    coords = train_set[indices]
    logging.info('Search time = %s seconds\n', search_time)
    logging.info('Average time spent in searching a single point = %s', search_time / test_set.shape[0])
    logging.info('Speed (points/s) = %s\n', test_set.shape[0] / search_time)



    # Store indices, coords and dist into a hdf5 file
    file_name = f"./ANN_Experiments/NearestNeighbors/{dataset}/knn_{dataset}_{k}_{metric}_{method}_nbits{nbits}.hdf5"
    
    save_neighbors_and_performance(indices, coords, dists, n_distances, search_time, index_size, file_name)



