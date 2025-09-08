from data.load_train_test_set import *
import joblib
import faiss
from ANN_Experiments.neighbors_utils import * #read_config_file
from ANN_Experiments.algorithms.FAISS_HNSW.module import FaissHNSW
from timeit import default_timer as timer

def FAISS_HNSW(config_file):
    # Read config file containing experiment's parameters
    dataset, k, metric, method, M, efConstruction, efSearch = read_config_file(config_file)

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
    faiss_index = FaissHNSW(metric, M, efConstruction, efSearch).HNSW_nn_index(train_set)
    end_time_fit = timer()

    logging.info('Indexing time= %s seconds', end_time_fit - start_time_fit)

    # Save the index to a file
    path = f"./ANN_Experiments/NearestNeighbors/{dataset}/indexes/FAISSHNSW_{dataset}_{metric}_M{M}_efC{efConstruction}_index.joblib"

    # Store the index using the customised function (in a joblib format))
    # store_index(faiss_index.index, path)

    # Store the index using FAISS built-in function (in a binary format)
    #print(type(faiss_index.index))
    #print(faiss_index.index.__class__)
    faiss.write_index(faiss_index.index, path)

    # Search for k nearest neighbors in test set
    start_time_search = timer()
    #TODO => n_distances que sea la lista de distancias por cada query
    # todo => forzar que las busquedas se hagan una a una
    dists, indices, n_distances = FaissHNSW.HNSW_nn_search(faiss_index, test_set, k)
    end_time_search = timer()

    search_time = end_time_search - start_time_search

    # Get index size
    index_size = get_index_size(dataset, 'FAISSHNSW', metric, {'M': M, 'efConstruction': efConstruction})
    # Drop from disk the file located in path containing the index to save space
    os.remove(path)
    
    # Get coordinates of the neighbors
    coords = train_set[indices]
    print(f"Search time for {search_time} seconds")
    logging.info('Search time = %s seconds\n', search_time)
    logging.info('Average time spent in searching a single point = %s', search_time / test_set.shape[0])
    logging.info('Speed (points/s) = %s\n', test_set.shape[0] / search_time)

    # Store indices, coords and dist into a hdf5 file
    file_name = f"./ANN_experiments/NearestNeighbors/{dataset}/knn_{dataset}_{k}_{metric}_{method}_M{M}_eC{efConstruction}_eS{efSearch}.hdf5"
    save_neighbors_and_performance(indices, coords, dists, n_distances, search_time, index_size, file_name)

