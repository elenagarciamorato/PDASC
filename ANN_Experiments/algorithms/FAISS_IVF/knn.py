from data.load_train_test_set import *
import joblib
from ANN_Experiments.neighbors_utils import *
from ANN_Experiments.algorithms.FAISS_IVF.module import FaissIVF
import faiss
from timeit import default_timer as timer

def FAISS_IVF(config_file):

    # Read config file containing experiment's parameters
    dataset, k, metric, method, nlist, nprobe = read_config_file(config_file)
    print(f"The nlist is {nlist} and nprobe is {nprobe}")

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

    # Time and memory on behalf of the index construction
    start_time_fit = timer()
    mem_before = memory_usage_mb()

    faiss_ivf = FaissIVF(metric, nlist, nprobe).IVF_nn_index(train_set)

    # Time and memory after the index construction
    mem_after = memory_usage_mb()
    end_time_fit = timer()


    # Save the index to a file
    os.makedirs(f'ANN_Experiments/NearestNeighbors/{dataset}/indexes', exist_ok=True) # If the directory does not exist, create it
    path = f"./ANN_Experiments/NearestNeighbors/{dataset}/indexes/IVF_{dataset}_{metric}_nlist{nlist}_index.joblib"

    # Store the index using the customised function (in a joblib format))
    # store_index(faiss_ivf.index, path)

    # Store the index using FAISS built-in function (in a binary format)
    #print(type(faiss_ivf.index))
    #print(faiss_ivf.index.__class__)
    faiss.write_index(faiss_ivf.index, path)
    

    # Configurar nprobe (número de clusters a explorar durante la búsqueda)
    # n_probe = 10  # Puedes ajustar este valor o leerlo del config
    # faiss_index.set_query_arguments(n_probe)
    
    logging.info('Indexing time= %s seconds', end_time_fit - start_time_fit)
    print(f"Memory usage during index construction: {mem_after - mem_before:.2f} MB")
    # logging.info('Configured nprobe= %s', n_probe)

    # Search for k nearest neighbors in test set
    start_time_search = timer()
    dists, indices, n_distances = faiss_ivf.IVF_nn_search(test_set, k)
    end_time_search = timer()

    search_time = end_time_search - start_time_search

    # Get index size by storing it temporarily on disk
    index_size = get_index_size(dataset, 'IVF', metric, {'nlist': nlist})
    # Drop from disk the file located in path containing the index to save space
    os.remove(path)

    # Get index size by calculating the size of the object in memory
    # index_size = round(mem_after - mem_before, 2)
    
    # Get coordinates of the neighbors
    coords = train_set[indices]
    logging.info('Search time = %s seconds\n', search_time)
    logging.info('Average time spent in searching a single point = %s', search_time / test_set.shape[0])
    logging.info('Speed (points/s) = %s\n', test_set.shape[0] / search_time)

    print("The mean is")
    print(np.mean(n_distances))
    print(f"The nlist is {nlist} and nprobe is {nprobe}")


    # Store indices, coords and dist into a hdf5 file
    file_name = f"./ANN_Experiments/NearestNeighbors/{dataset}/knn_{dataset}_{k}_{metric}_{method}_nlist{nlist}_nprobe{nprobe}.hdf5"
    
    save_neighbors_and_performance(indices, coords, dists, n_distances, search_time, index_size, file_name)



