import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import csv
import logging
import data.load_train_test_set as load_train_test_set
import re
import configparser
import psutil
import joblib


# Store query points and its neighbors on a csv file
def save_csv(filename, train_test_file, neighbors_file):
    # arguments: file name, train_test hdf5 file, neighbors file
    # save_csv("./benchmark/municipios_5_euclidean_FLANN", "./data/municipios_train_test_set.hdf5", "./ANN_Experiments/NearestNeighbors/municipios/knn_municipios_5_euclidean_FLANN.hdf5")

    with open(str(filename) + ".csv", 'w') as file:
        writer = csv.writer(file)
        header = ['index', 'query_point', 'neighbors']
        writer.writerow(header)

        train_test, test_set = load_train_test_set.load_train_test_h5py(train_test_file)
        indices_n, coords_n, dists_n = load_neighbors(neighbors_file)


        num_neighbors = re.split('_|\.',  neighbors_file)[3]

        for i in range(0, len(test_set)):
            writer.writerow([i, test_set[i], str(coords_n[i].tolist()).replace(",", "")])


# Store only coordinates on a csv file
def save_coordinates_csv(filename, coords):
    with open(str(filename) + ".csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerows(coords)


# Store neighbors (indices, coords and dist) into a hdf5 file
def save_neighbors(indices, coords, dists, file_name):

    # Store the 3 different matrix on a hdf5 file
    with h5py.File(file_name, 'w') as f:
        f.flush()
        dset1 = f.create_dataset('indices', data=indices)
        dset2 = f.create_dataset('coords', data=coords)
        dset3 = f.create_dataset('dists', data=dists)
        print("Neighbors stored at " + file_name)
        logging.info("Neighbors stored at " + file_name)
        f.close()


# Load neighbors (indices, coords and dist) from a hdf5 file
def load_neighbors(file_name):

    # If the filepath provided does not exist, return None
    if not os.path.exists(file_name):

        print("File " + file_name + " does not exist")
        logging.info("File " + file_name + " does not exist\n")

        return None, None, None

    # If the file exists
    else:

        # Load the indices, coords and dists from the chosen file as 3 independent arrays
        with h5py.File(file_name, 'r') as hdf5_file:

            print(f"Loading neighbors from {file_name}")
            logging.info(f"Loading neighbors from {file_name}")

            return np.array(hdf5_file['indices']), np.array(hdf5_file['coords']), np.array(hdf5_file['dists'])

# Store neighbors (indices, coords and dist) and performance (n_distances, search time and index_size) into a hdf5 file
def save_neighbors_and_performance(indices, coords, dists, n_distances, search_time, index_size, file_name):

    # Store the 3 different matrix on a hdf5 file
    with h5py.File(file_name, 'w') as f:
        f.flush()
        dset1 = f.create_dataset('indices', data=indices)
        dset2 = f.create_dataset('coords', data=coords)
        dset3 = f.create_dataset('dists', data=dists)

        dset4 = f.create_dataset('n_distances', data=n_distances)

        dset5 = f.create_dataset('search_time', data=search_time)

        dset6 = f.create_dataset('index_size', data=index_size)


        print(f"Neighbors, distances computed, search time and index size stored at {file_name}")
        logging.info(f"Neighbors distances computed and search time and indez size stored at {file_name}")
        f.close()


# Load neighbors (indices, coords and dist) and performance (n_distances & search time) from a hdf5 file
def load_neighbors_performance(file_name):

    # If the filepath provided does not exist, return None
    if not os.path.exists(file_name):

        print(f"File {file_name} does not exist")
        logging.info(f"File {file_name} does not exist\n")

        return None, None, None, None, None, None

    # If the file exists
    else:
        # Load the indices, coords & dist for each neighbour, as well as the number of distance computation and search time required
        with h5py.File(file_name, 'r') as hdf5_file:

            print(f"Loading neighbors, computed distances and search time from {file_name}")
            logging.info(f"Loading neighbors, computed distances and search time from {file_name}")

            return np.array(hdf5_file['indices']), np.array(hdf5_file['coords']), np.array(hdf5_file['dists']), np.array(hdf5_file['n_distances']), np.array(hdf5_file['search_time']), np.array(hdf5_file['index_size'])



# Print train set, test set and neighbors on a file
def print_knn(train_set, test_set, neighbors, dataset_name, d, method, knn, file_name):

    # Plot with points, centroids and title
    fig, ax = plt.subplots()
    title = str(dataset_name) + "_" + str(d) + "_" + method + "_" + str(knn) + "nn"
    plt.title(title)

    train_set = zip(*train_set)
    test_set = zip(*test_set)

    ax.scatter(train_set[0], train_set[1], marker='o', s=1, color='#1f77b4', alpha=0.5)

    for point in neighbors:
        point = zip(*point)
        ax.scatter(point[0], point[1], marker='o', s=1, color='#949494', alpha=0.5)

    ax.scatter(test_set[0], test_set[1], marker='o', s=1, color='#ff7f0e', alpha=0.5)

    plt.savefig(file_name)
    print(f"Train set, test set and neighbors printed at {file_name}")

    return plt.show()

# Method to read an experiment described into a .ini file
def read_config_file(config_file):

    # Get the path of the configuration file provided by the user
    dataset = re.split('_|\.', config_file)[2]
    configfile_path = "./ANN_Experiments/config/" + dataset + "/" + config_file

    # Verify that config file provided as an argument exists
    if not os.path.exists(configfile_path):
        print(f"[ERROR] Config file {configfile_path} doesn't exist. Please check it and try again.")
        exit(2)
        #raise FileNotFoundError

    # If it does, launch the experiment
    print(f"--- Reading {config_file} ---")

    # Open the configuration file
    config = configparser.ConfigParser()
    config.read(configfile_path)

    # Read test parameters
    dataset = config.get('test', 'dataset')
    k = config.getint('test', 'k')
    distance = config.get('test', 'distance')
    method = config.get('test', 'method')

    # Read specific parameters of the choosen method
    if method == 'Exact':
        exact_algorithm = config.get('method', 'algorithm')
        parameters = [dataset, k, distance, method, exact_algorithm]

    elif method == 'PDASC':
        tam_grupo = config.getint('method', 'tg')
        n_centroides = config.getint('method', 'nc')
        n_nodes = config.getint('method', 'n_nodes')  # Number of parallel processing nodes to be used
        radius = float(config.get('method', 'r'))  # Radius of the neighborhood to be considered
        # radius = [float(r) for r in config.get('method', 'r').split(', ')] # If we want to use multiple radius values, we can pass them as a list
        algorithm = config.get('method', 'algorithm')  # Possible values kmeans, kmedoids. others to be defined
        implementation = config.get('method', 'implementation')  # Possible values:
        #                  for kmeans: sklearn, kclust
        #                  for kmedoids: sklearnextra, fastkmedoids

        parameters = [dataset, k, distance, method, tam_grupo, n_centroides, n_nodes, radius, algorithm, implementation]

    elif method == 'FLANN':
        ncentroids = config.getint('method', 'ncentroids')  # At PDASC, ncentroids = tam_grupo*n_centroides = 8*16 = 128
        algorithm = config.get('method','algorithm')  # Possible values: linear, kdtree, kmeans, composite, autotuned - default: kdtree

        parameters = [dataset, k, distance, method, ncentroids, algorithm]

    elif method == 'PYNN':
        # Query parameters
        n_neighbors = config.getint('method', 'n_neighbors')
        diversify_prob = config.getfloat('method', 'diversify_prob')
        pruning_degree_multiplier = config.getfloat('method', 'pruning_degree_multiplier')

        # Search parameters
        epsilon = config.getfloat('method', 'epsilon')  # Approximation factor

        parameters = [dataset, k, distance, method, n_neighbors, diversify_prob, pruning_degree_multiplier, epsilon]

    elif method == 'IVF':
        nlist = config.get('method', 'nlist')
        nprobe = config.get('method', 'nprobe')
        parameters = [dataset, k, distance, method, nlist, nprobe]

    elif method == 'LSH':
        nbits = config.get('method', 'nbits')
        parameters = [dataset, k, distance, method, nbits]

    elif method == 'FAISSHNSW':
        M = config.getint('method', 'M')
        efConstruction = config.getint('method', 'efConstruction')
        efSearch = config.getint('method', 'efSearch')
        parameters = [dataset, k, distance, method, M, efConstruction, efSearch]

        """
        elif method == 'NMSLIBHNSW':
            M = config.getint('test', 'M')
            efConstruction = config.getint('test', 'efConstruction')
            efSearch = config.getint('test', 'efSearch')
            post = config.getint('test', 'post')
            coords_in_degrees = config.getboolean('test', 'coords_in_degrees')
            parameters = [dataset, k, distance, method,
                          {"M": M, "efConstruction": efConstruction, "efSearch": efSearch, "post": post,
                           "coords_in_degrees": coords_in_degrees}]
        """
    else:
        print("Method not able")
        exit(1)

    return parameters

# Get the memory usage of the current process in MB
def memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)

# Store index using joblib
def store_index(index, path):
    # Store index on disk
    with open(path, "wb") as f:

        joblib.dump(index, f)
        # For more realistic calculation of index size on disk, we store it without compression
        #  joblib.dump(index, f, compress=0)

# Function to get the index size in MB
def get_index_size(dataset, method, distance, method_params):

    # Set the directory path to load the experiments according to the dataset provided
    directory_path = f"./ANN_Experiments/NearestNeighbors/{dataset}/indexes/"

    if method == 'PDASC':
        n_nodes = method_params['n_nodes']
        index_file = f"{str(dataset)}_{str(distance)}_index_{n_nodes}-{0}.joblib"

    elif method == 'PYNN':
        print('Getting PYNN index size')
        n_neighbors = method_params['n_neighbors']
        diversify_prob = method_params['diversify_prob']
        pruning_degree_multiplier = method_params['pruning_degree_multiplier']
        index_file = f"PYNN_{str(dataset)}_{str(distance)}_nn{n_neighbors}_div{diversify_prob}_pru{pruning_degree_multiplier}_index.joblib"
        #index_file = f"{str(method)}_{str(dataset)}_{str(distance)}_index.joblib"

    elif method == 'IVF':
        nlist = method_params['nlist']
        index_file = f"IVF_{str(dataset)}_{str(distance)}_nlist{nlist}_index.joblib"

    elif method == 'LSH':
        index_file = f"LSH_{str(dataset)}_{str(distance)}_index.joblib"

    elif method == 'FAISSHNSW':
        print('Getting FAISS HNSW index size')
        M = method_params['M']
        efConstruction = method_params['efConstruction']
        index_file = f"FAISSHNSW_{str(dataset)}_{str(distance)}_M{M}_efC{efConstruction}_index.joblib"
    else:

        index_file = f"{str(method)}_{dataset}_{distance}_index.joblib"

    full_path = directory_path + index_file

    if os.path.isfile(full_path):
        size_bytes = os.path.getsize(full_path)
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)
    else:
        return np.nan
