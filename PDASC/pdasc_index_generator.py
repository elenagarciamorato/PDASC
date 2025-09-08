import os, sys, argparse
from PDASC.pdasc_ import *
from ANN_Experiments.neighbors_utils import read_config_file
from sklearn.preprocessing import normalize

import data.load_train_test_set as lts


def PDASC_index_genenator(dataset, optional_filters=None):

    # PDASC_config_files = [f for f in os.listdir("./ANN_Experiments/config/" + dataset) if f.endswith('.ini') and 'PDASC' in f]


    # Check if the argument correspond to a directory
    if os.path.isdir("./ANN_Experiments/config/" + dataset):
        PDASC_config_files = [f for f in os.listdir("./ANN_Experiments/config/" + dataset) if f.endswith('.ini') and 'PDASC' in f]

        # Apply optional filter if provided
        if optional_filters:
            filter_options = optional_filters
            PDASC_config_files = [f for f in PDASC_config_files if all(opt in f for opt in filter_options)]
    else:
        # Print usage message and exit if the argument is invalid
        print("Usage: ./pdasc_index_generator.sh [dataset name] [optional_filter]")
        sys.exit(22)

    for config_file in PDASC_config_files:

        # Read config file containing experiment's parameters
        dataset, k, distance, method, tam_grupo, n_centroides, initial_radius, algorithm, implementation = read_config_file(config_file)

        #Load train and test datasets
        file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"
        vector_training, vector_testing = lts.load_train_test_h5py(file_name)

        # If distance is haversine, convert data to radians
        if distance == 'haversine':
            vector_training = np.radians(vector_training)

        # If distance is cosine, normalize the vectors
        if distance == 'cosine':
            vector_training = normalize(vector_training, axis=1, norm='l2')

        # And generate the index
        n_capas, grupos_capa, puntos_capa, labels_capa, promoted_points = create_tree(vector_training, tam_grupo, n_centroides, distance, algorithm, implementation)

        print(f"Number of layers: {n_capas}")
        for i in range(len(grupos_capa)):
            print(f"Layer {i}: {len(grupos_capa[i])} groups")
        print(f"Groups in each layer: {grupos_capa}")
        #print(f"Points in each layer: {puntos_capa}")
        #print(f"Labels in each layer: {labels_capa}")
        print(labels_capa[8])

        #print(len(puntos_capa[len(puntos_capa)-1][0]))
        #print(grupos_capa)

        # Store the index built by PDASC in a file
        #store_PDASC_index(dataset, distance, grupos_capa, puntos_capa, labels_capa)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Name of the dataset to be indexed", type=str)
    parser.add_argument("optional_filters", help="Optional filters to apply", nargs='*', default=[])

    args = parser.parse_args()

    PDASC_index_genenator(args.dataset, args.optional_filters)