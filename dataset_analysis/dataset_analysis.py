import PDASC.pdasc_flues_ as pdasc_flues_
from ANN_Experiments.algorithms.Exact.knn import Exact_nn_search
from data.load_train_test_set import load_train_test_h5py
from scipy.stats import skew, kurtosis
from scipy.spatial import distance
import numpy as np
import logging


# Load a random sample of the dataset
def load_random_sample(dataset, sample_size):
    # Load the random sample of the dataset
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"
    vector_training, vector_testing = load_train_test_h5py(file_name)

    # We take a sample of n random elements from the dataset
    np.random.seed(42)  # Fijar la semilla para reproducibilidad
    sample = vector_training[np.random.choice(len(vector_training), sample_size, replace=True)]

    return sample

def load_random_sample_flue(partition, sample_size):

    # We take a sample of n random elements from a dataset partition (flue)
    np.random.seed(42)  # Fijar la semilla para reproducibilidad
    sample = partition[np.random.choice(len(partition), sample_size, replace=True)]

    return sample

# Load the prototype points composing a layer of the PDASC index corresponding to a flue
def load_PDASC_sample(dataset, sample_size, distance_function, n_flues=1, id_flue=1):

    # Load the PDASC index
    index_flue = pdasc_flues_.load_PDASC_index_flue(dataset, distance_function, n_flues, id_flue)

    n_capas = index_flue[0]
    grupos_capa = index_flue[1]
    puntos_capa = index_flue[2]
    labels_capa = index_flue[3]
    promoted_points = index_flue[4]

    # Reconstruct the puntos_capa from the labels_capa
    for i in range(len(grupos_capa) - 2, -1, -1):
        for j in range(len(grupos_capa[i])):
            puntos = puntos_capa[i][j]
            # print(f'Layer {i}, Group {j}: {len(puntos)} points')

            for k in range(len(puntos)):
                if np.all(np.isnan(puntos[k])):
                    label_punto = labels_capa[i + 1][j // 2][k]
                    # print(f"Label of that point: {label_punto}")
                    puntos_capa[i][j][k] = puntos_capa[i + 1][j // 2][label_punto]

    # print(grupos_capa)

    # Regarding the sample_size, we identify the layer whose points we will analyse
    # Initialize variables to store the index and the closest sum
    closest_index = -1
    closest_sum = 0

    # Regarding the sample_size, we identify the layer whose points we will analyse
    if sample_size >= np.sum(grupos_capa[0]):
        closest_index = 1
    else:
        # Explore the size of each layer to identify the closest one

        # Iterate over the elements of grupos_capa
        for i, subarray in enumerate(grupos_capa):
            current_sum = np.sum(subarray)
            if sample_size >= current_sum > closest_sum:
                closest_sum = current_sum
                closest_index = i

    # Print the result
    print(
        f"The dataset has {len(grupos_capa)} layers, and the layer with the number of elements closest to {sample_size} without exceeding it is: {closest_index}")
    # print(f"The sum of the subelements of this element is: {closest_sum}")

    # We take all the points from the desired layer
    # (Lets take into account that the layer of puntos_capa equivalent to lablels_capa is one below)
    puntos_capa_concatenado = np.vstack(puntos_capa[closest_index - 1])
    print(f"Number of points in the layer {closest_index}: {len(puntos_capa_concatenado)}")

    # These prototypes will constitute the sample to be analysed
    return puntos_capa_concatenado

# Compute the distance between every element in a random sample of the dataset and the other elements
def compute_distances_pairwise(set, distance_function):

    # if sample elements are of type float64, convert them to float32
    if set.dtype == np.float64:
        set = set.astype(np.float32)

    # Calculate the pairwise distances between every element on the dataset using Linear Scan
    indices, coords, dists, n_dist = Exact_nn_search(set, set, len(set), distance_function, None, False)

    return dists

# Compute the distance between every element in a random sample of the dataset and its k-th nearest neighbour in the given complete dataset
def compute_distances_kth_nn(subset, complete_set, k, distance_metric):

    # if sample elements are of type float64, convert them to float32
    if subset.dtype == np.float64:
        subset = subset.astype(np.float32)
        complete_set = complete_set.astype(np.float32)

    # Calculate the distances between every element on the dataset and their knn using Linear Scan
    indices, coords, dists, n_dist = Exact_nn_search(complete_set, subset, k, distance_metric, None, False)

    # Get the the distance to the k-th nearest neighbour of each element
    kth_nn_dists = dists[:, k -1]

    return kth_nn_dists

# Perform a descriptive analysis of a given dataset. It includes the analysis of the dataset's dimensions and distances
## To be reimplemented
def descriptive_analysis(dataset, distances):

    # Set log configuration
    log_file = f"./dataset_analysis/{dataset}/analisis_descriptivo_{dataset}.log"
    logging.basicConfig(filename=log_file, filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
    logging.info('------------------------------------------------------------------------')
    logging.info(f'             {dataset} Dataset Descriptive Analysis ')
    logging.info('------------------------------------------------------------------------\n')
    logging.info("")

    # Regarding the dataset name, set the file name to load the train and test set
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"

    # Read data
    vector_training, vector_testing = load_train_test_h5py(file_name)

    # Log size and dimensionality of the dataset
    logging.info(f"Size: {vector_training.shape[0]}\n")
    n_dimensiones = vector_training.shape[1]
    logging.info(f"Dimensionality: {n_dimensiones}\n")

    # Initialize lists for each dimension's statistics
    da_minmax = []
    da_range = []
    da_medias = []
    da_medianas = []
    da_std = []
    da_cv = []
    da_kur = []
    da_asimetrias = []
    da_dist = []

    # If vector_training has 10,000 or more elements, take a random 10% sample
    if len(vector_training) >= 10000:
        sample_size = int(len(vector_training) * 0.1)
        vector_training = vector_training[np.random.choice(len(vector_training), sample_size, replace=False)]

    # Else, use the whole dataset

    # Explore every dimension of the dataset to get relevant statistics
    for i in range(0, vector_training.shape[1]):

        dimension = vector_training[:, i]
        print(f"Dimension {i} analysis")

        '''
        # Dibujar histograma (resultan una distribucion normal)
        if i == 12:
            plot= plt.hist(dimension, bins='auto')
            plt.show()
        '''

        # Get min and max value
        minmax = (min(dimension), max(dimension))
        da_minmax.append(minmax)

        # Get range of the values
        range = (minmax[1]-minmax[0])
        da_range.append(range)

        # Get mean
        media = np.mean(dimension, axis=0)
        da_medias.append(media)

        # Get median
        mediana = np.median(dimension, axis=0)
        da_medianas.append(mediana)

        # Get standard deviation
        std = np.std(dimension, axis=0)
        da_std.append(std)

        # Coefficient of variation doesn't report relevant information cause, as mean is near 0 (0,1),
        # cv would tend to infinite, so we won't use it
        # cv = variation(dimension, ddof=0)  # Also calculated as cv=std/media
        # da_cv.append(cv)

        # Get kurtosis
        kur = kurtosis(dimension)
        da_kur.append(kur)

        # Get skewness
        asimetria = skew(dimension)
        da_asimetrias.append(asimetria)

        # Get distance between the point values in an specific dimension
        # As it's a 1-d analysis, distance choosen has no impact. We use euclidean for simplicity
        dist = distance.pdist(dimension.reshape(-1, 1), metric='euclidean')
        mean_dist = np.sum(dist)/dist.size
        da_dist.append(mean_dist)

    # Log the descriptive analysis for each dimension of the dataset
    logging.info("------- Descriptive analysis for each dimension of the dataset-------\n")
    logging.info(f"MinMax: {da_minmax}\n")
    logging.info(f"Range: {da_range}\n")
    logging.info(f"Mean value: {da_medias}\n")
    logging.info(f"Median value: {da_medianas}\n")
    logging.info(f"Standard Deviation: {da_std}\n")
    logging.info(f"Kurtosis: {da_kur}\n")
    logging.info(f"Skewness (Asimetria): {da_asimetrias}\n")
    logging.info(f"Mean distance between points (1-d): {da_dist}\n")

def localIntrinsicDimensionality(dataset, sample_size, distance_function):

    # LID para cada punto
    def compute_lid_row(row):
        r_k = row[-1]
        return -1.0 / np.mean(np.log(row / r_k + 1e-12))  # suma pequeña para evitar log(0)

    # Load a random sample of the dataset
    sample = load_random_sample(dataset, sample_size)

    # Load the complete dataset to compute the k-th nearest neighbour distances
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"
    vector_training, vector_testing = load_train_test_h5py(file_name)

    # Compute the k-th nearest neighbour distances
    k = 10  # Number of neighbours to consider
    kth_nn_dists = compute_distances_kth_nn(sample, vector_training, k, distance_function)
    pairwise_nn_dists = compute_distances_pairwise(sample, distance_function)

    if distance_function == 'haversine':
        # Convierte de radianes a kilómetros (radio de la Tierra ≈ 6371 km)
        knn_distances_km = kth_nn_dists * 6371
        pairwise_distances_km = pairwise_nn_dists * 6371

    lids = np.apply_along_axis(compute_lid_row, 1, pairwise_distances_km)
    lid_mean = lids.mean()

    print(f"LID promedio para el dataset {dataset}: {lid_mean:.3f}")

    return lid_mean
