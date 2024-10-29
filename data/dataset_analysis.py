from data.load_train_test_set import *
from benchmarks.algorithms.Exact.module import Exact_nn_search
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.spatial import distance
import data.load_train_test_set as lts


# Perform a descriptive analysis of a given dataset. It includes the analysis of the dataset's dimensions and distances
def descriptive_analysis(dataset, distances):

    # Set log configuration
    log_file = f"./benchmarks/logs/{dataset}/analisis_descriptivo_{dataset}.log"
    logging.basicConfig(filename=log_file, filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
    logging.info('------------------------------------------------------------------------')
    logging.info(f'             {dataset} Dataset Descriptive Analysis ')
    logging.info('------------------------------------------------------------------------\n')
    logging.info("")

    # Regarding the dataset name, set the file name to load the train and test set
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"

    # Read data
    vector_training, vector_testing = lts.load_train_test_h5py(file_name)

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

    # Make an analysis of the distances between elements for every chosen distance metric
    distance_analysis(vector_training, distances)


# Analysis of distances between elements for every chosen distance metric
def distance_analysis(vector_training, distances):

    logging.info("\n------------ Distance Matrixes (built using several distance metrics)-----------\n")

    for d in distances:
        print(f"-- Distance analysis using {d} distance --")

        # If the distance is 'manhattan', we change it to 'cityblock' to use scipy
        if d == 'manhattan':
            d = 'cityblock'

        # Distance Matrix - distance between every point in the dataset. Calculation using scipy
        distances = np.array(distance.pdist(vector_training, metric=d))

        # Min and max distance between points (calculated over a flattened version of the distances matrix)
        minmax_distances = (np.min(distances), np.max(distances))

        # Mean distance between points
        mean_dist_distances = np.sum(distances) / distances.size

        # Quantiles
        q1_distances = np.quantile(distances, 0.25)
        q2_distances = np.quantile(distances, 0.5)
        q3_distances = np.quantile(distances, 0.75)

        logging.info(f"\n-------- {d} distance --------\n")
        logging.info(f"MinMax distance: {minmax_distances}\n")
        logging.info(f"Mean distance between points (all-d): {mean_dist_distances}\n")
        logging.info(f"Quantiles:  q1={q1_distances}  -  q2={q2_distances}  -  q3={q3_distances}")


# Plot the distribution of pairwise distances between the elements composing the dataset
def distances_distribution_plot(dataset):

    # Print a title for the analysis
    print(f"-- Analysis of the {k}th neighbour distances --")

    # Regarding the dataset name, set the file name to load the train and test set
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"
    vector_training, vector_testing = load_train_test_h5py(file_name)

    # If vector_training is bigger than 10000, we take a sample of 10% of the data
    if len(vector_training) > 1000:
        sample1 = vector_training[np.random.choice(len(vector_training), int(vector_training.shape[0]/10), replace=True)]
        sample2 = vector_training[np.random.choice(len(vector_training), int(vector_training.shape[0]/10), replace=True)]

        # Calculate the pairwise distances between every element on the dataset using Linear Scan
        indices, coords, dists, n_dist = Exact_nn_search(sample1, sample2, len(sample2), 'euclidean', None, False)

    # If vector_training is smaller than 10000, we calculate the pairwise distances between every element on the dataset
    else:
        sample = vector_training

        # Calculate the pairwise distances between every element on the dataset using Linear Scan
        indices, coords, dists, n_dist = Exact_nn_search(sample, sample, len(sample)-1, 'euclidean', None, True)

    # Flatten the distances matrix to get the number of distance computations
    distances = dists.flatten()

    # Obtain the medium distance
    mean_distance = np.mean(distances)
    print(f"Mean distance: {mean_distance}")

    # Obtain the median distance
    median_distance = np.median(distances)
    print(f"Median distance: {median_distance}\n")

    # Plot the distribution of pairwise distances (histogram)
    plt.hist(distances, bins=30, edgecolor='black')
    plt.title('Distribution of Pairwise Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.show()

# Plot the distribution of the distances regarding the kth neighbour of each element in the dataset
def neighbours_distribution_plot(dataset, k):

    # Print a title for the analysis
    print(f"-- Analysis of the {k}th neighbour distances --")

    # Regarding the dataset name, set the file name to load the train and test set
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"
    vector_training, vector_testing = load_train_test_h5py(file_name)

    # Y el número de distancias calculadas en cada ejecución
    n_distances = np.empty(len(vector_testing))

    # Obtenemos los k vecinos mas cercanos

    dists = []

    # If vector_training is bigger than 10000, we take a sample of 10% of the data
    if len(vector_training) > 1000:
        sample1 = vector_training[np.random.choice(len(vector_training), int(vector_training.shape[0]/10), replace=True)]
        sample2 = vector_training[np.random.choice(len(vector_training), int(vector_training.shape[0]/10), replace=True)]

        # Calculate the pairwise distances between every element on the dataset using Linear Scan
        indices, coords, dists, n_dist = Exact_nn_search(sample1, sample2, len(sample2), 'euclidean', None, False)

        # The fifth neighbours are those with index 4
        kth_neighbour = dists[:, k-1]

    # If vector_training is smaller than 10000, we calculate the pairwise distances between every element on the dataset
    else:
        sample = vector_training

        # Calculate the pairwise distances between every element on the dataset using Linear Scan
        indices, coords, dists, n_dist = Exact_nn_search(sample, sample, len(sample)-1, 'euclidean', None, True)

        # The fifth neighbours are those with index 5 (as the first one is the element itself)
        kth_neighbour = dists[:, k]

    # k th neighbour mean distance
    mean_distance = np.mean(kth_neighbour)
    print(f"{k}th neighbour mean distance: {mean_distance}")

    # k th neighbour median distance
    median_distance = np.median(kth_neighbour)
    print(f"{k}th neighbour median distance: {median_distance}")

    # Compute the 3rd quartile of fifth neighbour
    q3 = np.percentile(kth_neighbour, 75)
    print(f"Third quartile of {k}th neighbour: {q3}")

    # Compute the 90% percentile of fifth neighbour
    p90 = np.percentile(kth_neighbour, 90)
    print(f"90% percentile of {k}th neighbour: {p90}")

    # Plot the distribution of pairwise distances (histogram)
    plt.hist(kth_neighbour, bins=30, edgecolor='black')
    plt.title(f'Distribution of {k}th Neighbour Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == "__main__":

    dataset = 'wdbc'
    distances = ['euclidean', 'manhattan', 'chebyshev', 'cosine']
    k =5

    # Perform the descriptive analysis of the dataset
    # dataset_analysis(dataset, distances)

    # Plot the distribution of pairwise distances between the elements composing the dataset
    distances_distribution_plot(dataset)

    # Plot the distribution of the distances regarding the k-th neighbour of each element in the dataset
    neighbours_distribution_plot(dataset, k)

