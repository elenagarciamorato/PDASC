import scipy

from data.load_train_test_set import *
from benchmarks.algorithms.Exact.module import Exact_nn_search
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.spatial import distance
import data.load_train_test_set as lts
import seaborn as sns
import argparse
import sklearn

from fitter import Fitter, get_common_distributions, get_distributions


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
def distance_analysis(vector_training, distance_metrics):

    logging.info(f"\n------------ {dataset} dataset Distance Matrixes (built using several distance metrics)-----------\n")

    for d in distance_metrics:
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


# Compute the distances between the elements composing the dataset (or a sample of it if too big)
def get_distances_between_elements(dataset, distance_metric):

    # Regarding the dataset name, set the file name to load the train and test set
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"
    vector_training, vector_testing = load_train_test_h5py(file_name)

    # If vector_training is bigger than 15000, we take a sample of 1500 ramdom elements of the data
    if len(vector_training) > 1500:
        sample1 = vector_training[np.random.choice(len(vector_training), int(1500), replace=True)]
        sample2 = vector_training[np.random.choice(len(vector_training), int(1500), replace=True)]

        # Calculate the pairwise distances between every element on the dataset using Linear Scan
        indices, coords, dists, n_dist = Exact_nn_search(sample1, sample2, len(sample2), distance_metric, None, False)

    # If vector_training is smaller than 10000, we calculate the pairwise distances between every element on the dataset
    else:
        sample = vector_training

        # Calculate the pairwise distances between every element on the dataset using Linear Scan
        indices, coords, dists, n_dist = Exact_nn_search(sample, sample, len(sample)-1, distance_metric, None, True)

    return dists


# Plot the distribution of pairwise distances between the elements composing the dataset
def distances_distribution_plot(dataset, distances, distance_metric):

    # Descriptive analysis of the distances
    print(f"\n-- Descriptive analysis of the pairwise {distance_metric} distances for the {dataset} dataset--")

    # Flatten the distances matrix to get a 1-d array containing all the pairwise distances
    distances = distances.flatten()

    # Obtain the medium distance
    mean_distance = np.mean(distances)
    print(f"Mean distance: {mean_distance}")

    # Obtain the median distance
    median_distance = np.median(distances)
    print(f"Median distance: {median_distance}")

    # Plot the distribution of pairwise distances (histogram)
    print("\n-- Generating the histogram defining the pairwise distances distribution --")
    plt.hist(distances, bins=50, edgecolor='black')
    plt.title('Distribution of Pairwise Distances')
    plt.xlabel(f'{distance_metric} Distance')
    plt.ylabel('Frequency')

    # Force x-axis to start at 0
    plt.xlim(left=0)


    # Show the plot
    plt.show()

    # Clear the plot
    plt.clf()

# Plot the probability density plot of the pairwise distances between the elements composing the dataset
def distances_probability_density_plot(dataset, distances, distance_metric):

    # Flatten the distances matrix to get a 1-d array containing all the pairwise distances
    distances = distances.flatten()

    # Print info about the fit
    print(f"-- Fitting the data to a distribution for {dataset} dataset--")
    # Fit the data to a distribution
    f = Fitter(distances, distributions=get_common_distributions(), timeout=120)
    f.fit()
    f.summary()

    # Print the best fitting distribution
    best_dist = f.get_best(method='sumsquare_error')
    print(f'\nThe best fitting distribution is {best_dist}')

    # Plot the data distribution and the best fitting distribution
    plt.title(f'Pairwise distances Distribution and Best Fitting Distribution \n (normalised) for {dataset} dataset')
    plt.xlabel(f'{distance_metric} Distance')
    plt.ylabel('Frequency')

    # Force x-axis to start at 0
    plt.xlim(left=0)

    # Store the plot
    plt.savefig(f'./benchmarks/logs/{dataset}/{dataset}_{distance_metric}_probability_density.png')

    # Show the plot
    # plt.show()

    # Clear the plot
    plt.clf()


# Plot the comulative distribution of pairwise distances between the elements composing the dataset
def distances_comulative_distribution_plot(dataset, distances, distance_metric):

    # Flatten the distances matrix to get a 1-d array containing all the pairwise distances
    distances = distances.flatten()

    # Plot the comulative distribution function of pairwise distances estimated through the KDE curve obtained from the comulative histogram
    # histogram = sns.histplot(distances, stat='percent', bins=30, kde=True, alpha=0, edgecolor=None, cumulative=True)

    # Plot the comulative distribution function of pairwise distances estimated through the KDE curve
    kde = sns.kdeplot(distances, cumulative=True)
    # plt.yticks(kde.get_yticks(), [f'{int(tick * 100)}%' for tick in kde.get_yticks()])

    plt.title(f'Comulative Distribution of Pairwise Distances\nfor {dataset} dataset')
    plt.xlabel(f'{distance_metric} Distance')
    plt.ylabel('Probability')

    # Get the data of the KDE curve
    kde_x = kde.get_lines()[0].get_data()[0]
    kde_y = kde.get_lines()[0].get_data()[1]

    # Define percentiles to be used
    percentiles = [0.7, 0.8, 0.9]

    for p in percentiles:

        # Calculate the x-coordinate for the given percentile using interpolation
        kde_percentile_x = np.interp(p, kde_y, kde_x)

        # Draw the horizontal line precisely to the KDE curve intersection
        plt.plot([0, kde_percentile_x], [p, p], color='r', linestyle='--')

        # Annotate the percentile on the y-axis at the intersection point
        plt.text(0, p, str(int(p*100)) + "%", color='r', ha='left', va='bottom')

        # Calculate the y-coordinate of the KDE at the intersection point
        kde_percentile_y = np.interp(kde_percentile_x, kde_x, kde_y)

        # Draw the vertical line from the bottom to the KDE curve
        plt.plot([kde_percentile_x, kde_percentile_x], [0, kde_percentile_y], color='r', linestyle='--')

        # Annotate the KDE value on the x-axis at the intersection point
        plt.text(kde_percentile_x, p, f'{kde_percentile_x:.2f}', color='black', ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))


    # Force x-axis to start at 0
    plt.xlim(left=0)

    # Store the plot
    plt.savefig(f'./benchmarks/logs/{dataset}/{dataset}_{distance_metric}_comulative_distribution.png')

    # Show the plot
    # plt.show()

    # Clear the plot
    plt.clf()

# Plot the distribution of the distances regarding the kth neighbour of each element in the dataset
def neighbours_distribution_plot(dataset, distances, distance_metric, k):

    # Print a title for the analysis
    print(f"\n-- Analysis of the {k}th neighbour distances --")


    # If vector_training is bigger than 1500, we take a sample of 1500 random elements of the data
    #if len(distances) > 1500:

        # The fifth neighbours are those with index 4
    kth_neighbour = distances[:, k-1]

    # If vector_training is smaller than 1500, we calculate the pairwise distances between every element on the dataset
    #else:
        # The fifth neighbours are those with index 5 (as the first one is the element itself)
    #    kth_neighbour = distances[:, k]

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
    plt.hist(kth_neighbour, bins=50, edgecolor='black')
    plt.title(f'Distribution of {k}th Neighbour Distances\nfor {dataset} dataset')
    plt.xlabel(f'{distance_metric} Distance')
    plt.ylabel('Frequency')

    # Force x-axis to start at 0
    plt.xlim(left=0)

    # Store the plot
    plt.savefig(f'./benchmarks/logs/{dataset}/{dataset}_{distance_metric}_{k}-neighbours_distribution.png')

    # Show the plot
    # plt.show()

    # Clear the plot
    plt.clf()


def neighbours_comulative_distribution_plot(dataset, distances, distance_metric, k):

    # Print a title for the analysis
    print(f"\n-- Analysis of the {k}th neighbour distances --")

    # The kth (5th) neighbours are those with index k-1 (4)
    kth_neighbour = distances[:, k-1]

    # Flatten the distances matrix to get a 1-d array containing all the distances to the kth-neigbour
    kth_neighbour = kth_neighbour.flatten()

    # Plot the comulative distribution function of pairwise distances estimated through the KDE curve obtained from the comulative histogram
    # histogram = sns.histplot(kth_neighbour, stat='percent', bins=30, kde=True, alpha=0, edgecolor=None, cumulative=True)

    # Plot the comulative distribution function of pairwise distances estimated through the KDE curve
    kde = sns.kdeplot(kth_neighbour, cumulative=True)
    # plt.yticks(kde.get_yticks(), [f'{int(tick * 100)}%' for tick in kde.get_yticks()])
    # kde = scipy.gaussian_kde(kth_neighbour)


    plt.title(f'Comulative distribution of {k}th Neighbour Distances\nfor {dataset} dataset')
    plt.xlabel(f'{distance_metric} Distance')
    plt.ylabel('Probability')

    kde_x = kde.get_lines()[0].get_data()[0]
    kde_y = kde.get_lines()[0].get_data()[1]

    # Define percentiles to be used
    percentiles = [0.7, 0.8, 0.9, 0.95]

    for p in percentiles:

        # Calculate the x-coordinate for the given percentile using interpolation
        kde_percentile_x = np.interp(p, kde_y, kde_x)

        # Draw the horizontal line precisely to the KDE curve intersection
        plt.plot([0, kde_percentile_x], [p, p], color='r', linestyle='--')

        # Annotate the percentile on the y-axis at the intersection point
        plt.text(0, p, str(int(p*100)) + "%", color='r', ha='left', va='bottom')

        # Calculate the y-coordinate of the KDE at the intersection point
        kde_percentile_y = np.interp(kde_percentile_x, kde_x, kde_y)

        # Draw the vertical line from the bottom to the KDE curve
        plt.plot([kde_percentile_x, kde_percentile_x], [0, kde_percentile_y], color='r', linestyle='--')

        # Annotate the KDE value on the x-axis at the intersection point
        plt.text(kde_percentile_x, p, f'{kde_percentile_x:.2f}', color='black', ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Force x-axis to start at 0
    plt.xlim(left=0)

    # Store the plot
    plt.savefig(f'./benchmarks/logs/{dataset}/{dataset}_{distance_metric}_{k}-neighbours_comulative_distribution.png')

    # Show the plot
    # plt.show()

    # Clear the plot
    plt.clf()


if __name__ == "__main__":

    # Set some options for the analysis
    distance_metrics = ['euclidean', 'manhattan', 'chebyshev', 'cosine']
    k = 10

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Name of the dataset to analyse", type=str)

    args = parser.parse_args()
    dataset = args.dataset

    # Perform the descriptive analysis of the dataset
    # dataset_analysis(dataset, distance_metrics)

    # Get the distances between the elements composing the dataset (or a sample of it if too big)
    distances = get_distances_between_elements(dataset, 'euclidean')


    print(f"-- Analysis of the pairwise distances between elements composing the dataset--")

    # Plot the distribution of pairwise distances between the elements composing the dataset (or a sample of it if too big)
    #distances_distribution_plot(dataset, distances, 'euclidean')

    # Plot the probability density of the pairwise distances between the elements composing the dataset (or a sample of it if too big)
    distances_probability_density_plot(dataset, distances, 'euclidean')

    # Plot the comulative distribution of pairwise distances between the elements composing the dataset (or a sample of it if too big)
    distances_comulative_distribution_plot(dataset, distances, 'euclidean')

    # Plot the distribution of the distances regarding the k-th neighbour of each element in the dataset (or a sample of it if too big)
    neighbours_distribution_plot(dataset, distances, 'euclidean', k)

    # Plot the comulative distribution of distances for the k-th neighbour of each element in the dataset (or a sample of it if too big)
    neighbours_comulative_distribution_plot(dataset, distances, 'euclidean', k)

    exit(0)

