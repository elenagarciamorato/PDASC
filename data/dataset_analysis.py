from matplotlib.pyplot import minorticks_on

from data.load_train_test_set import *
from benchmarks.algorithms.Exact.module import Exact_nn_search
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.spatial import distance
import data.load_train_test_set as lts
import seaborn as sns
import argparse
import sklearn

import PDASC.pdasc_ as pdasc

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
    distances_analysis(vector_training, distances)

# Analysis of distances between elements for every chosen distance metric
def distances_analysis(vector_training, distance_metrics):

    logging.info(f"\n------------ {dataset} dataset Distance Matrixes (built using several distance metrics)-----------\n")

    for d in distance_metrics:
        print(f"-- Distance analysis using {d} distance --")

        # If the distance is 'manhattan', we change it to 'cityblock' to use scipy
        if d == 'manhattan':
            d = 'cityblock'

        elif d == 'haversine':
            vector_training = np.radians(vector_training)

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

# Obtain the distance between every element in a random sample of the dataset and the other elements
def get_distances_pairwise(set, distance_function):

    # If the distance is 'haversine', we convert data to radians
    if distance_function == 'haversine':
        set = np.radians(set)

    # Calculate the pairwise distances between every element on the dataset using Linear Scan
    indices, coords, dists, n_dist = Exact_nn_search(set, set, len(set), distance_function, None, False)

    return dists

# Obtain the distance between every element in a random sample of the dataset and its k-th nearest neighbour
def get_distances_kth_nn(set, k, distance_metric):

    # If the distance is 'haversine', we convert data to radians
    if distance_metric == 'haversine':
        set = np.radians(set)

    # Calculate the distances between every element on the dataset and their knn using Linear Scan
    indices, coords, dists, n_dist = Exact_nn_search(set, set, k, distance_metric, None, False)

    # Get the the distance to the k-th nearest neighbour of each element
    kth_nn_dists = dists[:, k - 1]

    return kth_nn_dists

# Plot the probability density function (PDF) of pairwise distances between the elements composing the dataset
def elements_dists_pdf_plot(dataset, distances_dict):

    # Get the sample size
    sample_size = len(next(iter(distances_dict.values())))

    # Print a title for the analysis
    print(f"\n-- Descriptive analysis of the pairwise distances for the {dataset} dataset and PDF plot--")

    # Create a figure for the plots
    plt.figure(figsize=(15, 10))

    # Iterate over each distance metric and its corresponding distances
    for i, (distance_metric, distances) in enumerate(distances_dict.items()):
        # Flatten the distances matrix to get a 1-d array containing all the pairwise distances
        distances = distances.flatten()

        # Obtain the mean distance
        mean_distance = np.mean(distances)
        print(f"Mean distance ({distance_metric}): {mean_distance}")

        # Obtain the median distance
        median_distance = np.median(distances)
        print(f"Median distance ({distance_metric}): {median_distance}")

        # Create a subplot for each distance metric
        plt.subplot((len(distances_dict) + 1) // 2, 2, i + 1)
        plt.hist(distances, bins=50, edgecolor='black')
        plt.title(f'{distance_metric} Distance')
        plt.ylabel('Frequency')
        plt.xlim(left=0)

    # Add the main title
    plt.suptitle(f'Distribution of Pairwise Distances for {dataset} dataset')

    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Store the plot
    plt.savefig(f'./benchmarks/logs/{dataset}/{dataset}_pdf_{sample_size}.png')

    # Show the plot
    #plt.show()

    # Clear and close the plot
    plt.clf()
    plt.close()

# Plot the probability density function (PDF) of the pairwise distances between the elements composing the dataset
# And fit the data to a distribution
def elements_dists_fitting_pdf_plot(dataset, distances_dict):
    # Print info about the fit
    print(f"-- Fitting the data to a distribution for {dataset} dataset--")

    # Create a figure for the plots
    plt.figure(figsize=(15, 10))

    # Iterate over each distance metric and its corresponding distances
    for i, (distance_metric, distances) in enumerate(distances_dict.items()):
        # Flatten the distances matrix to get a 1-d array containing all the pairwise distances
        distances = distances.flatten()

        # Create a subplot for each distance metric
        plt.subplot((len(distances_dict) + 1) // 2, 2, i + 1)

        # Fit the data to a distribution
        f = Fitter(distances, distributions=get_common_distributions(), timeout=120)
        f.fit()
        f.summary()

        # Print the best fitting distribution
        best_dist = f.get_best(method='sumsquare_error')
        print(f'\nThe best fitting distribution for {distance_metric} is {best_dist}')

        # Plot the data distribution and the best fitting distribution
        plt.title(f'{distance_metric} Distance')
        plt.ylabel('Frequency')

        # Force x-axis to start at 0
        plt.xlim(left=0)

    # Add the main title
    plt.suptitle(f'Pairwise distances Distribution and Best Fitting Distribution \n (normalised) for {dataset} dataset')

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'./benchmarks/logs/{dataset}/{dataset}_{distance_metric}_pdf_fitted.png')
    #plt.show()

    # Clear and close the plot
    plt.clf()
    plt.close()

# Plot the cumulative distribution function (CDF) of pairwise distances between the elements composing the dataset
def elements_dists_cdf_plot(dataset, distances_dict):

    # Get the sample size
    sample_size = len(next(iter(distances_dict.values())))

    # Print a title for the analysis
    print(f"\n-- Analysis of the CDF of the pairwise distances of {sample_size} random elements of the {dataset} dataset--")

    # Create a figure for the plots
    plt.figure(figsize=(15, 10))

    # Iterate over each distance metric and its corresponding distances
    for i, (distance_metric, distances) in enumerate(distances_dict.items()):
        # Flatten the distances matrix to get a 1-d array containing all the pairwise distances
        distances = distances.flatten()

        # Create a subplot for each distance metric
        plt.subplot((len(distances_dict) + 1) // 2, 2, i + 1)

        # Plot the cumulative distribution function of pairwise distances estimated through the KDE curve
        kde = sns.kdeplot(distances, cumulative=True)

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
            plt.text(0, p, str(int(p * 100)) + "%", color='r', ha='left', va='bottom')

            # Calculate the y-coordinate of the KDE at the intersection point
            kde_percentile_y = np.interp(kde_percentile_x, kde_x, kde_y)

            # Draw the vertical line from the bottom to the KDE curve
            plt.plot([kde_percentile_x, kde_percentile_x], [0, kde_percentile_y], color='r', linestyle='--')

            # Annotate the KDE value on the x-axis at the intersection point
            plt.text(kde_percentile_x, p, f'{kde_percentile_x:.2f}', color='black', ha='left', va='bottom',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        plt.title(f'{distance_metric} Distance')
        plt.ylabel('Probability')
        plt.xlim(left=0)

    plt.suptitle(f'Cumulative distribution of Pairwise Distances for {dataset} dataset')

    # Store the plot
    plt.savefig(f'./benchmarks/logs/{dataset}/{dataset}_cdf_{sample_size}.png')

    # Show the plot
    #plt.show()

    # Clear and close the plot
    plt.clf()
    plt.close()

# Plot the probability density function (PDF) of the distances regarding the kth neighbour of each element in the dataset
def neighbours_dists_pdf_plot(dataset, distances_dict, k, pdasc_index=None):

    # Get the sample size
    sample_size = len(next(iter(distances_dict.values())))

    # Print a title for the analysis
    print(f"\n-- PDF of the distances to the {k}th nearest neighbors for a random sample of {sample_size} elements from the {dataset} dataset --")

    # Create a figure for the plots
    plt.figure(figsize=(15, 10))

    # Iterate over each distance metric and its corresponding distances
    for i, (distance_metric, distances) in enumerate(distances_dict.items()):
        # The kth neighbours are those with index k-1
        kth_neighbour = distances[:]

        # k th neighbour mean distance
        mean_distance = np.mean(kth_neighbour)
        print(f"{k}th neighbour mean distance ({distance_metric}): {mean_distance}")

        # k th neighbour median distance
        median_distance = np.median(kth_neighbour)
        print(f"{k}th neighbour median distance ({distance_metric}): {median_distance}")

        # Compute the 3rd quartile of kth neighbour
        q3 = np.percentile(kth_neighbour, 75)
        print(f"Third quartile of {k}th neighbour ({distance_metric}): {q3}")

        # Compute the 90% percentile of kth neighbour
        p90 = np.percentile(kth_neighbour, 90)
        print(f"90% percentile of {k}th neighbour ({distance_metric}): {p90}")

        # Plot the distribution of kth neighbour distances (histogram)
        plt.subplot((len(distances_dict) + 1) // 2, 2, i + 1)
        plt.hist(kth_neighbour, bins=50, edgecolor='black')
        plt.title(f'{distance_metric} Distance')
        plt.ylabel('Frequency')
        plt.xlim(left=0)

    # Add the main title
    plt.suptitle(f'Distribution of {k}th Neighbour Distances\nfor {dataset} dataset')

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    if pdasc_index:
        # Add the main title
        plt.suptitle(f'Probability Density of {k}th Neighbour Distances\nfor {dataset} dataset (PDASC index)')
        # Store the plot
        plt.savefig(f'./benchmarks/logs/{dataset}/{dataset}_neighbours_pdf_PDASC_{sample_size}.png')

    else:
         # Add the main title
        plt.suptitle(f'Probability Density of {k}th Neighbour Distances\nfor {dataset} dataset')

        # Store the plot
        plt.savefig(f'./benchmarks/logs/{dataset}/{dataset}_neighbours_pdf_{sample_size}.png')

    #plt.show()

    # Clear and close the plot
    plt.clf()
    plt.close()

# Plot the cumulative distribution function (CDF) of the distances of the kth neighbour of each element in the dataset
def neighbours_dists_cdf_plot(dataset, distances_dict, k, pdasc_index=None):

    # Get the sample size
    sample_size = len(next(iter(distances_dict.values())))

    # Print a title for the analysis
    print(f"\n-- CDF of the distances to the {k}th nearest neighbors for a random sample of {sample_size} elements from the {dataset} dataset --")
    # Create a figure for the plots
    plt.figure(figsize=(15, 10))

    # Iterate over each distance metric and its corresponding distances
    for i, (distance_metric, distances) in enumerate(distances_dict.items()):

        # Flatten the distances matrix to get a 1-d array containing all the distances to the kth-neighbour
        kth_neighbour = distances[:].flatten()

        # Create a subplot for each distance metric
        plt.subplot((len(distances_dict) + 1) // 2, 2, i + 1)

        # Plot the cumulative distribution function of pairwise distances estimated through the KDE curve
        kde = sns.kdeplot(kth_neighbour, cumulative=True, label=distance_metric, linewidth=2)


        kde_x = kde.get_lines()[-1].get_data()[0]
        kde_y = kde.get_lines()[-1].get_data()[1]

        # Define percentiles to be used
        percentiles = [0.7, 0.8, 0.9, 1]

        for p in percentiles:
            # Calculate the x-coordinate for the given percentile using interpolation
            kde_percentile_x = np.interp(p, kde_y, kde_x)

            # Draw the horizontal line precisely to the KDE curve intersection
            plt.plot([0, kde_percentile_x], [p, p], color='r', linestyle='--',alpha=0.5)

            # Annotate the percentile on the y-axis at the intersection point
            plt.text(0, p, str(int(p*100)) + "%", color='black', ha='left', va='bottom')

            # Calculate the y-coordinate of the KDE at the intersection point
            kde_percentile_y = np.interp(kde_percentile_x, kde_x, kde_y)

            # Draw the vertical line from the bottom to the KDE curve
            plt.plot([kde_percentile_x, kde_percentile_x], [0, kde_percentile_y], color='r', linestyle='--', alpha=0.5, marker='.')

            # Annotate the KDE value on the x-axis at the intersection point
            # plt.text(kde_percentile_x, p, f'{kde_percentile_x:.2f}', color='black', ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

            # Increase minor ticks
            import matplotlib.ticker as ticker

            # Get the current axis
            ax = plt.gca()

            # Set the number of intervals between major ticks
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=6))  # n=6 gives 5 minor ticks between each major

            # Optional: customize appearance of minor ticks
            #ax.tick_params(axis='x', which='minor', length=4, color='gray')  # shorter gray minor ticks


        plt.title(f'{distance_metric} distance')
        plt.ylabel('Probability')
        plt.xlim(left=0)

    if pdasc_index:
        # Add the main title
        plt.suptitle(f'Cumulative distribution of {k}th Neighbour Distances\nfor {dataset} dataset (PDASC index)')
        # Store the plot
        plt.savefig(f'./benchmarks/logs/{dataset}/{dataset}_neighbours_cdf_PDASC_{sample_size}.png')

    else:
         # Add the main title
        plt.suptitle(f'Cumulative distribution of {k}th Neighbour Distances\nfor {dataset} dataset')

        # Store the plot
        plt.savefig(f'./benchmarks/logs/{dataset}/{dataset}_neighbours_cdf_{sample_size}.png')

    # Show the plot
    #plt.show()

    # Clear and close the plot
    plt.clf()
    plt.close()


if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Name of the dataset to analyse", type=str)
    parser.add_argument("-p", "--pdasc", help="PDASC index to be used", type=str)
    parser.add_argument("-f", "--distance_functions", help="Distance functions to use", type=str, nargs='+', required=True)

    args = parser.parse_args()

    if args.dataset:
        dataset = args.dataset
    elif args.pdasc:
        dataset = args.pdasc

    distance_functions = args.distance_functions

    if not args.dataset and not args.pdasc:
        print("Error: You must provide either a dataset name with -d or a PDASC index with -p.")
        parser.print_help()
        exit(1)

    if args.dataset and args.pdasc:
        print("Error: You cannot provide both a dataset name with -d and a PDASC index with -p.")
        parser.print_help()
        exit(1)


    # Set some initial parameters for the analysis
    # distance_metrics = ['euclidean', 'manhattan', 'chebyshev', 'cosine']
    k_neighbours = 12
    datasets_size = {
        "wdbc": 1000,
        "municipios": 8130,
        "MNIST": 69000,
        "NYtimes": 290000,
        "GLOVE": 1000000,
    }

    if datasets_size[dataset] > 100000:
        sample_size = int(datasets_size[dataset] * 0.01)
    else:
        sample_size = int(datasets_size[dataset] * 0.1)


    # If the user choose to use the prototypes points of a PDASC index already generated:
    if args.pdasc:

        distances_between_elements = {}
        distances_kth_nn = {}

        for distance_function in distance_functions:

            # Load the PDASC index
            distance, grupos_capa, puntos_capa, labels_capa = pdasc.load_PDASC_index(dataset, distance_function)


            # Reconstruct the puntos_capa from the labels_capa
            for i in range(len(grupos_capa) - 2, -1, -1):
                for j in range(len(grupos_capa[i])):
                    puntos = puntos_capa[i][j]
                    #print(f'Layer {i}, Group {j}: {len(puntos)} points')

                    for k in range(len(puntos)):
                        if np.all(np.isnan(puntos[k])):
                            label_punto = labels_capa[i+1][j//2][k]
                            #print(f"Label of that point: {label_punto}")
                            puntos_capa[i][j][k] = puntos_capa[i+1][j//2][label_punto]

            #print(puntos_capa)

            # Regarding the sample_size, we identify the layer whose points we will analyse
            # Initialize variables to store the index and the closest sum
            closest_index = -1
            closest_sum = 0

            # Iterate over the elements of grupos_capa
            for i, subarray in enumerate(grupos_capa):
                current_sum = np.sum(subarray)
                if sample_size >= current_sum > closest_sum:
                    closest_sum = current_sum
                    closest_index = i

            # Print the result
            print(f"The dataset has {len(grupos_capa)} layers, and the layer with the number of elements closest to {sample_size} without exceeding it is: {closest_index}")
            # print(f"The sum of the subelements of this element is: {closest_sum}")

            # We take all the points from the desired layer
            # (Lets take into account that the layer of puntos_capa equivalent to lablels_capa is one below)
            puntos_capa_concatenado = np.vstack(puntos_capa[closest_index-1])
            print(f"Number of points in the layer {closest_index}: {len(puntos_capa_concatenado)}")

            # These prototypes will constitute the sample to be analysed
            sample = puntos_capa_concatenado

            # Perform the descriptive analysis of the dataset
            # dataset_analysis(set_to_analyse, distance_metrics)

            # Get the distances between the prototypes composing a a layer of the PDASC index
            # for a distance function and store them into a dictionary
            #distances_between_elements[distance_function] = get_distances_pairwise(sample, distance_function)

            # Get the distances between the prototypes composing a a layer of the PDASC index and its k-th nearest neighbour
            # for a distance metric and store them into a dictionary
            distances_kth_nn[distance_function] = get_distances_kth_nn(sample, k_neighbours, distance_function)

    # If the user choose to use a random sample of the dataset
    else:

        file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"
        vector_training, vector_testing = load_train_test_h5py(file_name)

        # We take a sample of n random elements from the dataset
        sample = vector_training[np.random.choice(len(vector_training), sample_size, replace=True)]

        # Perform the descriptive analysis of that sample
        # dataset_analysis(sample, distance_metrics)

        # Get the distances between the elements composing a set of elements
        # for each distance metric and store them into a dictionary
        #distances_between_elements = {distance_function: get_distances_pairwise(sample, distance_function) for distance_function in distance_functions

        # Get the distances between the elements composing a set of elements and its k-th nearest neighbour
        # for each distance metric and store them into a dictionary
        distances_kth_nn = {distance_function: get_distances_kth_nn(sample, k_neighbours, distance_function) for distance_function in distance_functions}


    print(f"-- Probability Functions--")

    # Plot the PDF of the pairwise distances between the elements composing the dataset (or a sample of it if too big)
    #elements_dists_pdf_plot(dataset, distances_between_elements)

    # Plot the PDF fitting of the pairwise distances between the elements composing the dataset (or a sample of it if too big)
    #elements_dists_fitting_pdf_plot(dataset, distances_between_elements)

    # Plot the CDF of pairwise distances between the elements composing the dataset (or a sample of it if too big)
    #elements_dists_cdf_plot(dataset, distances_between_elements)

    # Plot the PDF of the distances regarding the k-th neighbour of each element in the dataset (or a sample of it if too big)
    #neighbours_dists_pdf_plot(dataset, distances_kth_nn, k_neighbours, args.pdasc)

    # Plot the CDF of the distances for the k-th neighbour of each element in the dataset (or a sample of it if too big)
    neighbours_dists_cdf_plot(dataset, distances_kth_nn, k_neighbours, args.pdasc)


    exit(0)

