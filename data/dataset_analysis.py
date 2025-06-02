from data.load_train_test_set import *
from benchmarks.algorithms.Exact.module import Exact_nn_search
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.spatial import distance
import data.load_train_test_set as lts
import seaborn as sns
import argparse
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from plotnine import *
import os

import sklearn

import PDASC.pdasc_ as pdasc

from fitter import Fitter, get_common_distributions, get_distributions

# Load a random sample of the dataset
def load_random_sample(dataset, sample_size):
    # Load the random sample of the dataset
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"
    vector_training, vector_testing = load_train_test_h5py(file_name)

    # We take a sample of n random elements from the dataset
    np.random.seed(42)  # Fijar la semilla para reproducibilidad
    sample = vector_training[np.random.choice(len(vector_training), sample_size, replace=True)]

    return sample

def load_PDASC_sample(dataset, sample_size, distance_function):

    # Load the PDASC index
    distance, grupos_capa, puntos_capa, labels_capa = pdasc.load_PDASC_index(dataset, distance_function)

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

# Perform a descriptive analysis of a given dataset. It includes the analysis of the dataset's dimensions and distances
def descriptive_analysis(dataset, distances):

    # Set log configuration
    log_file = f"./data/dataset_analysis/{dataset}/analisis_descriptivo_{dataset}.log"
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

    # if sample elements are of type float64, convert them to float32
    if set.dtype == np.float64:
        set = set.astype(np.float32)

    # Calculate the pairwise distances between every element on the dataset using Linear Scan
    indices, coords, dists, n_dist = Exact_nn_search(set, set, len(set), distance_function, None, False)

    return dists


# Get the distance between every element in a random sample of the dataset and its k-th nearest neighbour in the given complete dataset
def get_distances_kth_nn(subset, complete_set, k, distance_metric):

    # if sample elements are of type float64, convert them to float32
    if subset.dtype == np.float64:
        subset = subset.astype(np.float32)
        complete_set = complete_set.astype(np.float32)

    # Calculate the distances between every element on the dataset and their knn using Linear Scan
    indices, coords, dists, n_dist = Exact_nn_search(complete_set, subset, k, distance_metric, None, False)

    # Get the the distance to the k-th nearest neighbour of each element
    kth_nn_dists = dists[:, k -1]

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
    plt.savefig(f'./data/dataset_analysis/{dataset}/{dataset}_pdf_{sample_size}.png')

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
    plt.savefig(f'./data/dataset_analysis/{dataset}/{dataset}_{distance_metric}_pdf_fitted.png')
    #plt.show()

    # Clear and close the plot
    plt.clf()
    plt.close()

# Plot the cumulative distribution function (CDF) of pairwise distances between the elements composing the dataset
def elements_dists_cdf_plot(dataset, distances_dict, pdasc_index=None):

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

        # Increase minor ticks
        # Get the current axis
        ax = plt.gca()

        # Set the number of intervals between major ticks
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        # Optional: customize appearance of minor ticks
        # ax.tick_params(axis='x', which='minor', length=4, color='gray')  # shorter gray minor ticks

        # Fix the limits of the x axis regarding the dataset/distance combination used
        max_val = float({'municipios': {'euclidean': 28, 'manhattan': 38, 'chebyshev': 25, 'cosine': 0.25, 'haversine': 0.5},
                         'MNIST': {'euclidean': 4500, 'manhattan': 75000, 'chebyshev': 300, 'cosine': 1.2},
                         'GLOVE': {'euclidean': 25, 'manhattan': 210, 'chebyshev': 12, 'cosine': 1.9},
                         'NYtimes': {'euclidean': 2, 'manhattan': 25, 'chebyshev': 0.8, 'cosine': 1.4}
                         }.get(dataset, {}).get(distance_metric) or 0)

        plt.xlim(0, max_val)

        plt.title(f'{distance_metric} Distance')
        plt.ylabel('Probability')
        plt.xlim(left=0)

    if pdasc_index:
        # Add the main title
        plt.suptitle(f'Cumulative distribution of Pairwise Distances\nfor {dataset} dataset (PDASC index)')
        # Store the plot
        plt.savefig(f'./data/dataset_analysis/{dataset}/{dataset}_pairwise_cdf_PDASC_{sample_size}.png')

    else:
        # Add the main title
        plt.suptitle(f'Cumulative distribution of Pairwise Distances\nfor {dataset} dataset')

        # Store the plot
        plt.savefig(f'./data/dataset_analysis/{dataset}/{dataset}_pairwise_cdf_{sample_size}.png')

    # Show the plot
    # plt.show()

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
        plt.savefig(f'./data/dataset_analysis/{dataset}/{dataset}_neighbours_pdf_PDASC_{sample_size}.png')

    else:
         # Add the main title
        plt.suptitle(f'Probability Density of {k}th Neighbour Distances\nfor {dataset} dataset')

        # Store the plot
        plt.savefig(f'./data/dataset_analysis/{dataset}/{dataset}_neighbours_pdf_{sample_size}.png')

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
            #plt.text(kde_percentile_x, p, f'{kde_percentile_x:.2f}', color='black', ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        # Increase minor ticks
        # Get the current axis
        ax = plt.gca()
        # Set the number of intervals between major ticks
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        # Optional: customize appearance of minor ticks
        #ax.tick_params(axis='x', which='minor', length=4, color='gray')  # shorter gray minor ticks

        # Fix the limits of the x axis regarding the dataset/distance combination used
        max_val = float({'municipios': {'euclidean': 17, 'manhattan': 23.0, 'chebyshev': 13.0,
                            'cosine': 0.08, 'haversine': 0.27},
                            'MNIST': {'euclidean': 2800, 'manhattan': 35000, 'chebyshev': 310, 'cosine': 0.6},
                            'GLOVE': {'euclidean': 20, 'manhattan': 150, 'chebyshev': 5, 'cosine': 0.85},
                            'NYtimes': {'euclidean': 1.5, 'manhattan': 18.5, 'chebyshev': 0.28, 'cosine': 1.1}
                           }.get(dataset, {}).get(distance_metric) or 0)

        plt.xlim(0 , max_val)

        plt.title(f'{distance_metric} distance')
        plt.ylabel('Probability')
        plt.xlim(left=0)

    if pdasc_index:
        # Add the main title
        plt.suptitle(f'Cumulative distribution of {k}th Neighbour Distances\nfor {dataset} dataset (PDASC index)')
        # Store the plot
        plt.savefig(f'./data/dataset_analysis/{dataset}/{dataset}_neighbours_cdf_PDASC_{sample_size}.png')

    else:
         # Add the main title
        plt.suptitle(f'Cumulative distribution of {k}th Neighbour Distances\nfor {dataset} dataset')

        # Store the plot
        plt.savefig(f'./data/dataset_analysis/{dataset}/{dataset}_neighbours_cdf_{sample_size}.png')

    # Show the plot
    #plt.show()

    # Clear and close the plot
    plt.clf()
    plt.close()

def neighbours_dists_cdf_plot_complete(dataset, distances_dict_random, distances_dict_pdasc, k):

    def plot_cdf(ax, distances, distance_metric, max_val):

        kde = sns.kdeplot(distances, cumulative=True, label=distance_metric, linewidth=2, ax=ax)
        kde_x = kde.get_lines()[-1].get_data()[0]
        kde_y = kde.get_lines()[-1].get_data()[1]

        percentiles = [0.7, 0.8, 0.9, 1]

        for p in percentiles:
            kde_percentile_x = np.interp(p, kde_y, kde_x)
            ax.plot([0, kde_percentile_x], [p, p], color='r', linestyle='--', alpha=0.5)
            ax.text(0, p, str(int(p * 100)) + "%", color='black', ha='left', va='bottom')
            kde_percentile_y = np.interp(kde_percentile_x, kde_x, kde_y)
            ax.plot([kde_percentile_x, kde_percentile_x], [0, kde_percentile_y], color='r', linestyle='--', alpha=0.5, marker='.')

        ax.set_xlim(0, max_val)

        # If the subplot is in the first column (random sample)
        if (ax.get_subplotspec().colspan.start % 2) == 0:
            ax.set_ylabel('Probability', labelpad=15)

        # If the subplot is in the second column (PDASC)
        else:
            ax.set_ylabel('')

            # Get values for the radius

            # By using percentiles of the KDE curve
            cdf_65 = np.interp(0.65, kde_y, kde_x)
            cdf_70 = np.interp(0.7, kde_y, kde_x)
            cdf_75 = np.interp(0.75, kde_y, kde_x)
            cdf_80 = np.interp(0.8, kde_y, kde_x)
            cdf_85 = np.interp(0.85, kde_y, kde_x)
            cdf_90 = np.interp(0.90, kde_y, kde_x)
            cdf_95 = np.interp(0.95, kde_y, kde_x)
            cdf_100 = np.interp(1, kde_y, kde_x)

            cdfs= [cdf_65, cdf_70, cdf_75, cdf_80, cdf_85, cdf_90, cdf_95, cdf_100]
            #print(f"Radius values for {distance_metric} experiments: {cdfs}")

            # Seleccionar decimales según la métrica
            if distance_metric in ['euclidean', 'manhattan']:
                decimales = 2
            elif distance_metric == 'chebyshev' or distance_metric == 'haversine':
                decimales = 3
            elif distance_metric == 'cosine':
                decimales = 4
            else:
                decimales = 2

            # Imprimir los valores de cdfs redondeados y separados por espacio
            #print(f"Radius values for {distance_metric} experiments: {cdfs}")
            print(f"r_{dataset}_{distance_metric}=\"" + " ".join([f"{cdf:.{decimales}f}" for cdf in cdfs]) + "\"")

            # By dividing the space between the 70th and 100th percentiles into 10 intervals
            # x_ticks = np.linspace(cdf_100, cdf_70, 10)
            # print(f"Radius values for {distance_metric} experiments: {x_ticks}")


        ax.set_xlabel(f'{distance_metric.capitalize()} distance')

    print(f"\n-- Comparison between CDF of the distances to the {k}th nearest neighbors for a random sample of elements and PDASC index prototypes from the {dataset} dataset --")

    plt.figure(figsize=(13, 18))
    for i, distance_metric in enumerate(distances_dict_random.keys()):
        max_val = float({
            'municipios': {'euclidean': 17, 'manhattan': 23.0, 'chebyshev': 13.0, 'cosine': 0.08, 'haversine': 0.27},
            'MNIST': {'euclidean': 2800, 'manhattan': 35000, 'chebyshev': 310, 'cosine': 0.6},
            'GLOVE': {'euclidean': 20, 'manhattan': 150, 'chebyshev': 5, 'cosine': 0.85},
            'NYtimes': {'euclidean': 1.5, 'manhattan': 18.5, 'chebyshev': 0.28, 'cosine': 1.1}
        }.get(dataset, {}).get(distance_metric) or 0)


        ax_random = plt.subplot(len(distances_dict_random), 2, 2 * i + 1)

        if i ==0 :  # Apply title only to the first row
            ax_random.set_title(f'Estimated by using a random sample of the dataset',  y = 1.15)

        plot_cdf(ax_random, distances_dict_random[distance_metric].flatten(), distance_metric, max_val)

        ax_pdasc = plt.subplot(len(distances_dict_pdasc), 2, 2 * i + 2)

        if i ==0 :  # Apply title only to the first row
            ax_pdasc.set_title(f'Estimated by using a set of prototype points from PDASC index', y=1.15)

        plot_cdf(ax_pdasc, distances_dict_pdasc[distance_metric].flatten(), distance_metric, max_val)

    plt.suptitle(f'Cumulative Distribution Function of {k}th Neighbour Distances\nfor {dataset} Dataset (Random vs PDASC)', fontsize=16, y=0.95)
    plt.tight_layout(rect=[0.04, 0.03, 0.96, 0.93], h_pad=2.5)
    plt.savefig(f'./data/dataset_analysis/{dataset}/{dataset}_neighbours_cdf_comparison_{len(next(iter(distances_dict_random.values())))}.png', dpi=300)
    plt.clf()
    plt.close()


def neighbours_dists_cdf_plot_solapado(dataset, distances_dict_random, distances_dict_pdasc, k):
    print(f"\n-- Comparison between CDF of the distances to the {k}th nearest neighbors for a random sample of elements and PDASC index prototypes from the {dataset} dataset --")

    ordered_metrics = ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'haversine']
    all_data = []

    for distance_metric in ordered_metrics:
        if distance_metric not in distances_dict_random:
            continue  # skip if not applicable

        rand_dists = np.sort(distances_dict_random[distance_metric])
        pdasc_dists = np.sort(distances_dict_pdasc[distance_metric])

        rand_cdf = np.arange(1, len(rand_dists)+1) / len(rand_dists)
        pdasc_cdf = np.arange(1, len(pdasc_dists)+1) / len(pdasc_dists)

        df_rand = pd.DataFrame({
            'distance': rand_dists,
            'cdf': rand_cdf,
            'metric': distance_metric,
            'method': 'Random'
        })

        df_pdasc = pd.DataFrame({
            'distance': pdasc_dists,
            'cdf': pdasc_cdf,
            'metric': distance_metric,
            'method': 'PDASC'
        })

        all_data.append(df_rand)
        all_data.append(df_pdasc)

    df_all = pd.concat(all_data)

    # Orden personalizado
    df_all['metric'] = pd.Categorical(df_all['metric'], categories=ordered_metrics, ordered=True)

    # Etiquetas personalizadas de las métricas
    metric_labels = {
        'euclidean': 'Euclidean Distance',
        'manhattan': 'Manhattan Distance',
        'chebyshev': 'Chebyshev Distance',
        'cosine': 'Cosine Distance',
        'haversine': 'Haversine Distance'
    }

    # Colores personalizados
    custom_colors = {
        'Random': '#4C78A8',  # azul suave
        'PDASC': '#F58518'    # naranja suave
    }

    p = (
            ggplot(df_all, aes(x='distance', y='cdf', fill='method')) +
            geom_area(alpha=0.4, position='identity') +
            geom_line(aes(color='method'), size=1.1) +
            scale_fill_manual(values=custom_colors) +
            scale_color_manual(values=custom_colors) +
            facet_wrap('~ metric', ncol=2, labeller=metric_labels, scales='free_x') +
            labs(
                title=f'Cumulative Distribution Function of {k}th Neighbour Distances for {dataset} Dataset (Random vs PDASC)',
                x='',
                y='Probability'
            ) +
            theme_minimal(base_size=13) +
            theme(
                figure_size=(12, 9),
                legend_position='bottom',
                panel_background=element_rect(fill='white', color='black'),
                plot_background=element_rect(fill='white', color='white'),
                panel_grid_major=element_line(color="#e5e5e5"),
                panel_grid_minor=element_line(color="#f5f5f5"),
                legend_title=element_blank(),
                plot_title=element_text(ha='center', family='Arial'),
                axis_ticks_major_x=element_line(),
                axis_ticks_major_y=element_line(),
                axis_ticks_minor_x=element_line(color='gray', size=0.5),
                axis_ticks_minor_y=element_line(color='gray', size=0.5),
                axis_text_x=element_text(size=10, family='Arial'),
                axis_text_y=element_text(size=10, family='Arial'),
                strip_text_x=element_text(size=11, weight='bold', margin={'t': 10}, family='Arial'),
            ) +
            scale_y_continuous(limits=(0, 1))
        )
    out_path = f'./data/dataset_analysis/{dataset}'
    os.makedirs(out_path, exist_ok=True)
    filename = f'{dataset}_neighbours_cdf_comparision_overlap_{len(rand_dists)}.png'
    p.save(os.path.join(out_path, filename), dpi=300)


if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Name of the dataset to analyse", type=str)
    parser.add_argument("-p", "--pdasc", help="PDASC index to be used", type=str)
    parser.add_argument("-f", "--distance_functions", help="Distance functions to use", type=str, nargs='+', required=True)

    args = parser.parse_args()

    if (args.dataset and args.pdasc) or (args.dataset and args.pdasc):
        dataset = args.dataset
    elif args.dataset:
        dataset = args.dataset
    elif args.pdasc:
        dataset = args.pdasc

    distance_functions = args.distance_functions

    if not args.dataset and not args.pdasc:
        print("Error: You must provide either a dataset name with -d or a PDASC index with -p.")
        parser.print_help()
        exit(1)

    """
    if args.dataset and args.pdasc:
        print("Error: You cannot provide both a dataset name with -d and a PDASC index with -p.")
        parser.print_help()
        exit(1)
    """

    # Set some initial parameters for the analysis
    # distance_metrics = ['euclidean', 'manhattan', 'chebyshev', 'cosine']
    k_neighbours = 10
    datasets_size = {
        "wdbc": 1000,
        "municipios": 8031,
        "MNIST": 59999,
        "NYtimes": 290000,
        "GLOVE": 1200000 #1183514
    }

    if datasets_size[dataset] > 100000:
        sample_size = int(datasets_size[dataset] * 0.01)
    else:
        sample_size = int(datasets_size[dataset] * 0.1)


    # If the user choose to use both the random sample of the dataset and the PDASC index:
    if args.dataset and args.pdasc:

        # Load a random sample of the dataset
        sample = load_random_sample(dataset, sample_size)
        complete_dataset = load_random_sample(dataset, datasets_size[dataset])

        # Perform the descriptive analysis of that sample
        # dataset_analysis(sample, distance_metrics)

        # Get the distances between the elements composing a set of elements
        # for each distance metric and store them into a dictionary
        # distances_between_elements = {distance_function: get_distances_pairwise(sample, distance_function) for distance_function in distance_functions}

        # Get the distances between the elements composing a set of elements and its k-th nearest neighbour
        # for each distance metric and store them into a dictionary
        distances_kth_nn_random = {distance_function: get_distances_kth_nn(sample, complete_dataset, k_neighbours, distance_function) for
                                   distance_function in distance_functions}

        # PDASC PROTOTYPES CDF

        distances_kth_nn_pdasc = {}

        for distance_function in distance_functions:

            sample = load_PDASC_sample(dataset, sample_size, distance_function)
            complete_dataset = load_PDASC_sample(dataset, datasets_size[dataset], distance_function)


            # Perform the descriptive analysis of the dataset
            # dataset_analysis(set_to_analyse, distance_metrics)

            # Get the distances between the prototypes composing a a layer of the PDASC index
            # for a distance function and store them into a dictionary
            # distances_between_elements[distance_function] = get_distances_pairwise(sample, distance_function)

            # Get the distances between the prototypes composing a a layer of the PDASC index and its k-th nearest neighbour
            # for a distance metric and store them into a dictionary
            distances_kth_nn_pdasc[distance_function] = get_distances_kth_nn(sample, complete_dataset, k_neighbours, distance_function)

    # If the user choose to use the prototypes points of a PDASC index already generated:
    elif args.pdasc:

        distances_between_elements = {}
        distances_kth_nn = {}

        for distance_function in distance_functions:

            sample = load_PDASC_sample(dataset, sample_size, distance_function)
            complete_dataset = load_PDASC_sample(dataset, datasets_size[dataset], distance_function)

            # Perform the descriptive analysis of the dataset
            # dataset_analysis(set_to_analyse, distance_metrics)

            # Get the distances between the prototypes composing a a layer of the PDASC index
            # for a distance function and store them into a dictionary
            # distances_between_elements[distance_function] = get_distances_pairwise(sample, distance_function)


            # Get the distances between the prototypes composing a a layer of the PDASC index and its k-th nearest neighbour of within the given dataset
            # for a distance metric and store them into a dictionary
            distances_kth_nn[distance_function] = get_distances_kth_nn(sample, complete_dataset, k_neighbours, distance_function)


    # If the user choose to use a random sample of the dataset
    elif args.dataset:

        sample = load_random_sample(dataset, sample_size)
        complete_dataset = load_random_sample(dataset, datasets_size[dataset])

        # Perform the descriptive analysis of that sample
        # dataset_analysis(sample, distance_metrics)

        # Get the distances between the elements composing a set of elements
        # for each distance metric and store them into a dictionary
        #distances_between_elements = {distance_function: get_distances_pairwise(sample, distance_function) for distance_function in distance_functions}

        # Get the distances between the elements composing a set of elements and its k-th nearest neighbour
        # for each distance metric and store them into a dictionary
        distances_kth_nn = {distance_function: get_distances_kth_nn(sample, complete_dataset, k_neighbours, distance_function) for distance_function in distance_functions}



    # print(f"-- Probability Functions--")

    # Plot the PDF of the pairwise distances between the elements composing the dataset (or a sample of it if too big)
    #elements_dists_pdf_plot(dataset, distances_between_elements)

    # Plot the PDF fitting of the pairwise distances between the elements composing the dataset (or a sample of it if too big)
    #elements_dists_fitting_pdf_plot(dataset, distances_between_elements)

    # Plot the CDF of pairwise distances between the elements composing the dataset (or a sample of it if too big)
    # elements_dists_cdf_plot(dataset, distances_between_elements, args.pdasc)

    # Plot the PDF of the distances regarding the k-th neighbour of each element in the dataset (or a sample of it if too big)
    #neighbours_dists_pdf_plot(dataset, distances_kth_nn, k_neighbours, args.pdasc)

    # Plot the CDF of the distances for the k-th neighbour of each element in the dataset (or a sample of it if too big)
    #neighbours_dists_cdf_plot(dataset, distances_kth_nn, k_neighbours, args.pdasc)

    # Plot the CDF of the distances for the k-th neighbour for a sample of the dataset and the PDASC index
    #neighbours_dists_cdf_plot_complete(dataset, distances_kth_nn_random, distances_kth_nn_pdasc, k_neighbours)

    # Plot the CDF of the distances for the k-th neighbour for a sample of the dataset and the PDASC index
    neighbours_dists_cdf_plot_solapado(dataset, distances_kth_nn_random, distances_kth_nn_pdasc, k_neighbours)


    exit(0)

