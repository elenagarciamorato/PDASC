import argparse
from scipy.stats import ks_2samp
from dataset_analysis.distances_distribution_generator import get_pairwise_distances_flue
from scipy.stats import wasserstein_distance
import os
import pandas as pd

def kolmogorov_smirnov_test(dataset, distance_function, sample_size, nc, tg, nodes):
    """
    Perform the Kolmogorov-Smirnov test on the pairwise distances of a dataset.

    Parameters:
    - dataset: Name of the dataset to process.
    - distance_function: Distance function to use for computing pairwise distances.
    - sample_size: Percentage of the dataset to sample for the test.

    Returns:
    - KS statistic and p-value from the test.
    """


    datasets_size = {
        "wdbc": 1000,
        "municipios": 8031,
        "MNIST": 59999,
        "NYtimes": 290000,
        "GLOVE": 1200000  # 1183514
    }

    sample_size = int(datasets_size[dataset] * (sample_size / 100))
    dataset_size = datasets_size[dataset]

    print(f"Processing {dataset} dataset with {distance_function} distance function and sample size {sample_size}")

    # Paths for the PDASC and random distances
    PDASC_path = f'./dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{sample_size}_nc{nc}_tg{tg}_PDASC.csv'
    random_path = f'./dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{sample_size}_random.csv'

    # If the paths do not exist, compute the distances
    if not (os.path.exists(PDASC_path)) or not (os.path.exists(random_path)):
        # Compute the distances and save them to CSV files
        # print(f"\nComputing distances for {dataset} with {distance_function} distance function and sample size {sample_size}")
        random_dists, pdasc_dists = get_pairwise_distances_flue(dataset, distance_function, sample_size,nc, tg)

    else:
        # If it exists, load the distances from the CSV files
        # print(f"\nLoading distances for {dataset} with {distance_function} distance function and sample size {sample_size}")
        random_dists = pd.read_csv(random_path).values.flatten()
        pdasc_dists = pd.read_csv(PDASC_path).values.flatten()

    # Obtain the cumulative distribution functions (CDFs)
    #rand_cdf = np.arange(1, len(random_dists) + 1) / len(random_dists)
    #pdasc_cdf = np.arange(1, len(pdasc_dists) + 1) / len(pdasc_dists)

    # Perform the Kolmogorov-Smirnov test
    # x1 y x2 son tus dos muestras originales (no las CDFs ya evaluadas)
    #statistic, p_value = ks_2samp(pdasc_cdf, rand_cdf)
    ks_statistic, p_value = ks_2samp(pdasc_dists, random_dists)


    print("\n--- Results of the KS test ---")

    # Interpret the KS statistic
    print(f"\nKolmogorov-Smirnov statistic: {ks_statistic:.4f}")
    print(f"The KS statistic represents the maximum difference between the empirical CDFs.")
    print(f"A higher KS statistic implies greater divergence between distributions.")
    if ks_statistic < 0.1:
        print("→ The distributions are very similar.")
    elif ks_statistic < 0.3:
        print("→ The distributions show moderate differences.")
    else:
        print("→ The distributions differ significantly.")

    # Interpret p-value
    print(f"\nP-value: {p_value:.4f}")
    #print(f"The p-value indicates the probability of observing the data if the null hypothesis is true.")
    #print(f"A lower p-value suggests stronger evidence against the null hypothesis.")
    alpha = 0.05  # Significance level
    if p_value <= alpha:
        print("• The difference is statistically significant (p <= 0.05).")
        print("→ The two samples likely come from different distributions.")
    else:
        print("• The difference is not statistically significant (p > 0.05).")
        print("→ The two samples may come from the same distribution.")

    """
    # Log-spaced values in (0, 1], more concentrated near 1
    log_vals = np.logspace(-2, 0, 30, base=10)  # values from 0.01 to 1
    log_vals = 1 - (log_vals - min(log_vals)) / (max(log_vals) - min(log_vals))  # invert scale

    # Scale to (0, 100]
    scaled_vals = log_vals * 100
    scaled_vals = np.round(scaled_vals, 2)

    # Convert to tuple
    log_spaced_0_100 = tuple(scaled_vals)

    print(sorted(log_spaced_0_100))

    """
    # KS es la distancia de Kolmogorov-Smirnov entre las dos distribuciones (valor entre 0 y 1)
    # print(f"KS test statistic: {statistic}")

    # - El KS statistic (D) mide la mayor diferencia vertical entre las CDFs.
    #   No debe confundirse con el p-value, que indica la significancia estadística.

    # Interpretacion de la distancia KS:
    # - KS bajo (≈ 0) => Las distribuciones son similares.
    # - KS alto (≈ 1) => Las distribuciones son diferentes. (A partir de 0.3 ya se consideran completamente diferentes)

    # p_value indica si la diferencia es significativa o no (valor entre 0 y 1)
    # print(f"KS test p-value: {p_value}")

    # Interpretación del p-value:
    # - p-value alto (≈ 1) => Las distribuciones podrían ser iguales.
    # - p-value bajo (≤ 0.05 o ≤ 0.01) => Las distribuciones son estadísticamente diferentes.

    # Notas:
    # - El p-value depende del tamaño de la muestra.
    #   En muestras grandes, pequeñas diferencias pueden dar p-valores bajos.

def wasserstein_distance_test(dataset, distance_function, sample_size, nc, tg, nodes):
    """
    Perform the Wasserstein distance test on the pairwise distances of a dataset.

    Parameters:
    - dataset: Name of the dataset to process.
    - distance_function: Distance function to use for computing pairwise distances.
    - sample_size: Percentage of the dataset to sample for the test.

    Returns:
    - Wasserstein distance between the two distributions.
    """

    datasets_size = {
        "wdbc": 1000,
        "municipios": 8031,
        "MNIST": 59999,
        "NYtimes": 290000,
        "GLOVE": 1183514
    }

    sample_size = int(datasets_size[dataset] * (sample_size / 100))
    dataset_size = datasets_size[dataset]

    print(f"Processing {dataset} dataset with {distance_function} distance function and sample size {sample_size}")

    # Paths for the PDASC and random distances
    PDASC_path = f'./dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{sample_size}_nc{nc}_tg{tg}_PDASC.csv'
    random_path = f'./dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{sample_size}_random.csv'

    # If the paths do not exist, compute the distances
    if not (os.path.exists(PDASC_path)) or not (os.path.exists(random_path)):
        # Compute the distances and save them to CSV files
        random_dists, pdasc_dists = get_pairwise_distances_flue(dataset, distance_function, sample_size, nc, tg)

    else:
        # If it exists, load the distances from the CSV files
        random_dists = pd.read_csv(random_path).values.flatten()
        pdasc_dists = pd.read_csv(PDASC_path).values.flatten()

    # Compute Wasserstein distance
    wasserstein_dist = wasserstein_distance(pdasc_dists, random_dists)

    print(f"\nWasserstein distance: {wasserstein_dist:.4f}")


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", help="Name of the dataset to process", type=str, required=True)
    parser.add_argument("-dist", help="Distance function to use", type=str, required=True)
    parser.add_argument("-size", help="Sample size to use", type=int, required=True)
    parser.add_argument("-nc", help="Indicate the number of centroids of the PDASC index to be used.", type=int)
    parser.add_argument("-tg", help="Indicate the group size of the PDASC index to be used.", type=int)
    parser.add_argument("-nodes", help="Indicate the nodes of the PDASC index to be used.", type=int)



    args = parser.parse_args()

    dataset = args.dataset
    distance_function = args.dist
    nc = args.nc
    tg = args.tg
    sample_size = args.size
    nodes = args.nodes


    # Call the Kolmogorov-Smirnov test function
    #kolmogorov_smirnov_test(dataset, distance_function, sample_size, nc, tg, nodes)

    # Call the Wasserstein distance test function
    #wasserstein_distance_test(dataset, distance_function, sample_size, nc, tg, nodes)

    exit(0)
