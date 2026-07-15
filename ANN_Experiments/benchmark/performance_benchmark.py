from ANN_Experiments.neighbors_utils import *
import argparse
import datetime
import logging
from pandas.api.types import CategoricalDtype
import pandas as pd
import os

# For each method, define the name of the query parameter (according to the file name) and its position in the file name
QUERY_PARAMETERS = {
    "NMSLIBHNSW": (7, "efS"),
    "FAISSHNSW": (7, "efS"),
    "IVF": (6, "nprobe"),
    "PYNN": (8, "eps"),
    "LSH": (5, "nbits"),
    "ANNOY": (6, "ksearch"),
    "PDASC": (7, "r")
}

# Function to get the recall of a k-nn experiment
def get_recall(dataset, k, distance, indices, coords, distances):

    # Load the baseline neighbors generated using an exact method
    baseline_file = f"./ANN_Experiments/NearestNeighbors/{dataset}/knn_{dataset}_{k}_{distance}_Exact_auto.hdf5"

    if not os.path.isfile(baseline_file):
        return np.nan

    indices_baseline, coords_baseline, dists_baseline = load_neighbors(baseline_file)

    # Count the number of neighbors that match between the obtained and baseline neighbors
    hit = sum(map(lambda x, y: len(np.intersect1d(x.astype(int), y)), list(indices), list(indices_baseline)))

    # Calculate recall as the percentage of matches (hits) relative to the total
    recall = hit / indices_baseline.size * 100

    return recall


# Load the performance of the k-nn experiments regarding the selected dataset
def explore_experiments(dataset, distance_function, optional_filters=None):

    # Set the directory path to load the experiments according to the dataset provided
    directory_path = "./ANN_Experiments/NearestNeighbors/" + dataset

    results = []

    # For every .hdf5 file in the directory (file containing the neighbors and performances)
    for root, _, files in os.walk(directory_path):

        # Apply filters to files
        if optional_filters:
            files = [f for f in files if distance_function in f and any(opt in f for opt in optional_filters)]
        else:
            files = [f for f in files if distance_function in f]

        for file in files:

            if file.endswith(".hdf5"):

                # Load the neighbors and performance of the k-nn experiments associated to that file
                indices, coords, distances, n_dist, index_size, index_time, search_time = \
                    load_neighbors_performance(directory_path + "/" + file)

                # Split the file name to get the information about the experiment
                parts = file.split("_")
                parts[-1] = parts[-1].replace(".hdf5", "")

                # -------------------------------------------------------------
                # Query parameter (radius, efSearch, nprobe, ...)
                # -------------------------------------------------------------
                query_par = None

                if parts[4] in QUERY_PARAMETERS:
                    idx, prefix = QUERY_PARAMETERS[parts[4]]
                    query_par = float(parts[idx][len(prefix):])

                # -------------------------------------------------------------
                # Number of Nodes (default 1 for all methods except PDASC)
                # -------------------------------------------------------------
                n_nodes = 1

                if parts[4] == "PDASC":
                    n_nodes = int(parts[8][1:])

                # -------------------------------------------------------------
                # Store experiment
                # -------------------------------------------------------------
                results.append({
                    "Method": parts[4],
                    "Distance": distance_function,
                    "n_nodes": n_nodes,
                    "Index_par": "_".join(parts[5:idx]),
                    "Query_par": query_par,
                    "Recall(Av)": get_recall(
                        dataset,
                        parts[2],
                        distance_function,
                        indices,
                        coords,
                        distances,
                    ),
                    "Dist_C(Av)": np.round(np.mean(n_dist), 2),
                    "Dist_C-Node(Av)": np.round(np.mean(n_dist / n_nodes), 2),
                    "In_S(MB)": index_size,
                    "In_T(s)": np.round(index_time, 3),
                    "Search_T(s)": np.round(search_time, 4),
                })

    # Primero convierte la lista de dicts a DataFrame
    df = pd.DataFrame(results)

    # Define el orden personalizado de distancias
    custom_order = ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'haversine', 'jaccard']

    # Distancias adicionales
    other_distances = sorted(set(df['Distance']) - set(custom_order))

    # Crea el tipo categórico ordenado
    distance_type = CategoricalDtype(
        categories=custom_order + other_distances,
        ordered=True
    )

    # Aplica el tipo al DataFrame y continúa con el flujo
    formatted_results = (
        df
        .assign(
            n_nodes=lambda d: d['n_nodes'].astype(int),
            Query_par=lambda d: d['Query_par'].astype(float),
            Distance=lambda d: d['Distance'].astype(distance_type)
        )
        .sort_values(
            by=['Method', 'Distance', 'Index_par', 'n_nodes', 'Query_par'],
            ascending=True
        )
    )

    # Selección de columnas para imprimir en el CSV
    excel_results = formatted_results[
        ['Distance', 'n_nodes', 'Index_par', 'Query_par', 'Dist_C(Av)', 'Dist_C-Node(Av)', 'Recall(Av)', 'Search_T(s)', 'In_T(s)', 'In_S(MB)']
    ]

    # Añadir una columna de percentiles
    # percentiles = (1, 15, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99, 100)
    # excel_results['Percentil'] = percentiles

    # Orden final
    excel_results = excel_results.sort_values(
        by=['Distance', 'Index_par', 'n_nodes', 'Query_par']
    )

    # Reemplazo de decimales con comas
    for col in [
        'Query_par', 'Dist_C(Av)', 'Dist_C-Node(Av)', 'Recall(Av)', 'Search_T(s)', 'In_T(s)', 'In_S(MB)'
    ]:
        excel_results[col] = excel_results[col].astype(str).str.replace('.', ',', regex=False)

    # Almacenar en csv
    excel_results.to_csv(
        f'./ANN_Experiments/NearestNeighbors/{dataset}/benchmark_results_10nn_{dataset}.csv',
        index=False,
        sep=';'
    )

    # Log the results
    logging.info('------------------------------------------------------------------------\n' + formatted_results.to_string())
    logging.shutdown()

    # Inserta una línea en blanco entre métodos
    formatted_results = pd.concat([
        pd.concat([
            formatted_results.iloc[[i]],
            pd.DataFrame([[''] * len(formatted_results.columns)],
                         columns=formatted_results.columns)
        ])
        if i < len(formatted_results) - 1
        and formatted_results.iloc[i]['Method'] != formatted_results.iloc[i + 1]['Method']
        else formatted_results.iloc[[i]]
        for i in range(len(formatted_results))
    ]).reset_index(drop=True)

    return formatted_results


if __name__ == "__main__":

    #dataset=("wdbc")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Name of the dataset whose results would be benchmarked", type=str)
    parser.add_argument("distance_function", help="Distance function used to carry out the experiment", type=str)
    parser.add_argument("optional_filters", help="Benchmark optional filters", nargs='*', default=[])
    parser.add_argument('--log', action='store_true', help="Activa el registro en log")

    args = parser.parse_args()

    # Create a log file to store the performance of the k-nn experiments
    # Get the current date and time, formatting it
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%d-%m-%Y_%H:%M")

    if args.log:
        logging.basicConfig(
            filename="./ANN_Experiments/logs/" + args.dataset + "/benchmark_knn_" + args.dataset + "_" + args.distance_function + "_" + str(formatted_time) + ".log",
            filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)

        logging.info('------------------------------------------------------------------------')
        logging.info('                    %s Dataset Benchmarking for %s distance function', args.dataset, args.distance_function)
        logging.info('------------------------------------------------------------------------')
    else:
        logging.disable(logging.CRITICAL)

    # Explore the results of the experiments regarding the dataset provided
    df = explore_experiments(args.dataset, args.distance_function, args.optional_filters)

    # Print the results
    print(df.to_string())

    # print_Recall_pointplot(args.dataset, df)  # pointplot

    exit(0)

