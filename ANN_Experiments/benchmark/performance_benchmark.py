from ANN_Experiments.neighbors_utils import *
import argparse
import datetime
import logging
from pandas.api.types import CategoricalDtype
import pandas as pd
import os

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
            #print(optional_filters)
            files = [f for f in files if distance_function in f and any(opt in f for opt in optional_filters)]
        else:
            files = [f for f in files if distance_function in f]

        #print(files)

        for file in files:

            if file.endswith('.hdf5'):

                # Load the neighbors and performance of the k-nn experiments associated to that file
                indices, coords, distances, n_dist, index_size, index_time, search_time = load_neighbors_performance(directory_path + "/" + file)

                # Split the file name to get the information about the experiment
                # by '_'  and remove the '.hdf5' extension
                parts = file.split('_')
                parts[-1] = parts[-1].replace('.hdf5', '')

                # If the method is PDASC
                if parts[4] == 'PDASC':
                    # We store the information about the experiment associated with the file
                    results.append({
                        'Method': parts[4],
                        'Distance': distance_function,
                        # 'k': parts[2],
                        'radius': float(parts[7][1:]),
                        'n_nodes': parts[8][1:],
                        'Config': f'{parts[5]}_{parts[6]}', # As PDASc doest has more configuration parameters, we set it to NaN
                        # 'Implementation': parts[9],
                        'Dist_C(Av)': np.round(np.mean(n_dist), 2),
                        'Dist_C-Node(Av)': np.round(np.mean(n_dist/int(parts[8][1:])), 2),
                        # Get the recall of the experiment
                        'Recall(Av)': get_recall(dataset, parts[2], distance_function, indices, coords, distances),
                        'Search_T(s)': np.round(search_time, 2),
                        #'Query_T(Av)(s)': np.round(search_time/len(indices), 4),
                        'In_T(s)': np.round(index_time, 3),
                        'In_S(MB)': index_size

                    })

                # If the method is other
                else:
                    # We store the information about the experiment associated with the file
                    results.append({
                        'Method': parts[4],
                        'Distance': distance_function,
                        # 'k': parts[2],
                        'radius': None,
                        'n_nodes': 1,
                        'Config': '_'.join(parts[5:]), # Join all remaining parts to show the full configuration parameters
                        # 'Algorithm': parts[5] if parts[5] != 'hdf5' else None,
                        # 'Algorithm': parts[5] if not parts[4].endswith('.hdf5') else None,
                        # 'Implementation': None,
                        'Dist_C(Av)': np.round(np.mean(n_dist), 2),
                        'Dist_C-Node(Av)': np.round(np.mean(n_dist), 2),
                        # Get the recall of the experiment
                        'Recall(Av)': get_recall(dataset, parts[2], distance_function, indices, coords, distances),
                        'Search_T(s)': np.round(search_time, 4),
                        #'Query_T(Av)(s)': np.round(search_time / len(indices), 4),
                        'In_T(s)': np.round(index_time, 3),
                        'In_S(MB)': index_size,
                    })

        # Primero convierte la lista de dicts a DataFrame
        df = pd.DataFrame(results)

        # Define el orden personalizado de distancias
        custom_order = ['euclidean', 'manhattan', 'chebyshev', 'cosine']

        # Distancias adicionales (no incluidas explícitamente en custom_order)
        other_distances = sorted(set(df['Distance']) - set(custom_order))

        # Crea el tipo categórico ordenado
        distance_type = CategoricalDtype(categories=custom_order + other_distances, ordered=True)

        # Aplica el tipo al DataFrame y continúa con el flujo
        formatted_results = (
            df
            .assign(n_nodes=lambda d: d['n_nodes'].astype(int),
                    radius=lambda d: d['radius'].astype(float) if d['radius'] is not None else None,
                    Distance=lambda d: d['Distance'].astype(distance_type))
            .sort_values(by=['Method', 'Distance', 'n_nodes', 'radius', 'Config'], ascending=[True, True, True, True, True])
        )

        # Selección de columnas
        excel_results = formatted_results[['Distance', 'radius', 'Config', 'n_nodes', 'Dist_C(Av)', 'Dist_C-Node(Av)', 'Recall(Av)', 'Search_T(s)', 'In_T(s)', 'In_S(MB)']]

        # Orden final
        excel_results = excel_results.sort_values(by=['Distance', 'n_nodes', 'radius', 'Config'])

        # Añadir una columna de percentiles
        #percentiles = (1, 15, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99, 100)
        #excel_results['Percentil'] = percentiles

        # Reemplazo de decimales con comas
        for col in ['radius', 'Dist_C(Av)', 'Dist_C-Node(Av)', 'Recall(Av)', 'Search_T(s)', 'In_T(s)', 'In_S(MB)']:
            excel_results[col] = excel_results[col].astype(str).str.replace('.', ',', regex=False)

        # Añade una columna más al dataset con el percentil correpondiente a cada radio
        # Al radio mas alto le corresponde el ultimo percentil de la lista y al
        # radio mas bajo el primer percentil de la lista

        # Almacenar en un csv
        excel_results.to_csv(f'./ANN_Experiments/NearestNeighbors/{dataset}/benchmark_results_10nn_{dataset}.csv', index=False, sep=';')

        # Log the results
        logging.info('------------------------------------------------------------------------\n' + formatted_results.to_string())
        logging.shutdown()

        # Entre metodo y metodo, inserta una linea en blnco en el dataset
        # Entre cada metodo, inserta una linea en blanco en el dataset
        formatted_results = pd.concat([
            pd.concat([formatted_results.iloc[[i]],
                       pd.DataFrame([[''] * len(formatted_results.columns)],
                                    columns=formatted_results.columns)])
            if i < len(formatted_results) - 1 and formatted_results.iloc[i]['Method'] != formatted_results.iloc[i + 1][
                'Method']
            else formatted_results.iloc[[i]]
            for i in range(len(formatted_results))
        ]).reset_index(drop=True)

        # Return the results
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

