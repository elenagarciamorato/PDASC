from benchmarks.plotting.performance_utils import *
from benchmarks.plotting.draw_benchmark_plots import print_Recall_pointplot
import argparse
import datetime
import logging
from pandas.api.types import CategoricalDtype


# Load the performance of the k-nn experiments regarding the selected dataset
def explore_experiments(dataset, optional_filters=None):

    # Set the directory path to load the experiments according to the dataset provided
    directory_path = "./benchmarks/NearestNeighbors/" + dataset

    results = []

    # For every .hdf5 file in the directory (file containing the neighbors and performances)
    for root, _, files in os.walk(directory_path):

        # Apply optional filter if provided
        if optional_filters:
            filter_options = optional_filters
            files = [f for f in files if all(opt in f for opt in filter_options)]

        for file in files:
            if file.endswith('.hdf5'):

                # Load the neighbors and performance of the k-nn experiments associated to that file
                indices, coords, distances, n_dist, search_time = load_neighbors_performance(directory_path + "/" + file)

                # Split the file name to get the information about the experiment
                # by '_' and '.' characters, except if the '.' is between two digits
                parts = re.split(r'[_]|(?<!\d)\.(?!\d)', file)

                # If the method is PDASC
                if parts[4] == 'PDASC':
                    # We store the information about the experiment associated with the file
                    results.append({
                        'Method': parts[4],
                        'Distance': parts[3],
                        'k': parts[2],
                        'radius': float(parts[7][1:]),
                        'Algorithm': parts[8],
                        # 'Implementation': parts[9],
                        'Dist_Computed(Av)': np.mean(n_dist),
                        # Get the recall of the experiment
                        'Recall(Av)': get_recall_new(dataset, parts[2], parts[3], indices, coords, distances),
                        'Search_Time': np.round(search_time, 8)
                    })

                # If the method is other
                else:
                    # We store the information about the experiment associated with the file
                    results.append({
                        'Method': parts[4],
                        'Distance': parts[3],
                        'k': parts[2],
                        'radius': None,
                        'Algorithm': parts[5] if parts[5] != 'hdf5' else None,
                        #'Algorithm': parts[5] if not parts[4].endswith('.hdf5') else None,
                        # 'Implementation': None,
                        'Dist_Computed(Av)': np.mean(n_dist),
                        # Get the recall of the experiment
                        'Recall(Av)': get_recall_new(dataset, parts[2], parts[3], indices, coords, distances),
                        'Search_Time': np.round(search_time, 8)
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
            .assign(k=lambda d: d['k'].astype(int),
                    Distance=lambda d: d['Distance'].astype(distance_type))
            .sort_values(by=['Method', 'Distance', 'radius', 'k'], ascending=[True, True, True, True])
        )

        # Selección de columnas
        excel_results = formatted_results[['Distance', 'radius', 'Dist_Computed(Av)', 'Recall(Av)', 'Search_Time']]

        # Orden final
        excel_results = excel_results.sort_values(by=['Distance', 'radius'])

        # Añadir una columna de percentiles
        percentiles = (1, 15, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99, 100)
        excel_results['Percentil'] = percentiles

        # Reemplazo de decimales con comas
        for col in ['Dist_Computed(Av)', 'Recall(Av)', 'Search_Time', 'radius']:
            excel_results[col] = excel_results[col].astype(str).str.replace('.', ',', regex=False)

        # Añade una columna más al dataset con el percentil correpondiente a cada radio
        # Al radio mas alto le corresponde el ultimo percentil de la lista y al
        # radio mas bajo el primer percentil de la lista

        # Almacenar en un csv
        excel_results.to_csv(f'./benchmarks/NearestNeighbors/{dataset}/benchmark_results_{formatted_results.iloc[0]["k"]}nn_{dataset}.csv', index=False, sep=';')

        # Log the results
        logging.info('------------------------------------------------------------------------\n' + formatted_results.to_string())
        logging.shutdown()

        # Return the results
        return formatted_results

if __name__ == "__main__":

    #dataset=("wdbc")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Name of the dataset whose results would be benchmarked", type=str)
    parser.add_argument("optional_filters", help="Benchmark optional filters", nargs='*', default=[])

    args = parser.parse_args()

    # Create a log file to store the performance of the k-nn experiments
    # Get the current date and time, formatting it
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%d-%m-%Y_%H:%M")

    logging.basicConfig(
        filename="./benchmarks/logs/" + args.dataset + "/benchmark_knn_" + args.dataset + "_" + str(formatted_time) + ".log",
        filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)

    logging.info('------------------------------------------------------------------------')
    logging.info('                    %s Dataset Benchmarking', args.dataset)
    logging.info('------------------------------------------------------------------------')

    # Explore the results of the experiments regarding the dataset provided
    df = explore_experiments(args.dataset, args.optional_filters)

    # Print the results
    print(df.to_string())

    print_Recall_pointplot(args.dataset, df)  # pointplot

    exit(0)

