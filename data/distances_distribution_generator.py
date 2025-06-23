from fitter import Fitter
from fitter import get_common_distributions
from matplotlib import pyplot as plt

from data.dataset_analysis import load_random_sample, load_PDASC_sample, compute_distances_kth_nn, compute_distances_pairwise
import argparse
import os
import numpy as np
import pandas as pd
from plotnine import *

# Function to get the k-th nearest neighbour distances for a random sample and PDASC index prototypes
def get_nn_distances(dataset, distance_function, k_neighbours, sample_size, dataset_size):

    # Load a random sample of points from the dataset and compute the k-th nearest neighbour distances
    random_sample = load_random_sample(dataset, sample_size)
    random_complete = load_random_sample(dataset, dataset_size)

    # If the distance is 'haversine', we convert data to radians
    if distance_function == 'haversine':
        random_sample= np.radians(random_sample)
        random_complete = np.radians(random_complete)
        print("Converting data to radians for haversine distance")

    random_dists = np.sort(compute_distances_kth_nn(random_sample, random_complete, k_neighbours, distance_function))

    # Save the random distances to a CSV file
    pd.DataFrame(random_dists).to_csv(
        f'./data/dataset_analysis/{dataset}/{dataset}_{k_neighbours}th_nn_{distance_function}_{sample_size}_random.csv',
        index=False)

    # Load a random sample of prototypes from PDASC index and compute the k-th nearest neighbour distances
    pdasc_sample = load_PDASC_sample(dataset, sample_size, distance_function)
    pdasc_complete = load_PDASC_sample(dataset, dataset_size, distance_function)
    pdasc_dists = np.sort(compute_distances_kth_nn(pdasc_sample, pdasc_complete, k_neighbours, distance_function))

    # Save the distances to a CSV file
    pd.DataFrame(pdasc_dists).to_csv(
        f'./data/dataset_analysis/{dataset}/{dataset}_{k_neighbours}th_nn_{distance_function}_{sample_size}_PDASC.csv',
        index=False)

    return random_dists, pdasc_dists

# Function to get the pairwise distances for a random sample and PDASC index prototypes
def get_pairwise_distances(dataset, distance_function, sample_size, dataset_size):

    # Load a random sample of points from the dataset and compute the pairwise distances
    random_sample = load_random_sample(dataset, sample_size)
    # random_complete = load_random_sample(dataset, dataset_size)

    # If the distance is 'haversine', we convert data to radians
    if distance_function == 'haversine':
        random_sample= np.radians(random_sample)
        # random_complete = np.radians(random_complete)
        print("Converting data to radians for haversine distance")

    random_dists = np.sort(compute_distances_pairwise(random_sample, distance_function).flatten())

    # Save the random distances to a CSV file
    pd.DataFrame(random_dists).to_csv(
        f'./data/dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{sample_size}_random.csv',
        index=False)

    # Load a random sample of prototypes from PDASC index and compute the pairwise distances
    pdasc_sample = load_PDASC_sample(dataset, sample_size, distance_function)
    # pdasc_complete = load_PDASC_sample(dataset, dataset_size, distance_function)
    pdasc_dists = np.sort(compute_distances_pairwise(pdasc_sample, distance_function).flatten())

    # Save the distances to a CSV file
    pd.DataFrame(pdasc_dists).to_csv(
        f'./data/dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{sample_size}_PDASC.csv',
        index=False)

    return random_dists, pdasc_dists

def plot_cdfs_nn_complete(datasets, sample_size):

    k_neighbours = 10
    datasets_size = {
        "wdbc": 1000,
        "municipios": 8031,
        "MNIST": 59999,
        "NYtimes": 290000,
        "GLOVE": 1183413 #1183514
    }

    all_data = []

    for dataset, distance_function in datasets:
        print(f"Processing dataset: {dataset} with distance function: {distance_function}")

        # Load a random sample of points from the dataset and compute the k-th nearest neighbour distances
        random_sample = load_random_sample(dataset, sample_size)
        random_complete = load_random_sample(dataset, datasets_size[dataset])
        random_dists = np.sort(compute_distances_kth_nn(random_sample, random_complete, k_neighbours, distance_function))

        # Load a random sample of prototypes from PDASC index and compute the k-th nearest neighbour distances
        pdasc_sample = load_PDASC_sample(dataset, sample_size, distance_function)
        pdasc_complete = load_PDASC_sample(dataset, datasets_size[dataset], distance_function)
        pdasc_dists = np.sort(compute_distances_kth_nn(pdasc_sample, pdasc_complete, k_neighbours, distance_function))


        rand_cdf = np.arange(1, len(random_dists) + 1) / len(random_dists)
        pdasc_cdf = np.arange(1, len(pdasc_dists) + 1) / len(pdasc_dists)

        if dataset=="municipios":
            dataset="Municipalities"

        df_rand = pd.DataFrame({
            'distances': random_dists,
            'cdf': rand_cdf,
            'Distance': distance_function,
            'method': 'Random Elements Sample',
            'Dataset': dataset
        })

        df_pdasc = pd.DataFrame({
            'distances': pdasc_dists,
            'cdf': pdasc_cdf,
            'Distance': distance_function,
            'method': 'PDASC Prototypes Sample',
            'Dataset': dataset
        })

        all_data.append(df_rand)
        all_data.append(df_pdasc)

    df_all = pd.concat(all_data)


    # Colores personalizados
    custom_colors = {
        'Random Elements Sample': '#4C78A8',  # azul suave
        'PDASC Prototypes Sample': '#F58518'  # naranja suave
    }

    def col_func(s):

        distances = ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'haversine']
        datasets = ['Municipalities', 'MNIST', 'NYtimes', 'GLOVE']

        text=""
        if s in distances:
            text = str(s).capitalize() + " Distance"
        elif s in datasets:
            text = str(s) + " Dataset"
        return text


    # Reemplaza los nombres de las métricas en el DataFrame
    p = (
            ggplot(df_all, aes(x='distances', y='cdf', fill='method')) +
            geom_area(alpha=0.4, position='identity') +
            geom_line(aes(color='method'), size=1.1) +
            scale_fill_manual(values=custom_colors) +
            scale_color_manual(values=custom_colors) +
            facet_wrap('~ Dataset + Distance', labeller=col_func, ncol=2, scales='free_x') +
            labs(
                title=f'Empirical Cumulative Distribution Function of {k_neighbours}th Neighbour Distances',
                subtitle=f'(Random sample vs PDASC sample)',
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
                plot_title=element_text(ha='center'),
                plot_subtitle=element_text(ha='center'),
                axis_ticks_major_x=element_line(),
                axis_ticks_major_y=element_line(),
                axis_ticks_minor_x=element_line(color='gray', size=0.5),
                axis_ticks_minor_y=element_line(color='gray', size=0.5),
                axis_text_x=element_text(size=10),
                axis_text_y=element_text(size=10),
                strip_text_x=element_text(size=11, weight='bold', margin={'t': 10}),
            ) +
            scale_y_continuous(limits=(0, 1))
    )
    out_path = f'./data/dataset_analysis/'
    os.makedirs(out_path, exist_ok=True)
    filename = f'SISAP_{k_neighbours}thnn_cdf_comparision_overlap.png'
    p.save(os.path.join(out_path, filename), dpi=300)

# Plotting CDFs of distances to the k-th nearest neighbours for a random sample of elements and PDASC index prototypes
# For all dataset/distance combinations
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

# Plotting CDFs for k-th nearest neighbour distances in a dataset
def plot_cdfs_nn_dataset(dataset, distance_function, sample_size):

    k_neighbours = 10
    datasets_size = {
        "wdbc": 1000,
        "municipios": 8031,
        "MNIST": 59999,
        "NYtimes": 290000,
        "GLOVE": 1183514
    }


    print(f"Processing dataset: {dataset} with distance function: {distance_function}")

    sample_size = int(datasets_size[dataset] * (sample_size/100))
    dataset_size=datasets_size[dataset]

    # If this path exist,
    PDASC_path=f'./data/dataset_analysis/{dataset}/{dataset}_{k_neighbours}th_nn_{distance_function}_{sample_size}_PDASC.csv'
    random_path=f'./data/dataset_analysis/{dataset}/{dataset}_{k_neighbours}th_nn_{distance_function}_{sample_size}_random.csv'


    if not(os.path.exists(PDASC_path)) or not(os.path.exists(random_path)):
        # Compute the distances and save them to CSV files
        # print(f"\nComputing distances for {dataset} with {distance_function} distance function and sample size {sample_size}")
        random_dists, pdasc_dists = get_nn_distances(dataset, distance_function, k_neighbours, sample_size, dataset_size)

    else:
        # Load the distances from the CSV files
        # print(f"\nLoading distances for {dataset} with {distance_function} distance function and sample size {sample_size}")
        random_dists = pd.read_csv(random_path).values.flatten()
        pdasc_dists = pd.read_csv(PDASC_path).values.flatten()


    # Obtain the cumulative distribution functions (CDFs)
    rand_cdf = np.arange(1, len(random_dists) + 1) / len(random_dists)
    pdasc_cdf = np.arange(1, len(pdasc_dists) + 1) / len(pdasc_dists)


    # Define the percentiles to calculate in a logarithmic scale
    # percentiles = (0.8, 0.829, 0.859, 0.89, 0.922, 0.954, 0.987, 1.0)
    # Secuencia extendida con 6 valores más 0.647, 0.67, 0.694, 0.719, 0.745, 0.772, 0.8, 0.829, 0.859, 0.89, 0.922, 0.954, 0.987, 1.0
    percentiles = (0.647, 0.67, 0.694, 0.719, 0.745, 0.772, 0.8, 0.829, 0.859, 0.89, 0.922, 0.954, 0.987, 1.0)

    # Obtener los valores de las distancias en los percentiles
    # rand_percentile_values = np.percentile(random_dists, [p * 100 for p in percentiles])
    pdasc_percentile_values = np.percentile(pdasc_dists, [p * 100 for p in percentiles])


    # Seleccionar decimales según la métrica
    if distance_function in ['euclidean', 'manhattan']:
        decimales = 2
    elif distance_function == 'chebyshev':
        decimales = 3
    elif distance_function == 'cosine'or distance_function == 'haversine':
        decimales = 4
    else:
        decimales = 2

    # Imprimir los valores de cdfs redondeados y separados por espacio
    # print(f"Radius values for {distance_metric} experiments: {cdfs}")
    print(f"r_{dataset}_{distance_function}=\"" + " ".join([f"{cdf:.{decimales}f}" for cdf in pdasc_percentile_values]) + "\"")

    # Crear DataFrame con los percentiles y sus valores
    df_points = pd.DataFrame({
        'distance': pdasc_percentile_values,
        'cdf': percentiles
    })

    # Crear líneas verticales y horizontales desde df_points
    df_vlines = df_points.assign(xend=lambda df: df['distance'], y=0, yend=lambda df: df['cdf'])
    df_hlines = df_points.assign(x=0, xend=lambda df: df['distance'], yend=lambda df: df['cdf'])

    # Combinar CDFs
    df_all = pd.concat([
        pd.DataFrame({'distance': random_dists, 'cdf': rand_cdf, 'method': 'Random'}),
        pd.DataFrame({'distance': pdasc_dists, 'cdf': pdasc_cdf, 'method': 'PDASC'})
    ])

    # Colores
    custom_colors = {'Random': '#4C78A8', 'PDASC': '#F58518'}

    # Crear gráfico
    p = (
            ggplot(df_all, aes(x='distance', y='cdf', fill='method')) +
            geom_area(alpha=0.3, position='identity') +
            geom_line(aes(color='method'), size=1.1) +
            scale_fill_manual(values=custom_colors) +
            scale_color_manual(values=custom_colors) +

            # Líneas y puntos
            geom_segment(data=df_vlines, mapping=aes(x='distance', xend='xend', y='y', yend='yend'),
                         linetype='dashed', color='#3b3b3b', inherit_aes=False) +
            geom_segment(data=df_hlines, mapping=aes(x='x', xend='xend', y='cdf', yend='yend'),
                         linetype='dashed', color='#3b3b3b', inherit_aes=False) +
            geom_point(data=df_points, mapping=aes(x='distance', y='cdf'),
                       color='#3b3b3b', size=1.8, inherit_aes=False) +

            # Ejes y tema
            labs(
                x=f'{distance_function.capitalize()} Distance',
                y='Probability'
            ) +
            scale_y_continuous(limits=(0, 1), breaks=np.arange(0, 1.1, 0.1)) +
            scale_x_continuous(breaks=np.linspace(0, max(df_all['distance']), num=10)) +
            theme_minimal(base_size=13) +
            theme(
                figure_size=(8, 6),
                legend_position="none",
                panel_background=element_rect(fill='white', color='black'),
                plot_background=element_rect(fill='white', color='white'),
                panel_grid_major=element_line(color="#e5e5e5"),
                panel_grid_minor=element_line(color="#f5f5f5"),
                legend_title=element_blank(),
                plot_title=element_text(ha='center'),
                axis_ticks_major_x=element_line(),
                axis_ticks_major_y=element_line(),
                axis_text_x=element_text(size=10),
                axis_text_y=element_text(size=10, margin={'r': 5}),  # Margen entre eje y y etiquetas
                axis_title_x=element_text(margin={'t': 20}),  # Añade margen superior para separar el título X del eje X
                axis_title_y = element_text(margin={'r': 20}),  # Añade margen derecho para separar el título Y del eje Y
            )
    )

    # Guardar
    out_path = f'./data/dataset_analysis/{dataset}'
    os.makedirs(out_path, exist_ok=True)
    filename = f'{dataset}_{distance_function}_{k_neighbours}th_nn_comparision_overlap_{len(random_dists)}.png'
    p.save(os.path.join(out_path, filename), dpi=300)

# Plotting CDFs for pairwise distances in a dataset
def plot_cdfs_pairwise_dataset(dataset, distance_function, sample_size):

    datasets_size = {
        "wdbc": 1000,
        "municipios": 8031,
        "MNIST": 59999,
        "NYtimes": 290000,
        "GLOVE": 1183514
    }

    print(f"Processing dataset: {dataset} with distance function: {distance_function}")

    sample_size = int(datasets_size[dataset] * (sample_size / 100))
    dataset_size = datasets_size[dataset]


    # Paths for the PDASC and random distances
    PDASC_path=f'./data/dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{sample_size}_PDASC.csv'
    random_path=f'./data/dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{sample_size}_random.csv'

    # If the paths do not exist, compute the distances
    if not(os.path.exists(PDASC_path)) or not(os.path.exists(random_path)):
        # Compute the distances and save them to CSV files
        # print(f"\nComputing distances for {dataset} with {distance_function} distance function and sample size {sample_size}")
        random_dists, pdasc_dists = get_pairwise_distances(dataset, distance_function, sample_size, dataset_size)

    else:
        # If it exists, load the distances from the CSV files
        # print(f"\nLoading distances for {dataset} with {distance_function} distance function and sample size {sample_size}")
        random_dists = pd.read_csv(random_path).values.flatten()
        pdasc_dists = pd.read_csv(PDASC_path).values.flatten()


    # Obtain the cumulative distribution functions (CDFs)
    rand_cdf = np.arange(1, len(random_dists) + 1) / len(random_dists)
    pdasc_cdf = np.arange(1, len(pdasc_dists) + 1) / len(pdasc_dists)

    # Define the percentiles to calculate in a logarithmic scale
    #percentiles = (0.647, 0.67, 0.694, 0.719, 0.745, 0.772, 0.8, 0.829, 0.859, 0.89, 0.922, 0.954, 0.987, 1.0)
    #rand_percentile_values = np.percentile(rand_dists, [p * 100 for p in percentiles])
    #pdasc_percentile_values = np.percentile(pdasc_dists, [p * 100 for p in percentiles])

    # Define the percentiles to calculate in a logarithmic scale from 15 to 100
    #percentiles = (1, 14.83, 27.49, 38.28, 47.49, 55.35, 62.05, 67.77, 72.65, 76.82, 80.37, 83.4, 85.99, 88.19, 90.07, 91.68, 93.05, 94.22, 95.22, 96.07, 96.79, 97.41, 97.94, 98.39, 98.78, 99.1, 99.38, 99.62, 99.83, 100.0)

    # Define the percentiles to calculate in an heuristic way
    percentiles = (1, 15, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99, 100)
    # Obtener los valores de las distancias en los percentiles
    # rand_percentile_values = np.percentile(random_dists, percentiles)
    pdasc_percentile_values = np.percentile(pdasc_dists, percentiles)

    # Seleccionar decimales según la métrica
    if distance_function in ['euclidean', 'manhattan']:
        decimales = 2
    elif distance_function == 'chebyshev':
        decimales = 3
    elif distance_function == 'cosine' or distance_function == 'haversine':
        decimales = 4
    else:
        decimales = 2

    # Imprimir los valores de cdfs redondeados y separados por espacio
    # print(f"Radius values for {distance_metric} experiments: {cdfs}")
    print(f"r_{dataset}_{distance_function}=\"" + " ".join(
        [f"{cdf:.{decimales}f}" for cdf in pdasc_percentile_values]) + "\"")

    # Crear DataFrame con los percentiles y sus valores
    df_points = pd.DataFrame({
        'distance': pdasc_percentile_values,
        'cdf': [p / 100 for p in percentiles]
    })

    # Crear líneas verticales y horizontales desde df_points
    df_vlines = df_points.assign(xend=lambda df: df['distance'], y=0, yend=lambda df: df['cdf'])
    df_hlines = df_points.assign(x=0, xend=lambda df: df['distance'], yend=lambda df: df['cdf'])

    # Combinar CDFs
    df_all = pd.concat([
        pd.DataFrame({'distance': random_dists, 'cdf': rand_cdf, 'method': 'Random'}),
        pd.DataFrame({'distance': pdasc_dists, 'cdf': pdasc_cdf, 'method': 'PDASC'})
    ])

    # Colores
    custom_colors = {'Random': '#4C78A8', 'PDASC': '#F58518'}

    # Crear gráfico
    p = (
            ggplot(df_all, aes(x='distance', y='cdf', fill='method')) +
            geom_area(alpha=0.3, position='identity') +
            geom_line(aes(color='method'), size=1.1) +
            scale_fill_manual(values=custom_colors) +
            scale_color_manual(values=custom_colors) +

            # Líneas y puntos
            geom_segment(data=df_vlines, mapping=aes(x='distance', xend='xend', y='y', yend='yend'),
                        linetype='dashed', color='black', inherit_aes=False) +
            geom_segment(data=df_hlines, mapping=aes(x='x', xend='xend', y='cdf', yend='yend'),
                        linetype='dashed', color='black', inherit_aes=False) +
            geom_point(data=df_points, mapping=aes(x='distance', y='cdf'),
                        color='black', size=1.8, inherit_aes=False) +

            # color='#3b3b3b' = grey

            # Ejes y tema
            labs(
                x=f'{distance_function.capitalize()} Distance',
                y='Probability'
            ) +
            scale_y_continuous(limits=(0, 1), breaks=np.arange(0, 1.1, 0.1)) +
            scale_x_continuous(breaks=np.linspace(0, max(df_all['distance']), num=10)) +
            theme_minimal(base_size=13) +
            theme(
                figure_size=(8, 6),
                legend_position="none",
                panel_background=element_rect(fill='white', color='black'),
                plot_background=element_rect(fill='white', color='white'),
                panel_grid_major=element_line(color="#e5e5e5"),
                panel_grid_minor=element_line(color="#f5f5f5"),
                legend_title=element_blank(),
                plot_title=element_text(ha='center'),
                axis_ticks_major_x=element_line(),
                axis_ticks_major_y=element_line(),
                axis_text_x=element_text(size=10),
                axis_text_y=element_text(size=10, margin={'r': 5}),  # Margen entre eje y y etiquetas
                axis_title_x=element_text(margin={'t': 20}),  # Añade margen superior para separar el título X del eje X
                axis_title_y=element_text(margin={'r': 20}),  # Añade margen derecho para separar el título Y del eje Y
            )
    )

    # Guardar
    out_path = f'./data/dataset_analysis/{dataset}'
    os.makedirs(out_path, exist_ok=True)
    filename = f'{dataset}_{distance_function}_pairwise_comparision_overlap_{sample_size}.png'
    p.save(os.path.join(out_path, filename), dpi=300)

# Plotting PDFs for pairwise distances in a dataset
def plot_pdfs_pairwise_dataset(dataset, distance_function, sample_size):

    datasets_size = {
        "wdbc": 1000,
        "municipios": 8031,
        "MNIST": 59999,
        "NYtimes": 290000,
        "GLOVE": 1183514
    }

    print(f"Processing dataset: {dataset} with distance function: {distance_function}")

    sample_size = int(datasets_size[dataset] * (sample_size / 100))
    dataset_size = datasets_size[dataset]

    # Paths for the PDASC and random distances
    PDASC_path = f'./data/dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{sample_size}_PDASC.csv'
    random_path = f'./data/dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{sample_size}_random.csv'

    # If the paths do not exist, compute the distances
    if not (os.path.exists(PDASC_path)) or not (os.path.exists(random_path)):
        # Compute the distances and save them to CSV files
        random_dists, pdasc_dists = get_pairwise_distances(dataset, distance_function, sample_size,
                                                               dataset_size)
    else:
        # If it exists, load the distances from the CSV files
        random_dists = pd.read_csv(random_path).values.flatten()
        pdasc_dists = pd.read_csv(PDASC_path).values.flatten()

    # Combine distances into a DataFrame
    df_all = pd.concat([
        pd.DataFrame({'distance': random_dists, 'method': 'Random'}),
        pd.DataFrame({'distance': pdasc_dists, 'method': 'PDASC'})
    ])

    # Colors
    custom_colors = {'Random': '#4C78A8', 'PDASC': '#F58518'}

    # Create the plot
    p = (
            ggplot(df_all, aes(x='distance', fill='method')) +
            geom_density(alpha=0.5) +
            scale_fill_manual(values=custom_colors) +
            labs(
                title=f'Probability Density Function of Pairwise Distances',
                x=f'{distance_function.capitalize()} Distance',
                y='Density'
            ) +
            theme_minimal(base_size=13) +
            theme(
                figure_size=(8, 6),
                legend_position="none",
                panel_background=element_rect(fill='white', color='black'),
                plot_background=element_rect(fill='white', color='white'),
                panel_grid_major=element_line(color="#e5e5e5"),
                panel_grid_minor=element_line(color="#f5f5f5"),
                legend_title=element_blank(),
                plot_title=element_text(ha='center'),
                axis_ticks_major_x=element_line(),
                axis_ticks_major_y=element_line(),
                axis_text_x=element_text(size=10),
                axis_text_y=element_text(size=10, margin={'r': 5}),
                axis_title_x=element_text(margin={'t': 20}),
                axis_title_y=element_text(margin={'r': 20}),
            )
    )

    # Save the plot
    out_path = f'./data/dataset_analysis/{dataset}'
    os.makedirs(out_path, exist_ok=True)
    filename = f'{dataset}_{distance_function}_pairwise_pdf_comparision_{sample_size}.png'
    p.save(os.path.join(out_path, filename), dpi=300)

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

# Plotting CDFs of pairwise distances for different sample sizes in a dataset
def plot_cdfs_nn_samplesize_dataset(dataset, distance_function):

    k_neighbours = 10
    datasets_size = {
        "wdbc": 1000,
        "municipios": 8031,
        "MNIST": 59999,
        "NYtimes": 290000,
        "GLOVE": 1200000  # 1183514
    }

    all_data = []

    percentages = (0.2, 0.5, 1, 1.2, 9.999)
    #create a dictionare with the percentages as keys and the sample size as values
    sample_sizes = {}

    for percentage in percentages:
        if datasets_size[dataset] > 100000:
            sample_size = int(datasets_size[dataset] * percentage * 0.01)
            sample_sizes[percentage] = sample_size
        else:
            sample_size = int(datasets_size[dataset] * percentage * 0.1)
            sample_sizes[percentage] = sample_size

    random_complete = load_random_sample(dataset, datasets_size[dataset])
    pdasc_complete = load_PDASC_sample(dataset, datasets_size[dataset], distance_function)

    for percentage in sample_sizes.keys():
        sample_size= sample_sizes[percentage]
        print(sample_size)

        # Load a random sample of points from the dataset and compute the k-th nearest neighbour distances
        random_sample = load_random_sample(dataset, sample_size)
        random_dists = np.sort(compute_distances_kth_nn(random_sample, random_complete, k_neighbours, distance_function))

        # Load a random sample of prototypes from PDASC index and compute the k-th nearest neighbour distances
        pdasc_sample=load_PDASC_sample(dataset, sample_size, distance_function)
        pdasc_dists = np.sort(compute_distances_kth_nn(pdasc_sample, pdasc_complete, k_neighbours, distance_function))


        rand_cdf = np.arange(1, len(random_dists) + 1) / len(random_dists)
        pdasc_cdf = np.arange(1, len(pdasc_dists) + 1) / len(pdasc_dists)



        df_rand = pd.DataFrame({
            'distances': random_dists,
            'cdf': rand_cdf,
            'Distance': distance_function,
            'method': 'Random Elements Sample',
            'Dataset': dataset,
            'sample_size': sample_size
        })

        df_pdasc = pd.DataFrame({
            'distances': pdasc_dists,
            'cdf': pdasc_cdf,
            'Distance': distance_function,
            'method': 'PDASC Prototypes Sample',
            'Dataset': dataset,
            'sample_size': sample_size
        })

        all_data.append(df_rand)
        all_data.append(df_pdasc)

    df_all = pd.concat(all_data)

    # Colores personalizados
    custom_colors = {
        'Random Elements Sample': '#4C78A8',  # azul suave
        'PDASC Prototypes Sample': '#F58518'  # naranja suave
    }

    # Reemplaza los nombres de las métricas en el DataFrame
    p = (
            ggplot(df_all, aes(x='distances', y='cdf', color='factor(sample_size)', group='factor(sample_size)')) +
            geom_area(alpha=0, position='identity') +
            geom_line(size=1.1) +
            scale_color_discrete(name='Sample Size') +
            facet_wrap('~ method', ncol=2, scales='free_x') +
            labs(
                title=f'Empirical Cumulative Distribution Function of {k_neighbours}th Neighbour Distances',
                subtitle=f'{dataset} dataset',
                x='',
                y='Probability'
            ) +
            theme_minimal(base_size=13) +
            theme(
                figure_size=(12, 9),
                legend_position="none",
                panel_background=element_rect(fill='white', color='black'),
                plot_background=element_rect(fill='white', color='white'),
                legend_title=element_blank(),
                plot_title=element_text(ha='center'),
                plot_subtitle=element_text(ha='center'),
                axis_ticks_major_x=element_line(),
                axis_ticks_major_y=element_line(),
                axis_ticks_minor_x=element_line(color='gray', size=0.5),
                axis_ticks_minor_y=element_line(color='gray', size=0.5),
                axis_text_x=element_text(size=10),
                axis_text_y=element_text(size=10),
                strip_text_x=element_text(size=11, weight='bold', margin={'t': 10}),
            ) +
            scale_y_continuous(limits=(0, 1))
    )
    out_path = f'./data/dataset_analysis/{dataset}'
    os.makedirs(out_path, exist_ok=True)
    filename = f'SISAP_{k_neighbours}thnn_cdf_comparision_{dataset}.png'
    p.save(os.path.join(out_path, filename), dpi=300)

# Plotting CDFs of k-NN distances for different sample sizes in a dataset
def plot_cdfs_pairwise_samplesize_dataset(dataset, distance_function):

    datasets_size = {
        "wdbc": 1000,
        "municipios": 8031,
        "MNIST": 59999,
        "NYtimes": 290000,
        "GLOVE": 1200000  # 1183514
    }

    all_data = []

    percentages = (0.2, 0.5, 1, 1.2)
    #create a dictionare with the percentages as keys and the sample size as values
    sample_sizes = {}

    for percentage in percentages:
        if datasets_size[dataset] > 100000:
            sample_size = int(datasets_size[dataset] * percentage * 0.01)
            sample_sizes[percentage] = sample_size
        else:
            sample_size = int(datasets_size[dataset] * percentage * 0.1)
            sample_sizes[percentage] = sample_size


    for percentage in sample_sizes.keys():
        sample_size= sample_sizes[percentage]

        # Load a random sample of points from the dataset and compute the k-th nearest neighbour distances
        random_sample=load_random_sample(dataset, sample_size)
        random_dists = np.sort(compute_distances_pairwise(random_sample, distance_function).flatten())

        # Load a random sample of prototypes from PDASC index and compute the k-th nearest neighbour distances
        pdasc_sample = load_PDASC_sample(dataset, sample_size, distance_function)
        pdasc_dists = np.sort(compute_distances_pairwise(pdasc_sample, distance_function).flatten())


        rand_cdf = np.arange(1, len(random_dists) + 1) / len(random_dists)
        pdasc_cdf = np.arange(1, len(pdasc_dists) + 1) / len(pdasc_dists)



        df_rand = pd.DataFrame({
            'distances': random_dists,
            'cdf': rand_cdf,
            'Distance': distance_function,
            'method': 'Random Elements Sample',
            'Dataset': dataset,
            'sample_size': sample_size
        })

        df_pdasc = pd.DataFrame({
            'distances': pdasc_dists,
            'cdf': pdasc_cdf,
            'Distance': distance_function,
            'method': 'PDASC Prototypes Sample',
            'Dataset': dataset,
            'sample_size': sample_size
        })

        all_data.append(df_rand)
        all_data.append(df_pdasc)

    df_all = pd.concat(all_data)

    # Colores personalizados
    custom_colors = {
        'Random Elements Sample': '#4C78A8',  # azul suave
        'PDASC Prototypes Sample': '#F58518'  # naranja suave
    }

    # Reemplaza los nombres de las métricas en el DataFrame
    p = (
            ggplot(df_all, aes(x='distances', y='cdf', color='factor(sample_size)', group='factor(sample_size)')) +
            geom_area(alpha=0, position='identity') +
            geom_line(size=1.1) +
            scale_color_discrete(name='Sample Size') +
            facet_wrap('~ method', ncol=2, scales='free_x') +
            labs(
                title=f'Empirical Cumulative Distribution Function of pairwise Distances',
                subtitle=f'{dataset} dataset',
                x='',
                y='Probability'
            ) +
            theme_minimal(base_size=13) +
            theme(
                figure_size=(12, 9),
                legend_position="none",
                panel_background=element_rect(fill='white', color='black'),
                plot_background=element_rect(fill='white', color='white'),
                legend_title=element_blank(),
                plot_title=element_text(ha='center'),
                plot_subtitle=element_text(ha='center'),
                axis_ticks_major_x=element_line(),
                axis_ticks_major_y=element_line(),
                axis_ticks_minor_x=element_line(color='gray', size=0.5),
                axis_ticks_minor_y=element_line(color='gray', size=0.5),
                axis_text_x=element_text(size=10),
                axis_text_y=element_text(size=10),
                strip_text_x=element_text(size=11, weight='bold', margin={'t': 10}),
            ) +
            scale_y_continuous(limits=(0, 1))
    )
    out_path = f'./data/dataset_analysis/{dataset}'
    os.makedirs(out_path, exist_ok=True)
    filename = f'SISAP_pairwise_cdf_comparision_{dataset}.png'
    p.save(os.path.join(out_path, filename), dpi=300)


if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-cdfsNN", help="Perform kNN-CDF analysis with a tuple or a set of tuples",
                        type=eval, nargs='+')
    parser.add_argument("-cdfsNN_SS", help="Perform kNN-CDF analysis for a single dataset and distance function",
                        type=eval)
    parser.add_argument("-cdfsPW", help="Perform Pairwise Distances CDF analysis with a tuple or a set of tuples",
                        type=eval, nargs='+')
    parser.add_argument("-cdfsPW_SS", help="Perform Pairwise Distances CDF analysis with varying sample sizes",
                        type=eval)
    parser.add_argument("-pdfsPW", help="Perform Pairwise Distances PDF analysis with a tuple or a set of tuples",
                        type=eval, nargs='+')
    parser.add_argument("-size", help="Indicate the size of the sample to be used.", type=int)

    args = parser.parse_args()


    if args.cdfsNN:
        # Itera sobre las tuplas proporcionadas
        for item in args.cdfsNN:
            # Verifica que cada elemento sea una tupla
            if isinstance(item, tuple):
                print(f"Tupla recibida: {item}")
            else:
                print(f"Error: {item} no es una tupla.")
                exit(1)

        if len(args.cdfsNN) == 1:
            print(f"Received a single tuple: {args.cdfsNN}")
            dataset=args.cdfsNN[0][0]
            distance_function = args.cdfsNN[0][1]
            sample_size = args.size
            plot_cdfs_nn_dataset(dataset, distance_function, sample_size)

        elif len(args.cdfsNN) > 1:
            print(f"Received a set of tuples: {args.cdfsNN}")
            datasets=args.cdfsNN
            sample_size = args.size
            plot_cdfs_nn_complete(datasets, sample_size)

        else:
            print("Error: The argument for -cdfNN must be a tuple or a list of tuples.")
            exit(1)

    elif args.cdfsNN_SS:
        if len(args.cdfsNN_SS) == 1:
            print(f"Received a single tuple: {args.cdfsNN_SS}")
            dataset=args.cdfsNN_SS[0][0]
            distance_function = args.cdfsNN_SS[0][1]
            plot_cdfs_nn_samplesize_dataset(dataset,distance_function)

        else:
            print("Error: The argument for --cdfNN_SS must be a tuple")
            exit(1)

    elif args.cdfsPW:
        if len(args.cdfsPW) == 1:
            print(f"Received a single tuple: {args.cdfsPW}")
            dataset=args.cdfsPW[0][0]
            distance_function = args.cdfsPW[0][1]
            sample_size = args.size
            plot_cdfs_pairwise_dataset(dataset, distance_function, sample_size)

    elif args.cdfsPW_SS:
        if len(args.cdfsPW_SS) == 1:
            print(f"Received a single tuple: {args.cdfsPW_SS}")
            dataset=args.cdfsPW_SS[0][0]
            distance_function = args.cdfsPW_SS[0][1]
            plot_cdfs_pairwise_samplesize_dataset(dataset,distance_function)

        else:
            print("Error: The argument for --cdfPW_SS must be a tuple")
            exit(1)

    elif args.pdfsPW:
        if len(args.pdfsPW) == 1:
            print(f"Received a single tuple: {args.pdfsPW}")
            dataset=args.pdfsPW[0][0]
            distance_function = args.pdfsPW[0][1]
            sample_size = args.size
            plot_pdfs_pairwise_dataset(dataset, distance_function, sample_size)

        else:
            print("Error: The argument for --pdfsPW must be a tuple")
            exit(1)

    exit(0)
