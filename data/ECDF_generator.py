from data.dataset_analysis import load_random_sample, load_PDASC_sample, get_distances_kth_nn, get_distances_pairwise
import argparse
import os
import numpy as np
import pandas as pd
from plotnine import *


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
        random_dists = np.sort(get_distances_kth_nn(random_sample, random_complete, k_neighbours, distance_function))

        # Load a random sample of prototypes from PDASC index and compute the k-th nearest neighbour distances
        pdasc_sample = load_PDASC_sample(dataset, sample_size, distance_function)
        pdasc_complete = load_PDASC_sample(dataset, datasets_size[dataset], distance_function)
        pdasc_dists = np.sort(get_distances_kth_nn(pdasc_sample, pdasc_complete, k_neighbours, distance_function))


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

def compute_nn_distances(dataset, distance_function, k_neighbours, sample_size, dataset_size):

    # Load a random sample of points from the dataset and compute the k-th nearest neighbour distances
    random_sample = load_random_sample(dataset, sample_size)
    random_complete = load_random_sample(dataset, dataset_size)

    # If the distance is 'haversine', we convert data to radians
    if distance_function == 'haversine':
        random_sample= np.radians(random_sample)
        random_complete = np.radians(random_complete)
        print("Converting data to radians for haversine distance")

    random_dists = np.sort(get_distances_kth_nn(random_sample, random_complete, k_neighbours, distance_function))

    # Save the random distances to a CSV file
    pd.DataFrame(random_dists).to_csv(
        f'./data/dataset_analysis/{dataset}/{dataset}_{k_neighbours}th_nn_{distance_function}_{sample_size}_random.csv',
        index=False)

    # Load a random sample of prototypes from PDASC index and compute the k-th nearest neighbour distances
    pdasc_sample = load_PDASC_sample(dataset, sample_size, distance_function)
    pdasc_complete = load_PDASC_sample(dataset, dataset_size, distance_function)
    pdasc_dists = np.sort(get_distances_kth_nn(pdasc_sample, pdasc_complete, k_neighbours, distance_function))

    # Save the distances to a CSV file
    pd.DataFrame(pdasc_dists).to_csv(
        f'./data/dataset_analysis/{dataset}/{dataset}_{k_neighbours}th_nn_{distance_function}_{sample_size}_PDASC.csv',
        index=False)

    return random_dists, pdasc_dists

def compute_pairwise_distances(dataset, distance_function, sample_size, dataset_size):

    # Load a random sample of points from the dataset and compute the pairwise distances
    random_sample = load_random_sample(dataset, sample_size)
    random_complete = load_random_sample(dataset, dataset_size)

    # If the distance is 'haversine', we convert data to radians
    if distance_function == 'haversine':
        random_sample= np.radians(random_sample)
        random_complete = np.radians(random_complete)
        print("Converting data to radians for haversine distance")

    random_dists = np.sort(get_distances_pairwise(random_sample, distance_function).flatten())

    # Save the random distances to a CSV file
    pd.DataFrame(random_dists).to_csv(
        f'./data/dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{sample_size}_random.csv',
        index=False)

    # Load a random sample of prototypes from PDASC index and compute the pairwise distances
    pdasc_sample = load_PDASC_sample(dataset, sample_size, distance_function)
    pdasc_complete = load_PDASC_sample(dataset, dataset_size, distance_function)
    pdasc_dists = np.sort(get_distances_pairwise(pdasc_sample, distance_function).flatten())

    # Save the distances to a CSV file
    pd.DataFrame(pdasc_dists).to_csv(
        f'./data/dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{sample_size}_PDASC.csv',
        index=False)

    return random_dists, pdasc_dists
# As the previous function but for individual plots
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
        random_dists, pdasc_dists = compute_nn_distances(dataset, distance_function, k_neighbours, sample_size, dataset_size)

    else:
        # Load the distances from the CSV files
        # print(f"\nLoading distances for {dataset} with {distance_function} distance function and sample size {sample_size}")
        random_dists = pd.read_csv(random_path).values.flatten()
        pdasc_dists = pd.read_csv(PDASC_path).values.flatten()


    # Obtain the cumulative distribution functions (CDFs)
    rand_cdf = np.arange(1, len(random_dists) + 1) / len(random_dists)
    pdasc_cdf = np.arange(1, len(pdasc_dists) + 1) / len(pdasc_dists)

    # Define the percentiles to calculate
    percentiles = (0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1)

    # Define the percentiles in a logarithmic scale
    # percentiles = (0.700, 0.732, 0.765, 0.801, 0.837, 0.876, 0.916, 0.957, 1.000)
    percentiles = (0.8, 0.829, 0.859, 0.89, 0.922, 0.954, 0.987, 1.0)

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
                legend_position='None',
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
        random_dists, pdasc_dists = compute_pairwise_distances(dataset, distance_function, sample_size, dataset_size)

    else:
        # If it exists, load the distances from the CSV files
        # print(f"\nLoading distances for {dataset} with {distance_function} distance function and sample size {sample_size}")
        random_dists = pd.read_csv(random_path).values.flatten()
        pdasc_dists = pd.read_csv(PDASC_path).values.flatten()


    # Obtain the cumulative distribution functions (CDFs)
    rand_cdf = np.arange(1, len(random_dists) + 1) / len(random_dists)
    pdasc_cdf = np.arange(1, len(pdasc_dists) + 1) / len(pdasc_dists)

    # Define the percentiles to calculate
    percentiles = (0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1)

    # Define the percentiles in a logarithmic scale
    # percentiles = (0.700, 0.732, 0.765, 0.801, 0.837, 0.876, 0.916, 0.957, 1.000)
    #Secuencia extendida con 4 valores más z 0.694, 0.719, 0.745, 0.772, 0.8, 0.829, 0.859, 0.89, 0.922, 0.954, 0.987, 1.0
    percentiles = (0.8, 0.829, 0.859, 0.89, 0.922, 0.954, 0.987, 1.0)

    # Obtener los valores de las distancias en los percentiles
    # rand_percentile_values = np.percentile(random_dists, [p * 100 for p in percentiles])
    pdasc_percentile_values = np.percentile(pdasc_dists, [p * 100 for p in percentiles])

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
                legend_position='None',
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
        random_dists = np.sort(get_distances_kth_nn(random_sample, random_complete, k_neighbours, distance_function))

        # Load a random sample of prototypes from PDASC index and compute the k-th nearest neighbour distances
        pdasc_sample=load_PDASC_sample(dataset, sample_size, distance_function)
        pdasc_dists = np.sort(get_distances_kth_nn(pdasc_sample, pdasc_complete, k_neighbours, distance_function))


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
                legend_position='bottom',
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
        random_dists = np.sort(get_distances_pairwise(random_sample, distance_function).flatten())

        # Load a random sample of prototypes from PDASC index and compute the k-th nearest neighbour distances
        pdasc_sample = load_PDASC_sample(dataset, sample_size, distance_function)
        pdasc_dists = np.sort(get_distances_pairwise(pdasc_sample, distance_function).flatten())


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
                legend_position='none',
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

    exit(0)
