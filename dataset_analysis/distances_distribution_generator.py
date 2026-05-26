from plotnine.themes.themeable import legend_position

from PDASC.pdasc_flues_ import simulate_flue_partitioning
from dataset_analysis.dataset_analysis import load_random_sample, load_random_sample_flue, load_PDASC_sample, compute_distances_kth_nn, compute_distances_pairwise
import argparse
import os
import numpy as np
import pandas as pd
from plotnine import *
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=DataConversionWarning)

from data.load_train_test_set import load_train_test_h5py, load_hdf5


# Function to get the k-th nearest neighbour distances for a random sample and PDASC index prototypes
def get_nn_distances(dataset, distance_function, k_neighbours, sample_size, nc, tg):

    # Load a random sample of points from the dataset and compute the k-th nearest neighbour distances
    random_sample = load_random_sample(dataset, sample_size)
    random_complete = load_random_sample(dataset, len(dataset))

    # If the distance is 'haversine', we convert data to radians
    if distance_function == 'haversine':
        random_sample= np.radians(random_sample)
        random_complete = np.radians(random_complete)
        print("Converting data to radians for haversine distance")

    random_dists = np.sort(compute_distances_kth_nn(random_sample, random_complete, k_neighbours, distance_function))

    # Save the random distances to a CSV file
    pd.DataFrame(random_dists).to_csv(
        f'./dataset_analysis/{dataset}/{dataset}_{k_neighbours}th_nn_{distance_function}_{sample_size}_random.csv',
        index=False)

    # Load a random sample of prototypes from PDASC index and compute the k-th nearest neighbour distances
    pdasc_sample = load_PDASC_sample(dataset, sample_size, distance_function, nc, tg)
    pdasc_complete = load_PDASC_sample(dataset, len(dataset), distance_function, nc, tg)
    pdasc_dists = np.sort(compute_distances_kth_nn(pdasc_sample, pdasc_complete, k_neighbours, distance_function))

    # Save the distances to a CSV file
    pd.DataFrame(pdasc_dists).to_csv(
        f'./dataset_analysis/{dataset}/{dataset}_{k_neighbours}th_nn_{distance_function}_{sample_size}_PDASC.csv',
        index=False)

    return random_dists, pdasc_dists

# Function to get the pairwise distances for a random sample and PDASC index prototypes
def get_pairwise_distances_flue(dataset, distance_function, sample_size, nc, tg, node, n_nodes):


    # Load a random sample of points from the dataset and compute the pairwise distances
    #print(sample_size)
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
        f'./dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{n_nodes}-{node}_{sample_size}_random.csv',
        index=False)

    # Load a random sample of prototypes from PDASC index and compute the pairwise distances
    pdasc_sample = load_PDASC_sample(dataset, sample_size, distance_function, nc, tg, n_nodes, node)
    # pdasc_complete = load_PDASC_sample(dataset, dataset_size, distance_function)
    pdasc_dists = np.sort(compute_distances_pairwise(pdasc_sample, distance_function).flatten())

    # Save the distances to a CSV file
    pd.DataFrame(pdasc_dists).to_csv(
        f'./dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{n_nodes}-{node}_nc{nc}_tg{tg}_{sample_size}_PDASC.csv',
        index=False)

    return random_dists, pdasc_dists

# Function to build a ECDF to be plotted with a fixed number of points
def build_cdf(data, n_points=500):
    """
    Construye una CDF aproximada con un número fijo de puntos.
    data: array de distancias
    n_points: número de puntos de muestreo
    """
    sorted_vals = np.sort(data)
    n = len(sorted_vals)

    # índices equiespaciados
    idx = np.linspace(0, n-1, n_points, dtype=int)
    sampled_vals = sorted_vals[idx]
    cdf = (idx+1) / n

    return sampled_vals, cdf

# Plotting CDFs for k-th nearest neighbour distances in a dataset
def plot_cdfs_nn_dataset_flues(dataset, distance_function, sample, nc, tg, n_nodes):

    k_neighbours = 10
    datasets_size = {
        "wdbc": 1000,
        "municipios": 8031,
        "MNIST": 59999,
        "NYtimes": 290000,
        "GLOVE": 1183514
    }

    print(f"Processing {dataset} dataset with {distance_function} distance function distributed in {n_nodes} nodes")

    # Load the dataset
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"

    if distance_function == "jaccard":
        train_set_csr, test_set_csr = load_hdf5(file_name)
        train_lists = [row.indices.tolist() for row in train_set_csr]
        test_lists = [row.indices.tolist() for row in test_set_csr]
        mlb = MultiLabelBinarizer()
        vector_training = mlb.fit_transform(train_lists).astype(bool)  # dtype=bool para Jaccard
        # vector_testing = mlb.transform(test_lists).astype(bool)  # dtype=bool para Jaccard
    else:
        # train_set, test_set = load_train_test(str(dataset))
        vector_training, vector_testing = load_train_test_h5py(file_name)

    partitions = simulate_flue_partitioning(vector_training, n_nodes, distance_function, nc)

    df_all_nodes = []
    df_points_all = []
    df_vlines_all = []
    df_hlines_all = []


    for node in range(n_nodes):
        partition_size = len(partitions[node])
        sample_size = int(partition_size * (sample / 100))

        print(f"Node {node}: Processing partition of size {partition_size} with sample size {sample_size}")

        # Crear el directorio si no existe
        out_dir = f'./dataset_analysis/{dataset}'
        os.makedirs(out_dir, exist_ok=True)

        # If this path exist,
        PDASC_path = f'./dataset_analysis/{dataset}/{dataset}_{k_neighbours}th_nn_{distance_function}_{n_nodes}-{node}_nc{nc}_tg{tg}_{sample_size}_PDASC.csv'
        random_path = f'./dataset_analysis/{dataset}/{dataset}_{k_neighbours}th_nn_{distance_function}_{n_nodes}-{node}_{sample_size}_random.csv'

        if not (os.path.exists(PDASC_path)) or not (os.path.exists(random_path)):
            # Compute the distances and save them to CSV files
            # print(f"\nComputing distances for {dataset} with {distance_function} distance function and sample size {sample_size}")
            random_dists, pdasc_dists = get_nn_distances(dataset, distance_function, k_neighbours, sample_size,
                                                         partition_size)

        else:
            # Load the distances from the CSV files
            # print(f"\nLoading distances for {dataset} with {distance_function} distance function and sample size {sample_size}")
            random_dists = pd.read_csv(random_path).values.flatten()
            pdasc_dists = pd.read_csv(PDASC_path).values.flatten()


        # CDFs con muestreo (500 puntos)
        rand_x, rand_cdf = build_cdf(random_dists, n_points=500)
        pdasc_x, pdasc_cdf = build_cdf(pdasc_dists, n_points=500)

        # Percentiles sobre PDASC
        percentiles = (1, 15, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99, 100)
        pdasc_percentile_values = np.percentile(pdasc_dists, percentiles)

        # Decimales según métrica
        if distance_function in ['euclidean', 'manhattan']:
            decimales = 2
        elif distance_function == 'chebyshev':
            decimales = 3
        elif distance_function in ['cosine', 'haversine']:
            decimales = 4
        else:
            decimales = 2

        print(
            f"r_{dataset}_{distance_function}_{node}=\""
            + " ".join([f"{v:.{decimales}f}" for v in pdasc_percentile_values])
            + "\""
        )

        node_label = f'Node {node}'
        df_all_nodes.append(pd.DataFrame({
            'distance': rand_x, 'cdf': rand_cdf, 'method': 'Random', 'node': node_label
        }))
        df_all_nodes.append(pd.DataFrame({
            'distance': pdasc_x, 'cdf': pdasc_cdf, 'method': 'PDASC', 'node': node_label
        }))

        df_points = pd.DataFrame({
            'distance': pdasc_percentile_values,
            'cdf': [p / 100 for p in percentiles],
            'node': node_label
        })
        df_points_all.append(df_points)

        df_vlines = df_points.assign(
            xend=lambda df: df['distance'],
            y=0,
            yend=lambda df: df['cdf']
        )
        df_hlines = df_points.assign(
            x=0,
            xend=lambda df: df['distance'],
            yend=lambda df: df['cdf']
        )
        df_vlines_all.append(df_vlines)
        df_hlines_all.append(df_hlines)

    df_all = pd.concat(df_all_nodes, ignore_index=True)
    points_all = pd.concat(df_points_all, ignore_index=True)
    vlines_all = pd.concat(df_vlines_all, ignore_index=True)
    hlines_all = pd.concat(df_hlines_all, ignore_index=True)

    custom_colors = {'Random': '#4C78A8', 'PDASC': '#F58518'}

    p = (
            ggplot(df_all, aes(x='distance', y='cdf', fill='method')) +
            geom_area(alpha=0.3, position='identity') +
            geom_line(aes(color='method'), size=1.1) +
            scale_fill_manual(values=custom_colors) +
            scale_color_manual(values=custom_colors) +
            geom_segment(
                data=vlines_all,
                mapping=aes(x='distance', xend='xend', y='y', yend='yend'),
                linetype='dashed', color='black', inherit_aes=False
            ) +
            geom_segment(
                data=hlines_all,
                mapping=aes(x='x', xend='xend', y='cdf', yend='yend'),
                linetype='dashed', color='black', inherit_aes=False
            ) +
            geom_point(
                data=points_all,
                mapping=aes(x='distance', y='cdf'),
                color='black', size=1.8, inherit_aes=False
            ) +
            labs(
                x=f'{distance_function.capitalize()} Distance',
                y='Probability',
                # title=f"CDFs for {dataset} with {distance_function} ({n_nodes} nodes)"
            ) +
            scale_y_continuous(limits=(0, 1), breaks=np.arange(0, 1.1, 0.1)) +
            scale_x_continuous(breaks=np.linspace(0, max(df_all['distance']), num=5),
                               labels=lambda l: [f"{x:.3f}" for x in l]) +
            facet_wrap('~node') +
            theme_minimal(base_size=13) +
            theme(
                figure_size=(10, 4),
                ##legend_position="bottom",
                legend_position="none",
                axis_text_x=element_text(size=9),
                axis_text_y=element_text(size=9),  # Cambia el tamaño de la letra del eje X
                axis_title_x=element_text(size=11),
                axis_title_y=element_text(size=11),
                aspect_ratio=1
            )
    )

    p = p + theme(
        figure_size=(10, 4),
        ##legend_position="bottom",
        legend_position="none",
        panel_background=element_rect(fill='white', color='black'),  # fondo panel blanco
        plot_background=element_rect(fill='white', color='white'),  # fondo figura blanco
        plot_title=element_text(ha='center')
    )

    out_path = f'./dataset_analysis/{dataset}'
    os.makedirs(out_path, exist_ok=True)
    filename = f'{dataset}_{distance_function}_NN_comparision_overlap_n{n_nodes}_nc{nc}_tg{tg}_{sample_size}.jpg'

    # Guardar en formato rasterizado (PNG real)
    # Muestra por pantalla el gráfico (con show quizas)
    # print(p)
    ggsave(p, os.path.join(out_path, filename), dpi=300, format='png')

def plot_cdfs_pairwise_dataset_flues(dataset, distance_function, sample, nc, tg, n_nodes):

    print(f"Processing {dataset} dataset with {distance_function} distance function distributed in {n_nodes} nodes")

    # Load the dataset
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"

    if distance_function == "jaccard":
        train_set_csr, test_set_csr = load_hdf5(file_name)
        train_lists = [row.indices.tolist() for row in train_set_csr]
        # test_lists = [row.indices.tolist() for row in test_set_csr]
        mlb = MultiLabelBinarizer()
        vector_training = mlb.fit_transform(train_lists).astype(bool)  # dtype=bool para Jaccard
        #vector_testing = mlb.transform(test_lists).astype(bool)  # dtype=bool para Jaccard
    else:
        # train_set, test_set = load_train_test(str(dataset))
        vector_training, vector_testing = load_train_test_h5py(file_name)

    partitions = simulate_flue_partitioning(vector_training, n_nodes, distance_function, nc)

    df_all_nodes = []
    df_points_all = []
    df_vlines_all = []
    df_hlines_all = []


    for node in range(n_nodes):

        # Get the size of the current partition for this node
        partition_size = len(partitions[node])

        # Ensure the sample size is at least 'nc'
        if (nc < partition_size < tg):
            sample_size = nc
        elif (partition_size <= nc):
            sample_size = partition_size
        else:
            # Calculate the sample size as a percentage of the partition size
            sample_size = int(partition_size * (sample / 100))

        print(f"Node {node}: Processing partition of size {partition_size} with sample size {sample_size}")

        # Crear el directorio si no existe
        out_dir = f'./dataset_analysis/{dataset}'
        os.makedirs(out_dir, exist_ok=True)

        # Nombrar los paths para PDASC y random
        PDASC_path=f'./dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{n_nodes}-{node}_nc{nc}_tg{tg}_{sample_size}_PDASC.csv'
        random_path=f'./dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{n_nodes}-{node}_{sample_size}_random.csv'

        print(f'nc: {nc}, tg: {tg}, n_nodes: {n_nodes}, node: {node}')

        if not (os.path.exists(PDASC_path)) or not (os.path.exists(random_path)):
            print(f"\nDistances computations not provided in updated implementation")
            #exit(1)
            ## TO FIX: adaptar extraccion de muestra de arbol de PDASC cuando la relación nc/tg no es 1/2
            random_dists, pdasc_dists = get_pairwise_distances_flue(
                dataset, distance_function, sample_size, nc, tg, node, n_nodes
            )
        else:
            random_dists = pd.read_csv(random_path).values.flatten()
            pdasc_dists = pd.read_csv(PDASC_path).values.flatten()

        # CDFs con muestreo (500 puntos)
        rand_x, rand_cdf = build_cdf(random_dists, n_points=500)
        pdasc_x, pdasc_cdf = build_cdf(pdasc_dists, n_points=500)

        # Percentiles sobre PDASC
        percentiles = (1, 15, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99, 100)
        pdasc_percentile_values = np.percentile(pdasc_dists, percentiles)

        print(distance_function)
        # Decimales según métrica
        if distance_function in ['euclidean', 'manhattan']:
            decimales = 2
        elif distance_function == 'chebyshev':
            decimales = 3
        elif distance_function in ['cosine', 'haversine', 'jaccard']:
            decimales = 4
        else:
            decimales = 2

        print(
            f"r_{dataset}_{distance_function}_{node}=\""
            + " ".join([f"{v:.{decimales}f}" for v in pdasc_percentile_values])
            + "\""
        )


        node_label = f'Node {node}'
        df_all_nodes.append(pd.DataFrame({
            'distance': rand_x, 'cdf': rand_cdf, 'method': 'Random', 'node': node_label
        }))
        df_all_nodes.append(pd.DataFrame({
            'distance': pdasc_x, 'cdf': pdasc_cdf, 'method': 'PDASC', 'node': node_label
        }))

        df_points = pd.DataFrame({
            'distance': pdasc_percentile_values,
            'cdf': [p/100 for p in percentiles],
            'node': node_label
        })
        df_points_all.append(df_points)

        df_vlines = df_points.assign(
            xend=lambda df: df['distance'],
            y=0,
            yend=lambda df: df['cdf']
        )
        df_hlines = df_points.assign(
            x=0,
            xend=lambda df: df['distance'],
            yend=lambda df: df['cdf']
        )
        df_vlines_all.append(df_vlines)
        df_hlines_all.append(df_hlines)

    df_all = pd.concat(df_all_nodes, ignore_index=True)
    points_all = pd.concat(df_points_all, ignore_index=True)
    vlines_all = pd.concat(df_vlines_all, ignore_index=True)
    hlines_all = pd.concat(df_hlines_all, ignore_index=True)

    custom_colors = {'Random': '#4C78A8', 'PDASC': '#F58518'} # Random azul, PDASC naranja

    if n_nodes==1:
        p = (
                ggplot(df_all, aes(x='distance', y='cdf', fill='method')) +
                geom_area(alpha=0.3, position='identity') +
                geom_line(aes(color='method'), size=1.1) +
                scale_fill_manual(values=custom_colors) +
                scale_color_manual(values=custom_colors) +
                # Líneas y puntos de percentiles
                geom_segment(data=vlines_all, mapping=aes(x='distance', xend='xend', y='y', yend='yend'),
                             linetype='dashed', color='black', size=0.4, inherit_aes=False) +
                geom_segment(data=hlines_all, mapping=aes(x='x', xend='xend', y='cdf', yend='yend'),
                             linetype='dashed', color='black', size=0.4, inherit_aes=False) +
                geom_point(data=points_all, mapping=aes(x='distance', y='cdf'),
                           color='black', size=1.8, inherit_aes=False) +
                labs(
                    x=f'{distance_function.capitalize()} Distance',
                    y='Probability'
                    # Título general eliminado
                ) +
                scale_y_continuous(limits=(0, 1), breaks=np.arange(0, 1.1, 0.1)) +
                scale_x_continuous(breaks=np.linspace(0, max(df_all['distance']), num=10)) +
                # SIN facet_wrap para quitar el "Node 0"
                theme_minimal(base_size=13) +
                theme(
                    figure_size=(12, 9),
                    legend_position='none',  # Cambiar a 'bottom' si quieres ver la leyenda
                    panel_background=element_rect(fill='white', color='black'),
                    plot_background=element_rect(fill='white', color='white'),
                    panel_grid_major=element_line(color="#e5e5e5"),
                    panel_grid_minor=element_line(color="#f5f5f5"),
                    legend_title=element_blank(),
                    plot_title=element_text(ha='center'),
                    # Ticks detallados que dan aspecto profesional
                    axis_ticks_major_x=element_line(),
                    axis_ticks_major_y=element_line(),
                    axis_ticks_minor_x=element_line(color='gray', size=0.5),
                    axis_ticks_minor_y=element_line(color='gray', size=0.5),
                    axis_title_x=element_text(size=19, margin={'t': 25}),
                    axis_title_y=element_text(size=19, margin={'r': 25}),
                    axis_text_x=element_text(size=15),
                    axis_text_y=element_text(size=15),  # Cambia el tamaño de la letra del eje X
                    # El strip_text_x ya no se verá si quitas el facet_wrap
                    strip_text_x=element_text(size=11, weight='bold', margin={'t': 10}),
                )
        )

    else:
        p = (
                ggplot(df_all, aes(x='distance', y='cdf', fill='method')) +
                geom_area(alpha=0.3, position='identity') +
                geom_line(aes(color='method'), size=1.1) +
                scale_fill_manual(values=custom_colors) +
                scale_color_manual(values=custom_colors) +
                geom_segment(
                    data=vlines_all,
                    mapping=aes(x='distance', xend='xend', y='y', yend='yend'),
                    linetype='dashed', color='black', inherit_aes=False
                ) +
                geom_segment(
                    data=hlines_all,
                    mapping=aes(x='x', xend='xend', y='cdf', yend='yend'),
                    linetype='dashed', color='black', inherit_aes=False
                ) +
                geom_point(
                    data=points_all,
                    mapping=aes(x='distance', y='cdf'),
                    color='black', size=1.8, inherit_aes=False
                ) +
                labs(
                    x=f'{distance_function.capitalize()} Distance',
                    y='Probability',
                    # title=f"CDFs for {dataset} with {distance_function} ({n_nodes} nodes)"
                ) +
                scale_y_continuous(limits=(0, 1), breaks=np.arange(0, 1.1, 0.1)) +
                scale_x_continuous(breaks=np.linspace(0, max(df_all['distance']), num=5),
                                   labels=lambda l: [f"{x:.0f}" for x in l]) +
                facet_wrap('~node', ncol=5) +
                theme_minimal(base_size=13) +
                theme(
                    figure_size=(12, 9),
                    ##legend_position="bottom",
                    legend_position="none",
                    axis_text_x=element_text(size=10),
                    axis_text_y=element_text(size=10),  # Cambia el tamaño de la letra del eje X
                    axis_title_x=element_text(size=15, margin={'t': 15}),
                    axis_title_y=element_text(size=15, margin={'r': 8}),
                    aspect_ratio=1,
                    panel_spacing_x=0.02
                )
        )

        p = p + theme(
            figure_size=(15, 9),
            ##legend_position="bottom",
            legend_position="none",
            panel_background=element_rect(fill='white', color='black'),  # fondo panel blanco
            plot_background=element_rect(fill='white', color='white'),  # fondo figura blanco
            plot_title=element_text(ha='center')
        )

    # Guardado con el nombre detallado de la primera función
    out_path = f'./dataset_analysis/{dataset}'
    os.makedirs(out_path, exist_ok=True)
    filename = f'{dataset}_{distance_function}_pairwise_comparision_overlap_n{n_nodes}_nc{nc}_tg{tg}_{sample_size}.png'

    ggsave(p, os.path.join(out_path, filename), dpi=300)

    """
        p = (
                ggplot(df_all, aes(x='distance', y='cdf', fill='method')) +
                geom_area(alpha=0.3, position='identity') +
                geom_line(aes(color='method'), size=1.1) +
                scale_fill_manual(values=custom_colors) +
                scale_color_manual(values=custom_colors) +
                geom_segment(
                    data=vlines_all,
                    mapping=aes(x='distance', xend='xend', y='y', yend='yend'),
                    linetype='dashed', color='black', inherit_aes=False
                ) +
                geom_segment(
                    data=hlines_all,
                    mapping=aes(x='x', xend='xend', y='cdf', yend='yend'),
                    linetype='dashed', color='black', inherit_aes=False
                ) +
                geom_point(
                    data=points_all,
                    mapping=aes(x='distance', y='cdf'),
                    color='black', size=1.8, inherit_aes=False
                ) +
                labs(
                    x=f'{distance_function.capitalize()} Distance',
                    y='Probability',
                    #title=f"CDFs for {dataset} with {distance_function} ({n_nodes} nodes)"  # Recuperamos el título
                ) +
                scale_y_continuous(limits=(0, 1), breaks=np.arange(0, 1.1, 0.1)) +
                scale_x_continuous(
                    breaks=np.linspace(0, max(df_all['distance']), num=10)) +  # Más marcas en X como la antigua
                #facet_wrap('~node') +
                theme_minimal(base_size=13) +
                theme(
                    figure_size=(15, 10),  # Tamaño grande de la segunda
                    legend_position="none",  # Sin leyenda
                    panel_background=element_rect(fill='white', color='black'),
                    plot_background=element_rect(fill='white', color='white'),
                    panel_grid_major=element_line(color="#e5e5e5"),  # Rejilla marcada de la segunda
                    panel_grid_minor=element_line(color="#f5f5f5"),
                    legend_title=element_blank(),
                    plot_title=element_text(ha='center', size=16),
                    axis_ticks_major_x=element_line(),
                    axis_ticks_major_y=element_line(),
                    axis_text_x=element_text(size=10),
                    axis_text_y=element_text(size=10),
                    axis_title_x=element_text(margin={'t': 20}),
                    axis_title_y=element_text(margin={'r': 20})
                )
        )
        """

# Plotting PDFs for pairwise distances in a dataset
def plot_pdfs_pairwise_dataset_flues(dataset, distance_function, sample_size, nc, tg, n_nodes):

    datasets_size = {
        "wdbc": 1000,
        "municipios": 8031,
        "MNIST": 59999,
        "NYtimes": 290000,
        "GLOVE": 1183514
    }

    print(f"Processing {dataset} dataset with {distance_function} distance function distributed in {n_nodes} nodes")

    dataset_size = datasets_size[dataset]
    partition_size = int(dataset_size // n_nodes)
    sample_size = partition_size * (sample_size / 100)

    for node in range(n_nodes):
        print(f"Node {node}: Processing partition of size {partition_size} with sample size {sample_size}")
        # Paths for the PDASC and random distances
        PDASC_path = f'./dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{n_nodes}-{node}_nc{nc}_tg{tg}_{sample_size}_PDASC.csv'
        random_path = f'./dataset_analysis/{dataset}/{dataset}_pairwise_{distance_function}_{sample_size}_{n_nodes}_{node}_random.csv'

        # If the paths do not exist, compute the distances
        if not (os.path.exists(PDASC_path)) or not (os.path.exists(random_path)):
            # Compute the distances and save them to CSV files
            random_dists, pdasc_dists = get_pairwise_distances_flue(dataset, distance_function, sample_size, nc, tg, node, n_nodes)
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
    out_path = f'./dataset_analysis/{dataset}'
    os.makedirs(out_path, exist_ok=True)
    filename = f'{dataset}_{distance_function}_pairwise_pdf_comparision_{sample_size}_nc{nc}_tg{tg}_n{n_nodes}_paper.jpg'
    p.save(os.path.join(out_path, filename), dpi=300)

# Plotting CDFs of pairwise distances for different sample sizes in a dataset
def plot_cdfs_nn_samplesize_dataset(dataset, distance_function):

    k_neighbours = 10
    datasets_size = {
        "wdbc": 1000,
        "municipios": 8031,
        "MNIST": 59999,
        "NYtimes": 290000,
        "GLOVE": 1200000,  # 1183514,
        "MovieLens": 690000
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
    out_path = f'./dataset_analysis/{dataset}'
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
    out_path = f'./dataset_analysis/{dataset}'
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

    parser.add_argument("-nc", help="Indicate the number of centroids of the PDASC index to be used.", type=int)

    parser.add_argument("-tg", help="Indicate the group size of the PDASC index to be used.", type=int)

    parser.add_argument("-nodes", help="Indicate the nodes of the PDASC index to be used.", type=int)


    args = parser.parse_args()


    if args.cdfsNN:
        if len(args.cdfsNN) == 1:
            print(f"Received a single tuple: {args.cdfsNN}")
            dataset=args.cdfsNN[0][0]
            distance_function = args.cdfsNN[0][1]
            nc = args.nc
            tg = args.tg
            sample_size = args.size
            n_nodes = args.nodes
            plot_cdfs_nn_dataset_flues(dataset, distance_function, sample_size, nc, tg, n_nodes)

    elif args.cdfsNN_SS:
        if len(args.cdfsNN_SS) == 1:
            print(f"Received a single tuple: {args.cdfsNN_SS}")
            dataset=args.cdfsNN_SS[0][0]
            distance_function = args.cdfsNN_SS[0][1]
            nc = args.nc
            tg = args.tg
            n_nodes = args.nodes
            plot_cdfs_nn_samplesize_dataset(dataset,distance_function)

        else:
            print("Error: The argument for --cdfNN_SS must be a tuple")
            exit(1)

    elif args.cdfsPW:
        if len(args.cdfsPW) == 1:
            print(f"Received a single tuple: {args.cdfsPW}")
            dataset=args.cdfsPW[0][0]
            distance_function = args.cdfsPW[0][1]
            nc = args.nc
            tg = args.tg
            sample_size = args.size
            n_nodes = args.nodes
            plot_cdfs_pairwise_dataset_flues(dataset, distance_function, sample_size, nc, tg, n_nodes)

    elif args.cdfsPW_SS:
        if len(args.cdfsPW_SS) == 1:
            print(f"Received a single tuple: {args.cdfsPW_SS}")
            dataset=args.cdfsPW_SS[0][0]
            distance_function = args.cdfsPW_SS[0][1]
            nc = args.nc
            tg = args.tg
            n_nodes = args.nodes
            plot_cdfs_pairwise_samplesize_dataset(dataset,distance_function)

        else:
            print("Error: The argument for --cdfPW_SS must be a tuple")
            exit(1)

    elif args.pdfsPW:
        if len(args.pdfsPW) == 1:
            print(f"Received a single tuple: {args.pdfsPW}")
            dataset=args.pdfsPW[0][0]
            distance_function = args.pdfsPW[0][1]
            nc = args.nc
            tg = args.tg
            sample_size = args.size
            n_nodes = args.nodes
            plot_pdfs_pairwise_dataset_flues(dataset, distance_function, sample_size, nc, tg, n_nodes)

        else:
            print("Error: The argument for --pdfsPW must be a tuple")
            exit(1)

    exit(0)
