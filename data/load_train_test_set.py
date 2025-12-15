import os
import h5py
import numpy as np
import data.synthetic_data.generate_gaussian_clouds as dt
from sklearn import preprocessing
import pandas as pd
import logging
import re
from pathlib import Path
import random
from scipy.sparse import csr_matrix
np.set_printoptions(suppress=True)

# Set constants for dataset generation
nclouds = 8
# npc = 100000
# overlap = True
normaliza = False

####### Load and store train and test set from a h5py file #########

# Store train and test set into a hdf5 file
def save_train_test_h5py(train_set, test_set, file_name):

    # Store the 2 different sets on a hdf5 file
    with h5py.File(file_name, 'w') as f:
        dset1 = f.create_dataset('train_set', data=train_set)
        dset1 = f.create_dataset('test_set', data=test_set)


# Load train and test set from a hdf5 file
def load_train_test_h5py(file_name):

    # Load train and test set from the choosen file
    # print (file_name)
    if not os.path.exists(file_name):
        print("File " + file_name + " does not exist")
        return None, None
    else:
        with h5py.File(file_name, 'r') as hdf5_file:
            # print("\n ######### Loading train and test set from " + file_name + " #########")
            logging.info("Loading train and test set from " + file_name + "\n")
            print("Loading train and test set from " + file_name + "\n")
            return np.array(hdf5_file['train_set']), np.array(hdf5_file['test_set'])



#### Load and store train and test set from a hdf5 file as CSR binary for Jaccard distance datasets (MovieLens & Kosarak) #########
def to_sorted_unique_int_list(lst):
    """Devuelve la lista ordenada y sin duplicados (valores convertidos a int)."""
    return sorted(set(int(x) for x in lst))


def split_train_test(X, test_size=100, seed=0):
    n = len(X)
    if test_size > n:
        raise ValueError(f"test_size ({test_size}) no puede ser mayor que n ({n})")
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    test_idx = set(idxs[:test_size])
    train = [X[i] for i in range(n) if i not in test_idx]
    test = [X[i] for i in range(n) if i in test_idx]
    return train, test


def infer_dimension(list_of_lists):
    """Calcula dimension = max(indice)+1 en todo el dataset; si vacío, 0."""
    m = -1
    for lst in list_of_lists:
        for x in lst:
            xi = int(x)
            if xi > m:
                m = xi
    return (m + 1) if m >= 0 else 0


def lists_to_csr(X, dimension=None, dtype=np.uint8, dedup=True):
    """
    Convierte lista de listas de índices en una CSR binaria.
    - X: list[list[int]]
    - dimension: nº de columnas (si None se infiere)
    - dedup: elimina duplicados por fila (recomendado para Jaccard)
    """
    n_rows = len(X)
    if dimension is None:
        dimension = infer_dimension(X)

    indptr = [0]
    indices = []
    data = []

    for row in X:
        if dedup:
            row = set(row)  # elimina duplicados
        # opcionalmente, filtra posibles índices fuera de rango
        row = [c for c in row if 0 <= c < dimension]
        # orden estable/consistente (no obligatorio)
        row = sorted(row)
        indices.extend(row)
        data.extend([1] * len(row))
        indptr.append(len(indices))

    # Construye CSR
    X_csr = csr_matrix(
        (np.asarray(data, dtype=dtype),
         np.asarray(indices, dtype=np.int32),
         np.asarray(indptr, dtype=np.int32)),
        shape=(n_rows, dimension),
        dtype=dtype
    )
    return X_csr


def _write_csr(group: h5py.Group, M: csr_matrix, compress=True):
    """Guarda una CSR en un subgrupo HDF5."""
    if compress:
        compression = "gzip"
        compression_opts = 4
    else:
        compression = None
        compression_opts = None

    group.create_dataset("data", data=M.data, compression=compression, compression_opts=compression_opts)
    group.create_dataset("indices", data=M.indices, compression=compression, compression_opts=compression_opts)
    group.create_dataset("indptr", data=M.indptr, compression=compression, compression_opts=compression_opts)
    group.attrs["shape"] = M.shape


def save_hdf5(path: Path, train_set, test_set, dimension):
    """
    Guarda train/test en HDF5 como CSR binaria.
    - train_set / test_set: list[list[int]] o CSR directamente
    - dimension: nº columnas global
    - meta: dict con metadatos (se guarda como JSON en attrs)
    """
    # Asegura CSR
    train_csr = train_set if isinstance(train_set, csr_matrix) else lists_to_csr(train_set, dimension)
    test_csr = test_set if isinstance(test_set, csr_matrix) else lists_to_csr(test_set, dimension)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        gtrain = f.create_group("train_set")
        gtest = f.create_group("test_set")

        _write_csr(gtrain, train_csr)
        _write_csr(gtest, test_csr)


def _read_csr(group: h5py.Group, dtype=np.uint8) -> csr_matrix:
    """Lee una CSR desde un subgrupo HDF5."""
    data = group["data"][()]
    indices = group["indices"][()]
    indptr = group["indptr"][()]
    shape = tuple(group.attrs["shape"])
    return csr_matrix((data, indices, indptr), shape=shape, dtype=dtype)


def load_hdf5(path: Path):
    """
    Carga train/test CSR y metadatos desde HDF5.
    Devuelve: (train_csr, test_csr, dimension, meta_dict)
    """
    with h5py.File(path, "r") as f:
        train_csr = _read_csr(f["train_set"])
        test_csr = _read_csr(f["test_set"])

    return train_csr, test_csr

####### Generate brand new train and test set #########

# If test_eq_train=True, that means test set is going to be set the same as train one,
# so a punctual search of any element contained on train dataset is going to be carry out
# At first, it would only be relevant over little size gaussian sets to carry on some tests

def load_train_test(dataset_name, test_eq_train=False):

    # print("\n ######### Creating train and test set from " + dataset_name + " dataset #########")
    logging.info("Creating train and test as " + dataset_name + " dataset\n")

    # Setting size of the test set
    test_set_size = 100

    # Generate Gaussian Clouds dataset and generate train and test sets
    if dataset_name.startswith("synthetic_data"):

        # From dataset name, obtain information about clouds features
        npc = int(re.sub("npc", "", re.split('_', dataset_name)[2]))
        overlap = bool(re.split('_', dataset_name)[3])

        # Generate n gaussian clouds and store them into a NumpyArray
        gaussian_clouds, coordx, coordy, puntos_nube = dt.generate_data_gaussian_clouds(nclouds, npc, overlap)
        gaussian_clouds = np.array(gaussian_clouds)

        # If normaliza, normalize the dataset
        if normaliza:
            gaussian_clouds = preprocessing.normalize(gaussian_clouds, axis=0, norm='l2')

        if test_eq_train:
            train_set = gaussian_clouds
            test_set = gaussian_clouds

        else:
            # For this experiment, compose the test_set (100 elements not contained on the train set) and the train_set
            np.random.seed(1234)
            index_testing = np.random.choice(len(gaussian_clouds), test_set_size, replace=False)
            test_set = gaussian_clouds[index_testing]
            index_complete = np.linspace(0, len(gaussian_clouds) - 1, len(gaussian_clouds), dtype=int)
            index_training = np.setdiff1d(index_complete, index_testing)
            train_set = gaussian_clouds[index_training]

        save_train_test_h5py(train_set, test_set, "./data/" + dataset_name + "_train_test_set.hdf5")

        return train_set, test_set

    # Load Geographical Dataset (Municipios) dataset and generate train and test sets
    elif dataset_name == "municipios":
        # Read the complete dataset from a csv file and store it into a NumpyArray
        datos = pd.read_csv('./data/raw_data/MUNICIPIOS-utf8.csv', sep=';')
        municipios = pd.DataFrame(datos, columns=['LONGITUD_ETRS89', 'LATITUD_ETRS89'])

        # Convertir a float (reemplazando comas por puntos decimales)
        municipios['LONGITUD_ETRS89'] = municipios['LONGITUD_ETRS89'].str.replace(',', '.').astype(float)
        municipios['LATITUD_ETRS89'] = municipios['LATITUD_ETRS89'].str.replace(',', '.').astype(float)

        # Convertir a numpy array
        municipios = municipios.to_numpy()

        # Mezclar aleatoriamente el dataset completo (shuffle)
        np.random.seed(1234)
        np.random.shuffle(municipios)

        # Normalizar si corresponde
        if normaliza:
            municipios = preprocessing.normalize(municipios, axis=0, norm='l2')

        # Separar en train y test
        index_testing = np.random.choice(len(municipios), test_set_size, replace=False)
        test_set = municipios[index_testing]

        index_complete = np.arange(len(municipios))
        index_training = np.setdiff1d(index_complete, index_testing)
        train_set = municipios[index_training]

        # Save train_set on a txt
        # a_file = open("test.txt", "w")
        # for row in train_set:
        #    a_file.write((str(row[0]) + " " + str(row[1]) + "\n"))
        # a_file.close()

        save_train_test_h5py(train_set, test_set, "./data/municipios_train_test_set.hdf5")

        return train_set, test_set

    # Load Images Dataset (MNIST) dataset and generate train and test sets
    elif dataset_name == "MNIST":

        # Read the train_set from a csv file and store it into a Numpy Array
        data = pd.read_csv('./data/raw_data/mnist_train.csv', delimiter=',', nrows=None)
        train_set = pd.DataFrame(data).to_numpy().astype(float)

        # Read the test_set from a csv file and store it into a Numpy Array
        data = pd.read_csv('./data/raw_data/mnist_test.csv', delimiter=',', nrows=None)
        test_set = pd.DataFrame(data).drop(columns='label').to_numpy().astype(float)

        # For this experiment,compose the test_set (100 elements not contained on the train set) and the train_set
        # (We guarantee this because original csv training and test sets are separated from each other)
        np.random.seed(1234)
        index_testing = np.random.choice(len(test_set), test_set_size, replace=False)
        test_set = test_set[index_testing]
        # n_test_set = test_set[0:10, :]


        # If normaliza, normalize the datasets
        if normaliza:
            train_set = preprocessing.normalize(train_set, axis=0, norm='l2')
            test_set = preprocessing.normalize(test_set, axis=0, norm='l2')

        save_train_test_h5py(train_set, test_set, "./data/MNIST_train_test_set.hdf5")

        return train_set, test_set

    # Load pre-trained 100-dimensional vector representations for words (GLOVE) dataset and generate train and test sets
    # English word vectors pre-trained on the combined Wikipedia 2014 +  Gigaword 5th Edition corpora (6B tokens, 400K vocab)
    elif dataset_name == "GLOVE":

        # Read the train_set and test_set from a hdf5 file and store it into Numpy Arrays
        with h5py.File('./data/raw_data/glove-100-angular.hdf5', 'r') as hdf5_file:
            # print("Keys: %s" % hdf5_file.keys())
            train_set = np.array(hdf5_file['train'])
            test_set = np.array(hdf5_file['test'])

        # For this experiment,compose the test_set (100 elements not contained on the train set) and the train_set
        # (We guarantee this because original hdf5 training and test sets are separated from each other)
        np.random.seed(1234)
        index_testing = np.random.choice(len(test_set), test_set_size, replace=False)
        test_set = test_set[index_testing]
        # n_test_set = train_set[1000:1099]

        # If normaliza, normalize the datasets
        if normaliza:
            train_set = preprocessing.normalize(train_set, axis=0, norm='l2')
            test_set = preprocessing.normalize(test_set, axis=0, norm='l2')

        save_train_test_h5py(train_set, test_set, "./data/GLOVE_train_test_set.hdf5")

        return train_set, test_set


    # Load pre-trained 100-dimensional vector representations for words (GLOVE) dataset
    # Compose a reduced train set of 100000 items
    elif dataset_name == "GLOVE100000":

        # Read the train_set and test_set from a hdf5 file and store it into Numpy Arrays
        with h5py.File('./data/raw_data/glove-100-angular.hdf5', 'r') as hdf5_file:
            # print("Keys: %s" % hdf5_file.keys())
            train_set = np.array(hdf5_file['train'])
            test_set = np.array(hdf5_file['test'])

        # For this experiment, compose a reduced train_set of 100000 elements
        np.random.seed(1234)
        index_training = np.random.choice(len(train_set), 100000, replace=False)
        train_set = train_set[index_training]
        # # n_train_set = train_set[0:999]

        # For this experiment,compose the test_set (100 elements not contained on the train set) and the train_set
        # (We guarantee this because original hdf5 training and test sets are separated from each other)
        np.random.seed(1234)
        index_testing = np.random.choice(len(test_set), test_set_size, replace=False)
        test_set = test_set[index_testing]
        # n_test_set = train_set[1000:1099]

        # If normaliza, normalize the datasets
        if normaliza:
            train_set = preprocessing.normalize(train_set, axis=0, norm='l2')
            test_set = preprocessing.normalize(test_set, axis=0, norm='l2')

        save_train_test_h5py(train_set, test_set, "./data/GLOVE100000_train_test_set.hdf5")

        return train_set, test_set

    elif dataset_name == ("NYtaxis"):
        datos = pd.read_parquet('./data/raw_data/NYtaxis.parquet', engine='pyarrow')
        NYtaxis = pd.DataFrame(datos, columns=['PULocationID', 'DOLocationID'])

        # Drop duplicates
        NYtaxis = NYtaxis.drop_duplicates()

        # Convert into Numpy Array of int32 (lighter to process)
        NYtaxis = NYtaxis.to_numpy()
        NYtaxis=NYtaxis.astype(np.int32)

        # If normaliza, normalize the dataset
        if normaliza:
            NYtaxis = preprocessing.normalize(NYtaxis, axis=0, norm='l2')

        # For this experiment, compose the test_set (100 elements not contained on the train set) and the train_set
        np.random.seed(1234)
        index_testing = np.random.choice(len(NYtaxis), test_set_size, replace=False)
        test_set = NYtaxis[index_testing]
        index_complete = np.linspace(0, len(NYtaxis) - 1, len(NYtaxis), dtype=int)
        index_training = np.setdiff1d(index_complete, index_testing) # The index training are the elements of the complete dataset wich are not on the test set
        train_set = NYtaxis[index_training]

        #Uncomment if we don't want to process the complete dataset, only a 1000000 sample
        #index_training100000 = np.random.choice(len(index_training), 100000, replace=False)
        #train_set = NYtaxis[index_training100000]


        save_train_test_h5py(train_set, test_set, "./data/NYtaxis_train_test_set.hdf5")

        return train_set, test_set


    elif dataset_name == ("wdbc"):

        datos = pd.DataFrame(pd.read_csv('./data/raw_data/wdbc.data', sep=","))

        # Drop first two columns (index and diagnoses)
        datos = datos.drop(datos.columns[[0,1]], axis=1)

        # Convert into Numpy Array
        wdbc = datos.to_numpy()

        # If normaliza, normalize the dataset
        if normaliza:
            #wdbc = preprocessing.normalize(wdbc, axis=0, norm='l2')
            scaler = preprocessing.MinMaxScaler()
            wdbc = scaler.fit_transform(wdbc)


        # For this experiment,compose the test_set (100 elements not contained on the train set) and the train_set
        test_set_size = 100
        np.random.seed(1234)
        index_testing = np.random.choice(len(wdbc), test_set_size, replace=False)
        test_set = wdbc[index_testing]
        index_complete = np.linspace(0, len(wdbc) - 1, len(wdbc), dtype=int)
        index_training = np.setdiff1d(index_complete, index_testing) # The index training are the elements of the complete dataset wich are not on the test set
        train_set = wdbc[index_training]

        # Save train_set on a txt
        # a_file = open("test.txt", "w")
        # for row in train_set:
        #    a_file.write((str(row[0]) + " " + str(row[1]) + "\n"))
        # a_file.close()

        save_train_test_h5py(train_set, test_set, "./data/wdbc_train_test_set.hdf5")

        return train_set, test_set

    elif dataset_name == ("wdbc"):

        datos = pd.DataFrame(pd.read_csv('./data/raw_data/wdbc.data', sep=","))

        # Drop first two columns (index and diagnoses)
        datos = datos.drop(datos.columns[[0, 1]], axis=1)

        # Convert into Numpy Array
        wdbc = datos.to_numpy()

        # If normaliza, normalize the dataset
        if normaliza:
            # wdbc = preprocessing.normalize(wdbc, axis=0, norm='l2')
            scaler = preprocessing.MinMaxScaler()
            wdbc = scaler.fit_transform(wdbc)

        # For this experiment,compose the test_set (100 elements not contained on the train set) and the train_set
        test_set_size = 100
        np.random.seed(1234)
        index_testing = np.random.choice(len(wdbc), test_set_size, replace=False)
        test_set = wdbc[index_testing]
        index_complete = np.linspace(0, len(wdbc) - 1, len(wdbc), dtype=int)
        index_training = np.setdiff1d(index_complete,
                                      index_testing)  # The index training are the elements of the complete dataset wich are not on the test set
        train_set = wdbc[index_training]

        # Save train_set on a txt
        # a_file = open("test.txt", "w")
        # for row in train_set:
        #    a_file.write((str(row[0]) + " " + str(row[1]) + "\n"))
        # a_file.close()

        save_train_test_h5py(train_set, test_set, "./data/wdbc_train_test_set.hdf5")

        return train_set, test_set

    elif dataset_name == "NYtimes":

        # Read the train_set and test_set from a hdf5 file and store it into Numpy Arrays
        with h5py.File('./data/raw_data/nytimes-256-angular.hdf5', 'r') as hdf5_file:
            # print("Keys: %s" % hdf5_file.keys())
            train_set = np.array(hdf5_file['train'])
            test_set = np.array(hdf5_file['test'])

        # For this experiment,compose the test_set (100 elements not contained on the train set) and the train_set
        # (We guarantee this because original hdf5 training and test sets are separated from each other)
        np.random.seed(1234)
        index_testing = np.random.choice(len(test_set), test_set_size, replace=False)
        test_set = test_set[index_testing]

        # If normaliza, normalize the datasets
        if normaliza:
            train_set = preprocessing.normalize(train_set, axis=0, norm='l2')
            test_set = preprocessing.normalize(test_set, axis=0, norm='l2')

        save_train_test_h5py(train_set, test_set, "./data/NYtimes_train_test_set.hdf5")

        return train_set, test_set

    elif dataset_name == "kosarak":

        #local_fn = "kosarak.dat.gz"
        # only consider sets with at least min_elements many elements
        min_elements = 20
        #url = "http://fimi.uantwerpen.be/data/%s" % local_fn
        #download(url, local_fn)

        X = []
        dimension = 0
        with open("./data/raw_data/kosarak.dat", "r") as f:
            content = f.readlines()
            # preprocess data to find sets with more than 20 elements
            # keep track of used ids for reenumeration
            for line in content:
                if len(line.split()) >= min_elements:
                    X.append(list(map(int, line.split())))
                    dimension = max(dimension, max(X[-1]) + 1)

        # 1) split
        train_set, test_set = split_train_test(X, test_size=100, seed=1234)
        print(f"train: {len(train_set)}   test: {len(test_set)}")

        # dimensión global (máx índice + 1)
        dimension = infer_dimension(X)
        print("dimension:", dimension)

        save_hdf5(
            Path("./data/kosarak_train_test_set.hdf5"),
            train_set,
            test_set,
            dimension
        )

        # save_train_test_h5py(train_set, test_set, "./data/kosarak_train_test_set.hdf5")

        return train_set, test_set
    elif dataset_name == "MovieLens":

        fn = "./data/raw_data/ml-10m.zip"
        #url = "http://files.grouplens.org/datasets/MovieLens/%s" % fn

        # asumo que tienes disponible download(url, fn) como en tu snippet
        #download(url, fn)

        ratings_file = "./data/raw_data/ml-10M100K/ratings.dat"
        separator = "::"
        min_rating = 3.0

        users = {}  # map userId -> row index
        X = []  # lista de listas: por usuario, ids de items
        dimension = 0  # nº total de items = max(itemId)+1

        with open(ratings_file, "r") as file:
            for line in file:
                el = line.strip().split(separator)
                if len(el) < 3:
                    continue
                userId = el[0]
                itemId = int(el[1])
                rating = float(el[2])

                if rating < min_rating:
                    continue

                if userId not in users:
                    users[userId] = len(users)
                    X.append([])

                X[users[userId]].append(itemId)
                if itemId + 1 > dimension:
                    dimension = itemId + 1

        # 1) split
        train_set, test_set = split_train_test(X, test_size=100, seed=1234)
        print(f"train: {len(train_set)}   test: {len(test_set)}")

        # dimensión global (máx índice + 1)
        dimension = infer_dimension(X)
        print("dimension:", dimension)

        save_hdf5(
            Path("./data/MovieLens_train_test_set.hdf5"),
            train_set,
            test_set,
            dimension
        )

        # save_train_test_h5py(train_set, test_set, "./data/MovieLens_train_test_set.hdf5")
        return train_set, test_set



    elif dataset_name == "LastFM":

        # Read the train_set and test_set from a hdf5 file and store it into Numpy Arrays
        with h5py.File('./data/raw_data/lastfm-64-dot.hdf5', 'r') as hdf5_file:
            #print("Keys: %s" % hdf5_file.keys())
            train_set = np.array(hdf5_file['train'])
            test_set = np.array(hdf5_file['test'])

        # For this experiment,compose the test_set (100 elements not contained on the train set) and the train_set
        # (We guarantee this because original hdf5 training and test sets are separated from each other)
        np.random.seed(1234)
        index_testing = np.random.choice(len(test_set), test_set_size, replace=False)
        test_set = test_set[index_testing]

        # If normaliza, normalize the datasets
        if normaliza:
            train_set = preprocessing.normalize(train_set, axis=0, norm='l2')
            test_set = preprocessing.normalize(test_set, axis=0, norm='l2')

        save_train_test_h5py(train_set, test_set, "./data/LastFM_train_test_set.hdf5")

        return train_set, test_set

    elif dataset_name == "LastFM100000":

        # Read the train_set and test_set from a hdf5 file and store it into Numpy Arrays
        with h5py.File('./data/lastfm-64-dot.hdf5', 'r') as hdf5_file:
            # print("Keys: %s" % hdf5_file.keys())
            train_set = np.array(hdf5_file['train'])
            test_set = np.array(hdf5_file['test'])

        # For this experiment,compose the test_set (100 elements not contained on the train set) and the train_set
        # (We guarantee this because original hdf5 training and test sets are separated from each other)
        np.random.seed(1234)
        index_training = np.random.choice(len(train_set), 100000, replace=False)
        train_set = train_set[index_training]
        # # n_train_set = train_set[0:999]

        # For this experiment, compose a reduced test_set of 100 elements
        np.random.seed(1234)
        index_testing = np.random.choice(len(test_set), test_set_size, replace=False)
        test_set = test_set[index_testing]
        # n_test_set = train_set[1000:1099]

        # If normaliza, normalize the datasets
        if normaliza:
            train_set = preprocessing.normalize(train_set, axis=0, norm='l2')
            test_set = preprocessing.normalize(test_set, axis=0, norm='l2')

        save_train_test_h5py(train_set, test_set, "./data/LastFM100000_train_test_set.hdf5")

        return train_set, test_set

    else:

        print("Dataset not found")
        logging.info("Dataset not found\n")
        return None, None


#load_train_test('kosarak')