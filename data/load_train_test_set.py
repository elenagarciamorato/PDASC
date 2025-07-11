import os
import h5py
import numpy as np
import data.synthetic_data.generate_gaussian_clouds as dt
from sklearn import preprocessing
import pandas as pd
import logging
import re
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
        # index = datos_geo.index
        # cant_ptos = len(index)
        municipios['LONGITUD_ETRS89'] = municipios['LONGITUD_ETRS89'].str.replace(',', '.').astype(float)
        municipios['LATITUD_ETRS89'] = municipios['LATITUD_ETRS89'].str.replace(',', '.').astype(float)
        municipios = municipios.to_numpy()

        # If normaliza, normalize the dataset
        if normaliza:
            municipios = preprocessing.normalize(municipios, axis=0, norm='l2')

        # For this experiment,compose the test_set (100 elements not contained on the train set) and the train_set
        np.random.seed(1234)
        index_testing = np.random.choice(len(municipios), test_set_size, replace=False)
        test_set = municipios[index_testing]
        index_complete = np.linspace(0, len(municipios) - 1, len(municipios), dtype=int)
        index_training = np.setdiff1d(index_complete, index_testing) #The index training are the elements of the complete dataset wich are not on the test set
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


#load_train_test('municipios')