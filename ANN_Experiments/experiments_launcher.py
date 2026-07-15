import pandas

from ANN_Experiments.algorithms.Exact.knn import Exact
from ANN_Experiments.algorithms.Pynndescent.knn import PYNN
from ANN_Experiments.algorithms.PDASC.knn import PDASC
from ANN_Experiments.algorithms.FAISS_IVF.knn import FAISS_IVF
from ANN_Experiments.algorithms.FAISS_LSH.knn import FAISS_LSH
from ANN_Experiments.algorithms.FAISS_HNSW.knn import FAISS_HNSW
from ANN_Experiments.algorithms.NMSLIB_HNSW.knn import NMSLIB_HNSW
from ANN_Experiments.algorithms.Annoy.knn import Annoy

import multiprocessing
import argparse
import os
import sys
import platform
import datetime
import logging
import psutil  # Install with `pip install psutil`
from cpuinfo import get_cpu_info  # Install with `pip install py-cpuinfo`


# Parameters common to all ANN methods
COMMON_PARAMETERS = [
    "k",
    "dataset",
    "method",
    "distance",
]

# Method-specific parameters
METHOD_PARAMETERS = {

    "Exact": [
        "algorithm"
    ],

    "PDASC": [
        "n_nodes",
        "np",
        "gl",
        "r",
        "p_ECDF",
        "algorithm",
        "implementation"
    ],

    "FLANN": [
        "ncentroids",
        "algorithm"
    ],

    "PYNN": [
        "n_neighbors",
        "diversify_prob",
        "pruning_degree_multiplier",
        "epsilon"
    ],

    "IVF": [
        "nlist",
        "nprobe"
    ],

    "LSH": [
        "nbits"
    ],

    "FAISSHNSW": [
        "M",
        "efConstruction",
        "efSearch"
    ],

    "NMSLIBHNSW": [
        "M",
        "efConstruction",
        "efSearch",
        "post",
        "coords_in_degrees"
    ],

    "ANNOY": [
        "n_trees",
        "k_search"
    ]
}

def experiment(exp):

    # Get the ANN method to be evaluated
    method = exp["method"]

    if method not in METHOD_PARAMETERS:
        raise ValueError(f"Method '{method}' not supported.")

    # Build a dictionary containing only the parameters
    # required by the selected ANN method.
    experiment_parameters = {
        parameter: exp[parameter]
        for parameter in COMMON_PARAMETERS + METHOD_PARAMETERS[method]
    }

    if method == "Exact":
        Exact(experiment_parameters)

    elif method == "PDASC":
        PDASC(experiment_parameters)

    elif method == "PYNN":
        PYNN(experiment_parameters)

    elif method == "IVF":
        FAISS_IVF(experiment_parameters)

    elif method == "LSH":
        FAISS_LSH(experiment_parameters)

    elif method == "FAISSHNSW":
        FAISS_HNSW(experiment_parameters)

    elif method == "NMSLIBHNSW":
        NMSLIB_HNSW(experiment_parameters)

    elif method == "ANNOY":
        Annoy(experiment_parameters)


def read_experiment_file(csv_file):
    """
    Read the experiments described in a CSV file.

    Parameters
    ----------
    csv_file : str
        CSV file name (e.g. test_kNN_MNIST_PDASC.csv).

    Returns
    -------
    list(dict)
        A list of dictionaries, where each dictionary represents one experiment.
    """

    # Extract dataset and method from filename
    filename = os.path.splitext(csv_file)[0]
    parts = filename.split("_")

    dataset_name = parts[2]
    method_name = parts[3]

    # Build CSV path
    csv_path = "./ANN_Experiments/config/" + dataset_name + "/" + csv_file

    # Verify that the CSV file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"[ERROR] Experiment file '{csv_path}' doesn't exist."
        )

    print(f"--- Reading {csv_file} ---")

    # Read CSV
    #experiments = pandas.read_csv(csv_path)
    #experiments = pandas.read_csv(csv_path, sep=";")
    experiments = pandas.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
    #print(experiments.columns.tolist())

    # Remove completely empty rows
    experiments.dropna(how="all", inplace=True)

    # Replace NaN with None
    experiments = experiments.where(pandas.notnull(experiments), None)

    # Check that the method is supported
    if method_name not in METHOD_PARAMETERS:
        raise ValueError(f"Method '{method_name}' is not supported.")

    # Parameters required by the selected ANN method
    required_parameters = COMMON_PARAMETERS + METHOD_PARAMETERS[method_name]

    # Check that all required parameters are present and contain values
    for parameter in required_parameters:

        if parameter not in experiments.columns:
            raise ValueError(
                f"Missing parameter '{parameter}' in '{csv_file}'."
            )

        if experiments[parameter].isnull().any():
            raise ValueError(
                f"Parameter '{parameter}' contains empty values in '{csv_file}'."
            )

    # Convert the DataFrame into a list of dictionaries,
    # where each dictionary represents one experiment.
    return experiments.to_dict(orient="records")


# Function to execute the experiments described in the configuration files provided
def execute_experiments(argument, log, optional_filters=None):

    # Check if the argument is a CSV file
    if argument.endswith(".csv"):
        csv_file = argument
    else:
        raise ValueError("Argument must be a CSV file (e.g., test_kNN_MNIST_PDASC.csv).")

    # Extract dataset and method from filename
    filename = os.path.splitext(csv_file)[0]
    parts = filename.split("_")

    dataset_name = parts[2]
    method_name = parts[3]

    # Read experiments
    experiments = read_experiment_file(csv_file)


    for filt in optional_filters:

        if "=" in filt:
            parameter, value = filt.split("=", 1)

            experiments = [
                exp for exp in experiments
                if str(exp[parameter]) == value
            ]

    # Get current date and time
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%d-%m-%Y_%H:%M")

    # Configure logging
    if log:
        logging.basicConfig(
            filename=f"./ANN_Experiments/logs/{dataset_name}/test_knn_{dataset_name}_{formatted_time}.log",
            filemode='w',
            format='%(asctime)s - %(name)s - %(message)s',
            level=logging.INFO
        )
    else:
        logging.disable(logging.CRITICAL)

    logging.info('------------------------------------------------------------------------')
    logging.info('          Experiments launcher for %s - %s', method_name, dataset_name)
    logging.info('------------------------------------------------------------------------\n')

    # Log system information
    logging.info("Platform: %s\n", platform.platform())

    cpu_info = get_cpu_info()
    logging.info("Processor: %s", cpu_info.get("brand_raw", "Unknown"))
    logging.info("Processor Architecture: %s", cpu_info.get("arch", "Unknown"))
    logging.info("Processor Cores: %d", psutil.cpu_count(logical=False))
    logging.info("Logical Processors: %d\n", psutil.cpu_count(logical=True))
    logging.info("Python Version: %s\n", sys.version)

    virtual_memory = psutil.virtual_memory()
    logging.info("Total RAM: %.2f GB", virtual_memory.total / (1024 ** 3))
    logging.info("Available RAM: %.2f GB", virtual_memory.available / (1024 ** 3))
    logging.info("Used RAM: %.2f GB", virtual_memory.used / (1024 ** 3))
    logging.info("RAM Usage Percentage: %.2f%%\n", virtual_memory.percent)
    logging.info('------------------------------------------------------------------------\n')

    # Execute all experiments
    for exp in experiments:
        experiment(exp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("argument", help="Name of the dataset or a single .ini file", type=str)
    parser.add_argument('--log', action='store_true', help="Activa el registro en log")
    parser.add_argument('--filter', nargs='*', default=[], help="Optional filters for experiments (e.g., 'tg=10 nc=5')")

    args = parser.parse_args()

    multiprocessing.set_start_method("fork")  # Solo si da problemas sin él (UNIX systems)

    execute_experiments(args.argument, args.log, args.filter)

    #exit(0)