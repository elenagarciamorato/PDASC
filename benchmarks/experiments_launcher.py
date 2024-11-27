from benchmarks.algorithms.Exact.knn import Exact
from benchmarks.algorithms.FLANN.knn import FLANN
from benchmarks.algorithms.Pynndescent.knn import PYNN
from benchmarks.algorithms.GDASC.knn import GDASC
import re
import argparse
import os
import sys
import platform
import datetime
import logging
import psutil  # Install with `pip install psutil`
from cpuinfo import get_cpu_info  # Install with `pip install py-cpuinfo`


# Function to execute the particular experiment described in the configuration file provided
def experiment(config_file):

    # Split the file name to get the method to be used
    # by '_' and '.' characters, except if the '.' is between two digits
    #method = re.split('_|\.', config_file)[5]
    method = re.split(r'[_]|(?<!\d)\.(?!\d)', config_file)[5]

    # Print the configuration file name
    #print(config_file)

    # According to the method chosen, carry out the experiment described in the configuration file
    if method == 'Exact':
        Exact(config_file)

    elif method == 'GDASC':
        GDASC(config_file)

    elif method == 'FLANN':
        FLANN(config_file)

    elif method == 'PYNN':
        PYNN(config_file)

    else:
        print("Method not able")


# Function to execute the experiments described in the configuration files provided
def execute_experiments(argument, optional_filters=None):

    # Check if the argument is a single .ini file
    if argument.endswith('.ini'):
        config_files = [argument]
        dataset_name = argument.split('_')[2]

    # Check if the argument is a directory
    elif os.path.isdir("./benchmarks/config/" + argument):
        config_files = [f for f in os.listdir("./benchmarks/config/" + argument) if f.endswith('.ini')]
        dataset_name = argument

        # Apply optional filter if provided
        if optional_filters:
            filter_options = optional_filters
            config_files = [f for f in config_files if all(opt in f for opt in filter_options)]
    else:
        # Print usage message and exit if the argument is invalid
        print("Usage: ./execute_experiments.sh [dataset name or .ini file] [optional_filter]")
        sys.exit(22)

    # Get the current date and time, formatting it
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%d-%m-%Y_%H:%M")

    # Create a log file to store the performance of the k-nn experiments
    logging.basicConfig(
        filename="./benchmarks/logs/" + dataset_name + "/test_knn_" + dataset_name + "_" + str(formatted_time) + ".log",
        filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)

    logging.info('------------------------------------------------------------------------')
    logging.info('                    Experiments launcher for %s Dataset', dataset_name)
    logging.info('------------------------------------------------------------------------\n')

    # Log system information
    logging.info("Gathering system architecture information...\n")

    # Log architecture information
    logging.info("Platform: %s\n", platform.platform())

    # Log detailed CPU information
    cpu_info = get_cpu_info()
    logging.info("Processor: %s", cpu_info.get("brand_raw", "Unknown"))
    logging.info("Processor Architecture: %s", cpu_info.get("arch", "Unknown"))
    logging.info("Processor Cores: %d", psutil.cpu_count(logical=False))
    logging.info("Logical Processors: %d\n", psutil.cpu_count(logical=True))

    logging.info("Python Version: %s\n", sys.version)

    # Log RAM information
    virtual_memory = psutil.virtual_memory()
    logging.info("Total RAM: %.2f GB", virtual_memory.total / (1024 ** 3))
    logging.info("Available RAM: %.2f GB", virtual_memory.available / (1024 ** 3))
    logging.info("Used RAM: %.2f GB", virtual_memory.used / (1024 ** 3))
    logging.info("RAM Usage Percentage: %.2f%%\n", virtual_memory.percent)
    logging.info('------------------------------------------------------------------------\n')

    # Execute the experiment for each configuration file
    for file in config_files:
        experiment(file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("argument", help="Name of the dataset or a single .ini file", type=str)
    parser.add_argument("optional_filters", help="Optional filters for .ini files", nargs='*', default=[])

    args = parser.parse_args()

    execute_experiments(args.argument, args.optional_filters)

    #exit(0)