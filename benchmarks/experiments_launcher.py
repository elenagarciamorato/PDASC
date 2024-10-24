from benchmarks.algorithms.Exact.knn import Exact
from benchmarks.algorithms.FLANN.knn import FLANN
from benchmarks.algorithms.Pynndescent.knn import PYNN
from benchmarks.algorithms.GDASC.knn import GDASC
import re
import argparse
import os
import sys

# Function to execute the particular experiment described in the configuration file provided
def experiment(config_file):

    # Extract the method from the configuration file name
    method = re.split('_|\.', config_file)[5]

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
def execute_experiments(argument, optional_filter=None):
    # Check if the argument is a single .ini file
    if argument.endswith('.ini'):
        config_files = [argument]
    # Check if the argument is a directory
    elif os.path.isdir(argument):
        config_files = [f for f in os.listdir(argument) if f.endswith('.ini')]
        # Apply optional filter if provided
        if optional_filter:
            config_files = [f for f in config_files if optional_filter in f]
    else:
        # Print usage message and exit if the argument is invalid
        print("Usage: ./execute_experiments.sh [/path/to/directory_or_file] [optional_filter]")
        sys.exit(22)

    # Execute the experiment for each configuration file
    for file in config_files:
        experiment(file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to a directory or a single .ini file", type=str)
    parser.add_argument("optional_filter", help="Optional filter for .ini files", nargs='?', default=None)

    args = parser.parse_args()

    execute_experiments(args.path, args.optional_filter)
