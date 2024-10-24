import os
import h5py
import numpy as np
import pandas as pd
import sys
from benchmarks.neighbors_utils import *
from benchmarks.plotting.performance_utils import *


# Load the performance of the k-nn experiments regarding the selected dataset
def explore_experiments(dataset):

    # Set the directory path to load the experiments according to the dataset provided
    directory_path = "./benchmarks/NearestNeighbors/" + dataset

    results = []

    # For every .hdf5 file in the directory (file containing the neighbors and performances)
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.hdf5'):

                # Load the neighbors and performance of the k-nn experiments associated to that file
                indices, coords, distances, n_dist, search_time = load_neighbors_performance(directory_path + "/" + file)

                # Split the file name to get the information about the experiment
                parts = re.split(r'[_\.]', file)

                # If the method is GDASC
                if parts[4] == 'GDASC':
                    # We store the information about the experiment associated with the file
                    results.append({
                        'Method': parts[4],
                        'Distance': parts[3],
                        'k': parts[2],
                        'radius': int(parts[7][1:]),
                        'Algorithm': parts[8],
                        'Implementation': parts[9],
                        'Dist_Computed': np.median(n_dist),
                        # Get the recall of the experiment
                        'Recall': get_recall_new(dataset, parts[2], parts[3], indices, coords, distances),
                        'Search_Time': search_time
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
                        'Implementation': None,
                        'Dist_Computed': np.median(n_dist),
                        # Get the recall of the experiment
                        'Recall': get_recall_new(dataset, parts[2], parts[3], indices, coords, distances),
                        'Search_Time': search_time
                    })

        # Return the results
        return pd.DataFrame(results).sort_values(by=['Method', 'radius'], ascending=[True, False])


if __name__ == "__main__":

    dataset=("wdbc")

    # Explore the results of the experiments regarding the dataset provided
    df = explore_experiments(dataset)

    # Print the results
    print(df.to_string())
