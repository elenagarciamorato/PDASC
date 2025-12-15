## API description
This appendix is a guide to the usage of the PDASC Experiments Launcher and Benchmarking tool. Both are designed to facilitate the execution of approximate k-Nearest Neighbours searches by using the proposed method but also several SOTA related algorithms, then evaluating its performance on different contexts.

### Experiments Launcher
The `experiments_launcher.py` script facilitates the execution of the experiments whose parameters are described in configuration files (.ini). 

#### Usage:  
The script expects two arguments:
- Experiment: The name of the dataset whose experiments should be launched or a single .ini configuration file.
- Optional Filters: Optional filters to apply to the configuration files (if a directory is provided).<br />
`python3 -m ANN_Experiments.experiments_launcher <dataset_name_or_ini_file> [optional_filters]`

#### Configuration Files:
The configuration files are .ini files that contain the parameters for each experiment to be launched. The parameters include:
- `dataset`: The name of the dataset to be used.
- `k`: The number of ANN to be searched.
- `distance`: The distance function to be used.
- `method`: The method to be used for the search (e.g., `PDASC`, `PyNNDescent`).'

If the configuration file is for an experiment using PDASC, it should also include:
- `gl`: The size of each group to be clusterized when building the PDASC index.
- `np`: The number of clusters to be used when clustering each group in the PDASC index.
- `n_nodes`: The number of data partitions to be used when building the PDASC index.
- `r`: The search radius to be used for narrowing down the search space in NSA.
- `algorithm`: The algorithm to be used for the search (e.g., `kmedoids`).
- `implementation`: The implementation of the algorithm to be used (e.g., `fasterpam`).

In case of employing the proposed pruning strategy, the configuration file should replace the `r` parameter with:
- `d_threshold`: The distance threshold to be used for pruning the search space in NSA.

If the configuration file is for an experiment using other methods, it should include the initial parameters that are required to configure it.

#### Examples:  
- To run an experiment using a single .ini file:<br />
`python3 -m ANN_Experiments.experiments_launcher test_knn_NYtimes_10_chebyshev_PDASC_nc500_tg1000_r30_n10.ini`
- To run experiments for a dataset with optional filters:<br />
`python3 -m benchmarks.experiments_launcher NYtimes chebyshev PDASC`

### Performance Benchmarking
The `performance_benchmark.py` script facilitates the the performance evaluation of these experiments.

#### Usage:  
The script expects two arguments:
- Dataset: The name of the dataset whose experiments want to be benchmarked.
- Optional Filters: Optional filters to only show info about the desired experiments.
`python3 -m ANN_Experiments.benchmark.performance_benchmark <dataset_name> [optional_filters]`

#### Examples:  
- To benchmark experiments for a dataset:<br />
`python3 -m ANN_Experiments.benchmark.performance_benchmark NYtimes`

- To benchmark experiments for a dataset with optional filters:<br />
`python3 -m ANN_Experiments.benchmark.performance_benchmark NYtimes chebyshev PDASC`

### Distances Distribution's Generation
The `distances_distribution_generator.py` script allows generating different Distribution Functions to analyze the distribution of distances in a dataset. 
These distributions are generate both from a random sample of elements from the dataset and from the prototypes that compose a specific layer of the previously generated PDASC index.

#### Usage:
The script should be launched as follows:
`python3 -m dataset_analysis.distances_distribution_generator -<Distribution_Functions> "('dataset_name', 'distance_function')" -np <np> -gl <gl> -size <size> -nodes <n_nodes>`

The script expects three arguments:
- The Distribution Function(s) to be generated.
  - `-pdfsNN`: Perform kNN Distances CDF analysis with a tuple or a set of tuples.
  - `-pdfsPW`: Perform Pairwise Distances PDF analysis with a tuple or a set of tuples.
  - `-cdfsNN`: Perform kNN Distances CDF analysis with a tuple or set of tuples.
  - `-cdfsPW`: Perform Pairwise Distances CDF analysis with a tuple or a set of tuples.
  - `-cdfsNN_SS`: Perform kNN Distances CDF analysis for with varying sample sizes of a single dataset and distance function.
  - `-cdfsPW_SS`: Perform Pairwise Distances CDF analysis with varying sample sizes.
  
- `-np`: Number of centroids to be used when clustering each group in the PDASC index.
- `-gl`: Size of each group to be clusterized when building the PDASC index
- `-size`: Sample size to use, as a percentage of the dataset (integer, e.g., `10` for 10%)
- `-nodes`: Number of distributed nodes composing the PDASC index.

**Note:** The arguments for the `-pdfsNN`, `-pdfsPW`, `-cdfsNN`,  `-cdfsPW`, `-cdfsNN_SS` and `-cdfsPW_SS` parameters must be provided as Python tuples, e.g., `("dataset_name", "distance_function")`.

#### Examples:
- kNN Distances PDF Analysis (`-pdfsNN`)
`python3 -m dataset_analysis.distances_distribution_generator -pdfsNN "('municipios', 'haversine')"  -np 30 -gl 60 -size 10 -nodes 3`

- Pairwise Distances PDF Analysis (`-pdfsPW`)
`python3 -m dataset_analysis.distances_distribution_generator -pdfsPW "('municipios', 'haversine')"  -np 30 -gl 60 -size 10 -nodes 3`

- kNN Distances CDF Analysis (`-cdfsNN`)
`python3 -m dataset_analysis.distances_distribution_generator -cdfsNN "('municipios', 'haversine')"  -np 30 -gl 60 -size 10 -nodes 3`

- Pairwise Distances CDF Analysis (`-cdfsPW`)
`python3 -m dataset_analysis.distances_distribution_generator -cdfsPW "('municipios', 'haversine')"  -np 30 -gl 60 -size 10 -nodes 3`

- kNN-CDF Analysis for different sample sizes(`-cdfsNN_SS`)
`python3 -m dataset_analysis.distances_distribution_generator -cdfsNN_SS "('municipios', 'haversine')" -np 30 -gl 60 `

- Pairwise Distances CDF Analysis for different sample sizes (`-cdfsPW_SS`)
`python3 -m dataset_analysis.distances_distribution_generator -pdfsNN_SS "('municipios', 'haversine')" -nc 30 -tg 60`

### Distances Distribution's Evaluation
The `distances_distribution_evaluation.py` script provides statistical analysis tools to compare distributions of pairwise distances in datasets using the Kolmogorov-Smirnov (KS) test and the Wasserstein distance. 
It is useful for evaluating how similar or different two distributions (a random sample of elements from the dataset and from the prototypes that compose a specific layer of the previously generated PDASC index) are.

#### Usage:
The script should be launched as follows:
`python3 -m dataset_analysis.distances_distribution_evaluation  -dataset <dataset_name> -dist <distance_function> -size <size> -nodes <n_nodes>`

The script expects four arguments:
- Dataset: The name of the dataset to process
- Distance Function: The distance function to be used for the analysis.
- `-size`: Sample size to use, as a percentage of the dataset (integer, e.g., `10` for 10%)
- `-nodes`: Number of distributed nodes composing the PDASC index.

#### Example
`python3 -m dataset_analysis.distances_distribution_generator -dataset MNIST -dist euclidean -np 500 -gl 1000 -size 10 -nodes 3`

#### Output
- Prints the KS statistic and its interpretation.
- Prints the p-value and its statistical significance.
- Prints the Wasserstein distance.
