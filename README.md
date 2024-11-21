# GDASC - General Distributed Approximate Similarity search with Clustering
[![DOI](https://zenodo.org/badge/827397858.svg)](https://zenodo.org/doi/10.5281/zenodo.13220359)

![GDASC](benchmarks/figures/fig_multilayer_structure.png)

This project proposes GDASC (General Distributed Approximate Similarity search with Clustering) a novel generalized algorithm for distributed approximate similarity search that accepts any arbitrary distance function. It employs clustering algorithms that induce Voronoi regions in a dataset and yield a representative element, such as k-medoids, to build a multilevel indexing structure suitable for large datasets with high dimensionality and sparsity, usually stored in distributed systems.
<!-- * This project proposes a new algorithm GDASC (General Distributed Approximate Similarity search with Clustering), a novel algorithm designed for efficient approximate similarity search.-->

<!-- 
This project proposes a new algorithm GDASC that is a generalized algorithm to solve the approximate nearest neighbours (ANN) search problem for distributed data that accepts any arbitrary distance function by employing data partitioning algorithms that induce Voronoi regions in a dataset and yield a representative element, such as k-medoids.-->

<!-- ## Installation: -->
## Summary of features
<!-- * GDASC constructs a multilevel indexing structure, making it suitable for large, high-dimensional, and sparse datasets typically stored in distributed systems. <!-- This algorithm is adaptable with various clustering algorithms, including k-means, k-medoids, and DBSCAN..., and has already been successfully applied to k-medoids.
* GDASC, is an adaptable algorithm  with various clustering algorithms, such as k-means and k-medoids. Notably, it has already been successfully applied with the k-medoids algorithm, demonstrating its versatility and effectiveness in diverse clustering scenarios.

* GDASC uses k-medoids to enhance compatibility with any distance function, Unlike many similarity search algorithms that rely on k-means (usually associated with the Euclidean distance).
-->


__Multilevel Indexing Structure and k-Nearest Neighbors Search__

GDASC framework implements an advanced multilevel indexing structure suitable for large, 
high-dimensional datasets, typically stored in distributed systems.

It relies on two algorithms to search for the k approximate nearest neighbors:
* The **Multilevel Structure Algorithm (MSA)** constructs the multilevel index by employing clustering algorithms that generate representative points summarizing the dataset, allowing GDASC to adapt to the specific characteristics of the data.
* The **Neighbors Search Algorithm (NSA)** complements this by efficiently searching for neighbors within the indexed structure, utilizing the multilevel representation to quickly identify relevant data points. 


__Versatile and Adaptable__


GDASC distinguishes itself as a versatile and adaptable framework for data indexing, capable of integrating various clustering algorithms and distance functions.

The proposed method for building the multilevel structure, therefore used in the search process, is designed to integrate any clustering algorithm that produces representative points summarising the underlying dataset and allows for the use of arbitrary distance measures, making the framework adaptable to the dataset’s specific properties.

This method greatly expands the range of applicable problems while ensuring scalability by utilizing a bottom-up strategy for building the search tree, providing a competitive edge over many current algorithms that rely on centralizing all data on a single server.


### Clustering Algorithms:


| Algorithm   | Parameters                                         | Implementation                                    |
|-------------|----------------------------------------------------|---------------------------------------------------|
| **k-means** | - `n_clusters`: Number of clusters (default: 8)    | - `scikit-learn`: `sklearn.cluster.KMeans`        |
|             | - `init`: Method for initialization (default: 'k-means++') | - `The Algorithms`: `Python.machine_learning.k_means_clust`|
|             | - `n_init`: Number of time the algorithm will run with different centroid seeds (default: 10) |                                                   |
|             | - `max_iter`: Maximum number of iterations (default: 300) |                                                   |
|             | - `tol`: Relative tolerance with regards to inertia to declare convergence (default: 1e-4) |                                                   |
| **k-medoids** | - `n_clusters`: Number of clusters (default: 8)    | - `scikit-learn-extra`: `sklearn_extra.cluster.KMedoids` |
|             | - `init`: Method for initialization (default: 'heuristic') | - `PyClustering`: `pyclustering.cluster.kmedoids` |
|             | - `max_iter`: Maximum number of iterations (default: 300) |                                                   |
|             | - `metric`: The distance metric to use (default: 'euclidean') |                                                   |


### Supported distances:

| Distance      | API         | Equation                                                                                                                                                                                                                                                                                                                   |
|---------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Euclidean** | `euclidean` | ![d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}](https://latex.codecogs.com/svg.latex?d%28x%2C%20y%29%20%3D%20%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5En%20%28x_i%20-%20y_i%29%5E2%7D)                                                                                                                                                 |
| **Manhattan** | `manhattan` | ![d(x, y) = \sum_{i=1}^n \|x_i - y_i\|](https://latex.codecogs.com/svg.latex?d%28x%2C%20y%29%20%3D%20%5Csum_%7Bi%3D1%7D%5En%20%7Cx_i%20-%20y_i%7C)                                                                                                                                                                         |
| **Chebyshev** | `chebyshev` | ![d(x, y) = \max_i \|x_i - y_i\|](https://latex.codecogs.com/svg.latex?d%28x%2C%20y%29%20%3D%20%5Cmax_i%20%7Cx_i%20-%20y_i%7C)                                                                                                                                                                                             |
| **Minkowski** | `minkowski` | ![d(x, y) = \left( \sum_{i=1}^n \|x_i - y_i\|^p \right)^{\frac{1}{p}}](https://latex.codecogs.com/svg.latex?d%28x%2C%20y%29%20%3D%20%5Cleft%28%20%5Csum_%7Bi%3D1%7D%5En%20%7Cx_i%20-%20y_i%7C%5Ep%20%5Cright%29%5E%7B%5Cfrac%7B1%7D%7Bp%7D%7D)                                                                             |
| **Cosine**    | `cosine`    | ![d(x, y) = 1 - \frac{\sum_{i=1}^n x_i y_i}{\sqrt{\sum_{i=1}^n x_i^2} \sqrt{\sum_{i=1}^n y_i^2}}](https://latex.codecogs.com/svg.latex?d%28x%2C%20y%29%20%3D%201%20-%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5En%20x_i%20y_i%7D%7B%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5En%20x_i%5E2%7D%20%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5En%20y_i%5E2%7D%7D) |
  
    

## Benchmarks:

To thoroughly assess its performance, a wide range of experiments is carried out on datasets differing in type, size, dimensionality, and sparsity.

By using the train set, we approach MSA using Euclidean, Manhattan, Chebyshev, and cosine distances, respectively. Then, we perform searches for the 5, 10, and 15 approximate nearest neighbors of every query point contained in the corresponding test set. 

Initially, we apply MSA with Euclidean, Manhattan, Chebyshev, and cosine distances on the training set. Subsequently, NSA is utilized to search for the 5, 10, and 15 approximate nearest neighbors of each query point in the test set. 
To assess GDASC's performance, we calculate the average recall for each distance metric and compare the results with those achieved by two other algorithms: PyNN and FLANN.
The average recall of the searches conducted is computed and the results are compared with those achieved by two other algorithms: PyNN and FLANN.

### Datasets:
| Dataset                                                       | Label       | N         | Dimensionality | High Sparsity | Data Type  | 
|---------------------------------------------------------------|-------------|-----------|----------------|---------------|------------|
| [**Municipalities**](https://doi.org/10.5281/zenodo.12759082) | X_muni      | 8,130     | 2              | No            | Geospatial |
| [**MNIST**](https://doi.org/10.5281/zenodo.12759284)          | X_MNIST     | 69,000    | 784            | Yes           | Image      |
| [**GLOVE**](https://doi.org/10.5281/zenodo.12759356)          | X_GLOVE     | 1,000,000 | 100            | No            | Text       |
| [**NYtimes**](https://doi.org/10.5281/zenodo.12760693)        | X_NYtimes   | 290,000   | 256            | No            | Text       |


### Approximate Algorithms:
  
* __PyNNDescent__  constructs a graph-based index by connecting each data point to its approximate nearest neighbours. It iteratively improves the graph structure through neighbour descent steps, employing techniques such as random initialisation, multi-tree searching, and pruning. The algorithm also provides parameters to control the trade-off between search accuracy and computational cost.
* __FLANN__ employs techniques such as random projection and hierarchical subdivision to construct an index structure that enables faster search operations.
 
### Results
For each dataset, a pointplot illustrates the recall of three different algorithms across various distance metrics (Manhattan, Euclidean, Chebyshev, Cosine) in approximate nearest neighbor search.
<!-- 
### Dataset 1:

![municipalities](benchmarks/figures/municipios_avgRecall.png)

### Dataset 2:
![mnist](benchmarks/figures/MNIST_avgRecall.png)

### Dataset 3:
![glove](benchmarks/figures/GLOVE_avgRecall.png)

### Dataset 4:

![NYtimes](benchmarks/figures/NYtimes_avgRecall.png) -->


![grafos4](benchmarks/figures/4grafos.png)


The experimental results not only highlight the efficiency of this novel framework but also support the hypothesis that employing alternative distance measures beyond the Euclidean metric can yield better outcomes for nearest-neighbor queries, depending on the specific characteristics of the dataset.
  
## How to use GDASC API
This appendix is a guide to the usage of the GDASC Experiments Launcher and Benchmarking tool. Both are designed to facilitate the execution of approximate k-Nearest Neighbours searches by using the proposed method but also several SOTA related algorithms, then evaluating its performance on different contexts.

### Experiments Launcher
The experiments_launcher.py script is designed to execute machine learning experiments based on configuration files (.ini). 
#### Usage:  
The script expects two arguments:
- Experiment: The name of the dataset whose experiments sould be launched or a single .ini configuration file.
- Optional Filters: Optional filters to apply to the configuration files (if a directory is provided).<br />
`python3 -m benchmarks/experiments_launcher <dataset_name_or_ini_file> [optional_filters]`

#### Examples:  
- To run an experiment using a single .ini file:<br />
`python3 -m benchmarks/experiments_launcher knn_NYtimes_10_chebyshev_GDASC_tg1000_nc500_r30_kmedoids_fastkmedoids.ini`
- To run experiments for a dataset with optional filters:<br />
`python3 -m benchmarks/experiments_launcher NYtimes chebyshev GDASC`

### Performance Benchmarking
The performance_benchmark.py script is designed to evaluate the performance of machine learning algorithms.

#### Usage:  
The script expects two arguments:
- Dataset: The name of the dataset whose experiments want to be benchmarked.
- Optional Filters: Optional filters to only show info about the desired experiments.
`python3 -m benchmarks/performance_benchmark <dataset_name> [optional_filters]`

#### Examples:  
- To benchmark experiments for a dataset:<br />
`python3 -m benchmarks/performance_benchmark NYtimes`

- To benchmark experiments for a dataset with optional filters:<br />
`python3 -m benchmarks/performance_benchmark NYtimes chebyshev GDASC`
