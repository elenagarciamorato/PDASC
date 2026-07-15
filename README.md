<p align="center">
  <img src="ANN_Experiments/figures/PDASC_logo.png" alt="PDASC" width="70%">
</p>

<p align="center">
  <a href="https://github.com/elenagarciamorato/PDASC">
    <img src="https://img.shields.io/badge/Source Code-PDASC-yellow?logo=github">
  </a>
    <a href="LICENSE">
      <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg">
    </a>
    <a href="https://doi.org/10.5281/zenodo.18247570">
      <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18247570-blue">
    </a>
  <a href="https://arxiv.org/abs/2405.13795">
    <img src="https://badgen.net/static/arXiv/2405.13795/red">
  </a>
</p>

PDASC (**Parametrizable Distributed Approximate Similarity Search with Clustering**) is a distributed Approximate Nearest Neighbour (ANN) search algorithm based on hierarchical clustering. It is designed to jointly support arbitrary distance functions, native distributed execution and memory-efficient indexing, making it suitable for large-scale similarity search in memory-constrained environments.

This repository provides:

- The reference implementation of the PDASC algorithm.
- A unified experimental framework for benchmarking multiple ANN methods under identical experimental conditions.
- Statistical tools for analysing distance distributions in datasets and PDASC indices.


---

# Algorithm Overview

PDASC builds a hierarchical clustering index through the **Multilevel Structure Algorithm (MSA)**. During index construction, the dataset is recursively partitioned into groups that are independently clustered, producing a multi-level hierarchy that can be naturally distributed across multiple computing nodes.

During query processing, the **Neighbours Search Algorithm (NSA)** traverses this hierarchy from the root towards the leaf groups. At each level, only the most promising branches are explored according to the selected pruning strategy, progressively reducing the search space until the final candidate set is obtained.

This hierarchical organization jointly enables support for arbitrary distance functions, native distributed execution and memory-efficient indexing while allowing the recall–efficiency trade-off to be adjusted through the configurable index topology.

<p align="center">
  <img src="ANN_Experiments/figures/PDASC_MSA.png" alt="Overview of PDASC index construction" width="90%">
</p>

<p align="center">
<b>Figure 1.</b> Overview of the index construction process performed by the Multilevel Structure Algorithm (MSA).
</p>

The resulting hierarchy is subsequently traversed by the Neighbours Search Algorithm (NSA) to perform approximate nearest neighbour retrieval.

---

# Key Features

## PDASC

- Support for arbitrary metric and non-metric distance functions.
- Distributed hierarchical ANN search.
- Memory-efficient index organization.
- Configurable index topology for different recall–efficiency trade-offs.

## Experimental Framework

- Unified evaluation pipeline for multiple ANN methods.
- CSV-driven experiment launcher.
- Automatic collection of search results and performance statistics.
- Reproducible benchmarking utilities.
- Distance distribution analysis tools.

---

# Repository Structure

The repository is organised around three main modules:

- **PDASC**: Reference implementation of the proposed ANN algorithm.
- **ANN_Experiments**: Unified experimentation and benchmarking framework.
- **dataset_analysis**: Statistical tools for analysing distance distributions and supporting the study of the proposed data-aware exploration strategy.

---

# Supported ANN Methods

| Method | Family |
|-----------------------|--------|
| PDASC | Hierarchical clustering |
| PyNNDescent | Graph |
| HNSW (NMSLIB & FAISS) | Graph |
| IVF (FAISS) | Clustering |
| LSH (FAISS) | Hashing |
| Annoy | Random Projection Trees |
| Exact Search | Exhaustive |

---

# Supported Datasets

| Dataset | Distance Emloyed |
|---------|------------------|
| Municipalities | Haversine        |
| MNIST | Euclidean        |
| GLOVE | Cosine           |
| NYTimes | Cosine           |
| MovieLens-10M | Jaccard          |
| Kosarak | Jaccard          |

---

# Getting Started

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/elenagarciamorato/PDASC.git
cd PDASC
python3 -m pip install -r requirements.txt
```

Detailed instructions for launching experiments, benchmarking results and using the distance distribution analysis tools are available in:

```text
docs/PythonUsage.md
```

---

# Project Structure

```text
PDASC/
├── PDASC/                  # Core ANN implementation
├── ANN_Experiments/        # Experiment launcher and benchmarking
├── dataset_analysis/       # Distance distribution analysis
├── datasets/               # Input datasets
├── docs/                   # User documentation
├── requirements.txt
└── README.md
```

---

# Citation

If you use PDASC in your research, please cite:

> Elena Garcia-Morato, Maria Jesus Algar, Cesar Alfaro, Felipe Ortega, Javier Gomez, Javier M. Moguerza.
>
> **A Memory-Efficient Distributed Algorithm for Approximate Nearest Neighbour Search with Arbitrary Distances**
>
> https://arxiv.org/abs/2405.13795