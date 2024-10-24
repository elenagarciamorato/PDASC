#!/bin/bash

# To be able to run FLANN on python3, the following dependencies must be installed
# Source: ann_benchmarks (https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/algorithms/flann/Dockerfile)

# Update the package list
apt-get update

# Install necessary dependencies
apt-get install -y cmake pkg-config liblz4-dev

# If the repository benchmarks/algorithms/FLANN/flann already exists, remove it
rm -rf benchmarks/algorithms/FLANN/flann

# Install the FLANN library (substitute the URL with the desired one to make an installation outside the project)
git clone https://github.com/mariusmuja/flann benchmarks/algorithms/FLANN/

# Create a build directory for FLANN
mkdir benchmarks/algorithms/FLANN/flann/build

# Then change the directory to the FLANN build directory
cd benchmarks/algorithms/FLANN/flann/build

# Configure the build with CMake
cmake ..

# Compile the FLANN library using 4 parallel jobs
make -j4

# Install the compiled FLANN library
make install

# Verify the installation by importing pyflann in Python
python3 -c 'import pyflann'