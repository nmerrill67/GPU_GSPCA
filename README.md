# Gram-Schmidt-PCA
Implements PCA GRAM-SCMIDT method to avoid the problem of non-orthagonal PCs by NIPALS.

Uses the algorithm written in http://arxiv.org/pdf/0811.1081.pdf as the backend for python wrappers.


This code includes the c/c++ interface as well as the python interface to run PCA on a cuda-capable gpu.

## Requirements:
- cmake
- gcc, g++
- cuda-capable gpu 
- nvidia drivers and cuda installed
- python
- numpy
- sklearn (for demo comparison to cpu pca implementation only)

## Installation

In a shell:

`cd /path/to/GPU_GSPCA
mkdir build && cd build
cmake .. && make`


Make sure the install directory is in your python path, this can be done in your .bashrc as 

`export PYTHONPATH=/path/to/GPU_GSPCA/build:$PYTHONPATH`


## Library Usage

`from py_kernel_pca import KernelPCA`

in any script that you want super fast pca.

Then, for a numpy array X:

`gpu_pca = KernelPCA(<number of components here>)
X_reduced = gpu_pca.fit_transform(X)`


Note that X and X_reduced are numpy arrays, and lie in the host's memory. The arrays are internally copied to gpu memory and back after the computation.




