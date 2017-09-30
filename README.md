# Gram-Schmidt-PCA [![Build Status](https://travis-ci.org/nmerrill67/GPU_GSPCA.svg?branch=master)](https://travis-ci.org/nmerrill67/GPU_GSPCA)


                                                                        


This library implements PCA using the  GRAM-SCMIDT method, using the code written in [this paper](http://arxiv.org/pdf/0811.1081.pdf) as the backend for python wrappers. 

This code includes the c/c++ interface as well as the python interface to run PCA on a cuda-capable gpu. It models the API of sklearn.decomposition.KernelPCA.

Only fit_transform is implemented, so the pca model cannot be used between different datasets. However, this is not very important since the compute time for a model on the gpu is orders of magnitude less than on the cpu using sklearn.  

## Requirements:
- UNIX machine 
- cmake
- gcc, g++
- ncurses (for waitbar)
- gnu scientific library (for c++ demo)
- cuda-capable gpu 
- nvidia drivers and cuda installed
- boost and boost python
- python 2.7 
- numpy
- sklearn (for demo comparison to cpu pca implementation only)

## Installation

In a shell:
```
cd /path/to/GPU_GSPCA
mkdir build && cd build
cmake .. && make
```

Make sure the install directory is in your python path, this can be done in your .bashrc as 

`export PYTHONPATH=/path/to/GPU_GSPCA/build:$PYTHONPATH`


## Demos

For c++ demo, run `./build/main`

For python demo to compare to sklearn, run `python demo.py`

This compares this library to sklearn's KernelPCA in speed and accuracy. In general, this library blows sklearn out of the water in both. This is what I got running the python demo:

```
PCA for 10000x500 matrix. Computing 4 principal components


PCA |=================================================================================| ETA: 0h00m01s
GPU PCA compute time =  0.786515951157
CPU PCA compute time =  3.5805721283


Orthogonality Test. All dot products of the resulting principal components should be ~ 0.
This is tested by dotting the first and second largest eigenvectors (principal components) of the output for the sklearn's pca and this library's pca.


This library's GPU PCA: T0 . T1 =  1.623e-06
sklearns's CPU PCA: T0 . T1 =  -8.26332e-05
```

## Library Usage

`from py_kernel_pca import KernelPCA`

in any script that you want super fast pca.

Then, for a numpy array X, in either single or double precision:

`gpu_pca = KernelPCA(<number of components here>) # do KernelPCA(-1) to return all principal components
X_reduced = gpu_pca.fit_transform(X, verbose=True) # verbose shows the waitbar, default is no waitbar`


Note that X and X_reduced are numpy arrays, and lie in the host's memory. The arrays are internally copied to gpu memory and back after the computation.




