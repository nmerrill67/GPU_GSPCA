# Gram-Schmidt-PCA [![Build Status](https://travis-ci.org/nmerrill67/GPU_GSPCA.svg?branch=master)](https://travis-ci.org/nmerrill67/GPU_GSPCA)



# ________  ________  ___  ___          ________  ________  ________     
#|\   ____\|\   __  \|\  \|\  \        |\   __  \|\   ____\|\   __  \    
#\ \  \___|\ \  \|\  \ \  \\\  \       \ \  \|\  \ \  \___|\ \  \|\  \   
# \ \  \  __\ \   ____\ \  \\\  \       \ \   ____\ \  \    \ \   __  \  
#  \ \  \|\  \ \  \___|\ \  \\\  \       \ \  \___|\ \  \____\ \  \ \  \ 
#   \ \_______\ \__\    \ \_______\       \ \__\    \ \_______\ \__\ \__\
#    \|_______|\|__|     \|_______|        \|__|     \|_______|\|__|\|__|
                                                                        
                                                                        
                                                                        


This library implements PCA using the  GRAM-SCMIDT method, using the code written in http://arxiv.org/pdf/0811.1081.pdf as the backend for python wrappers. 

This code includes the c/c++ interface as well as the python interface to run PCA on a cuda-capable gpu. It models the API of sklearn.decomposition.KernelPCA.

Only fit_transform is implemented, so the pca model cannot be used between different datasets. However, this is not very imortant since the compute time for a model on the gpu is orders of magnitude less thaton the cpu using sklearn.  

## Requirements:
- UNIX machine 
- cmake
- gcc, g++
- cuda-capable gpu 
- nvidia drivers and cuda installed
- boost and boost python
- python 2.7 
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

`gpu_pca = KernelPCA(<number of components here>) # do KernelPCA(-1) to return all principal components
X_reduced = gpu_pca.fit_transform(X)`


Note that X and X_reduced are numpy arrays, and lie in the host's memory. The arrays are internally copied to gpu memory and back after the computation.




