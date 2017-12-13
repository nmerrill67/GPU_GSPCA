# GPU PCA

If you have been looking for an open-source PCA library that runs on a GPU. This is it. This library is the only one of its kind.

This library implements PCA using the  GRAM-SCMIDT method, using the code written in [this paper](http://arxiv.org/pdf/0811.1081.pdf) as the backend for a c/c++ library and python wrappers. 

This code includes the c/c++ interface as well as the python interface to run PCA on a cuda-capable gpu. It models the API of sklearn.decomposition.KernelPCA. 

## Requirements:
C/C++ Library:
  - UNIX machine 
  - cmake
  - gcc, g++
  - ncurses (for waitbar)
  - gnu scientific library (for c++ demo)
  - cuda-capable gpu 
  - nvidia drivers and cuda installed

Python Wrappers:
  - C library requirements
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
If you have everything installed, this will build the backend c code, c demo, python wrappers, and the tests. Messages will be displayed if any of the above are not built due to libraries missing, so be on the lookout.

If you do not want to build the tests for whatever reason change the cmake call to:

`cmake -DBUILD_TESTS=0 ..` 

Make sure the install directory is in your python path, this can be done in your .bashrc as 

`export PYTHONPATH=/path/to/GPU_GSPCA/build:$PYTHONPATH`

## Testing

After building the library, simply run:

`make tests`

If you have the python wrappers built, it will run the C tests and the python tests in the test directory, otherwise it will just run the C tests.

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




