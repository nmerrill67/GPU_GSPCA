# Gram-Schmidt-PCA
Implements PCA GRAM-SCMIDT method to avoid the problem of non-orthagonal PCs by NIPALS.

Uses skcuda's cublas wrappers to implement GPU version stated in http://arxiv.org/pdf/0811.1081.pdf.

This python library was translated from the c code in the paper, and uses many of the same cublas functions as the paper, except with the python wrappers supplied by skcuda.

## Warning

This code currently has a bug. The columns of T are not currently orthogonal, as eigenvectors should be.

## Requirements:

- cuda-capable gpu 
- nvidia drivers and cuda installed
- python
- [pycuda](https://documen.tician.de/pycuda/)
- [skcuda](http://scikit-cuda.readthedocs.io/en/latest/)
- numpy
- sklearn (for demo comparison to cpu pca implementation only)


## Optional Cythonization

run `make` to build the shared c library if you want the code to run faster when imported into another python script.

## Library Usage

Make sure the cloned directory is in your python path, this can be done in your .bashrc as 

`export PYTHONPATH=/path/to/GPU_GSPCA:$PYTHONPATH`

After that, simply put 

`from gspca_gpu import KernelPCA`

in any script that you want super fast pca.

Then:


`gpu_pca = KernelPCA(n_components=<number of components here>)
X_reduced = gpu_pca.fit_transform(X)`







