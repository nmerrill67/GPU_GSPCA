// C/C++ example for the CUBLAS (NVIDIA)
// implementation of PCA-GS algorithm
//
// M. Andrecut (c) 2008
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// includes, cuda
#include <cublas.h>


// matrix indexing convention
#define ind(m, n, ld) (((n) * (ld) + (m)))

// useful macro
#define __min__(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })


class KernelPCA 
{

private:
int K;
cublasStatus status;

public:
KernelPCA(int K);
~KernelPCA();


/*
 Fit a PCA model to the data matrix X, and return the principal components T. The memory for X is freed in the function. 
 


 input
 X: float* - host pointer to data array. The array represents an MxN matrix, where each M elements of X is the ith column of the matrix.
 M: int - number of rows (samples) in X
 N: int - number of columns (features) in X

 return
 T: float* - host pointer to transformed matrix, with the same indexing as X
*/

float* fit_transform(int M, int N, float *X);

/*
 Change the number of components after intitialization

 input: 
 K_: int - new number of components

 return:
 void
*/
void set_n_components(int K_);



};


