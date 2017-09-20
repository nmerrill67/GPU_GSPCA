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
 input
 X: double* - host pointer to data matrix 
 M: int - number of rows (samples) in X
 N: int - number of columns (features) in X

 return
 T: double* - host pointer to transformed matrix
*/

float* fit_transform(int M, int N, float *X);


};


