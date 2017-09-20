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
#define id(m, n, ld) (((n) * (ld) + (m)))


class KernelPCA 
{

private:
int K;

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

double* fit_transform(int M, int N, double *X);


}


