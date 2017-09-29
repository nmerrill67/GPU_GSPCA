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


// This indexing macro is not used internally, but is useful for users contructing data arrays in c or c++ 

// matrix indexing convention for fortran-contiguous arrays
#define ind_f(m, n, num_rows) (((n) * (num_rows)) + (m))


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
 Fit a PCA model to the data matrix X, and return the principal components T. The memory for X is not freed in the function, so the user must do that after the call if X is no longer needed. 
 


 input
 X: double* - host pointer to data array. The array represents an MxN matrix, where each M elements of X is the ith column of the matrix.
 M: int - number of rows (samples) in X
 N: int - number of columns (features) in X
 verbose: bool - whether or not to display a progress bar in the terminal. This is very useful for large Xs

 return
 T: double* - host pointer to transformed matrix, with the same indexing as X
*/


double* fit_transform(int M, int N, double *X, bool verbose);


/*
 Overload of double-precision version.

 Fit a PCA model to the data matrix X, and return the principal components T. The memory for X is not freed in the function, so the user must do that after the call if X is no longer needed. 
 


 input
 X: float* - host pointer to data array. The array represents an MxN matrix, where each M elements of X is the ith column of the matrix.
 M: int - number of rows (samples) in X
 N: int - number of columns (features) in X
 verbose: bool - whether or not to display a progress bar in the terminal. This is very useful for large Xs

 return
 T: float* - host pointer to transformed matrix, with the same indexing as X
*/

float* fit_transform(int M, int N, float *X, bool verbose);

/*
 Change the number of components after intitialization

 input: 
 K_: int - new number of components

 return:
 void
*/
void set_n_components(int K_);

int get_n_components();

};


