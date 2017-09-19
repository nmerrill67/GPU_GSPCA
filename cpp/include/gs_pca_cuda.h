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



// declarations
int gs_pca_cuda(int, int, int, double *, double *, double *);
int print_results(int, int, int, double *, double *, double *, double *);

