#include "kernel_pca.h"
// includes, GSL & CBLAS
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>


// main
int main(int argc, char** argv)
{
        // PCA model: X = TP’ + R
        //
        //
        //
        //
        // input:
        // input:
        // input:
        // input:
        // X, MxN matrix
        // M = number of
        // N = number of
        // K = number of
        // (data)
        // rows in X
        // columns in X
        // components (K<=N)
        // output: T, MxK scores matrix
        // output: P, NxK loads matrix
        // output: R, MxN residual matrix

        int M = 2000, m;
        int N = 100, n;
        int K = 12;
        printf("\nProblem dimensions: MxN=%dx%d, K=%d", M, N, K);

        // initialize srand and clock

        srand (time(NULL));

        // initiallize some random test data X
        float *X;

        X = (float*)malloc(M*N * sizeof(X[0]));

        if(X == 0)
        {
                fprintf (stderr, "! host memory allocation error: X\n");
                return EXIT_FAILURE;
        }

        for(m = 0; m < M; m++)
        {
                for(n = 0; n < N; n++)
                {
        	        X[ind(m, n, M)] = rand() / (double)RAND_MAX;
                }
        }

        double dtime;
	clock_t start;

	start = clock();
	
	KernelPCA* pca;

	pca = new KernelPCA(K);

	float *T; // results matrix

        dtime = ((double)clock()-start)/CLOCKS_PER_SEC;

        printf("\nTime for cublas initialization: %f\n", dtime);


	// X is freed in the function

        start=clock();
  
        T = pca->fit_transform(M, N, X);
 
        dtime = ((double)clock()-start)/CLOCKS_PER_SEC;

        printf("\nTime for device GS-PCA computation: %f\n", dtime);

	delete pca;

	// check  that the bases are orthagonal
	gsl_matrix_float* T_mat = gsl_matrix_float_alloc(M, K);

	for (m=0; m<M; m++)
	{
		for (n=0; n<K; n++)
		{
			gsl_matrix_float_set(T_mat, m, n, T[ind(m,n,M)]);
		}
	}
	
	float dot_product;
	const gsl_vector_float T0 = gsl_matrix_float_column(T_mat, 0).vector;
	const gsl_vector_float T1 = gsl_matrix_float_column(T_mat, 1).vector;

	int gsl_status;
	gsl_status = gsl_blas_sdot(&T0, &T1, &dot_product);

	gsl_matrix_float_free(T_mat);

	printf("\n T0 . T1 = %f\n", dot_product); // Should be ~ 0	

        if(argc <= 1 || strcmp(argv[1], "-noprompt"))
        {
                printf("\nPress ENTER to exit...\n"); getchar();
        }
        return EXIT_SUCCESS;
}

