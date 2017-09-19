#include "gs_pca_cuda.h"

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

        int M = 256, m;
        int N = 52, n;
        int K = 4;
        printf("\nProblem dimensions: MxN=%dx%d, K=%d", M, N, K);

        // initialize srand and clock

        srand (time(NULL));

        clock_t start=clock();
        double dtime;

        // initialize cublas
        cublasStatus status;
        // initiallize some random test data X
        double *X;

        X = (double*)malloc(M*N * sizeof(X[0]));

        if(X == 0)
        {
                fprintf (stderr, "! host memory allocation error: X\n");
                return EXIT_FAILURE;
        }

        for(m = 0; m < M; m++)
        {
                for(n = 0; n < N; n++)
                {
                X[id(m, n, M)] = rand() / (double)RAND_MAX;
                }
        }

        dtime = ((double)clock()-start)/CLOCKS_PER_SEC;
        printf("\nTime for data allocation: %f\n", dtime);

        // call gs_pca_cublas

        start=clock();

	double *T; // results matrix

        T = gs_pca_cuda(M, N, K, X);
 
        dtime = ((double)clock()-start)/CLOCKS_PER_SEC;

        printf("\nTime for device GS-PCA computation: %f\n", dtime);
        // the results are in T, P, R
        //print_results(M, N, K, X, T, P, R);

        // shutdown
        status = cublasShutdown();
        if(status != CUBLAS_STATUS_SUCCESS)
        {
                fprintf (stderr, "! cublas shutdown error\n");
                return EXIT_FAILURE;
        }

        if(argc <= 1 || strcmp(argv[1], "-noprompt"))
        {
                printf("\nPress ENTER to exit...\n"); getchar();
        }
        return EXIT_SUCCESS;
}

