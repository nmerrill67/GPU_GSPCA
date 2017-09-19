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

        int M = 1000, m;
        int N = M/2, n;
        int K = 10;
        printf("\nProblem dimensions: MxN=%dx%d, K=%d", M, N, K);

        // initialize srand and clock

        srand (time(NULL));

        clock_t start=clock();
        double dtime;

        // initialize cublas
        cublasStatus status;
        status = cublasInit();

 if(status != CUBLAS_STATUS_SUCCESS)
        {
                fprintf(stderr, "! CUBLAS initialization error\n");
                return EXIT_FAILURE;
        }

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

        // allocate host memory for T, P, R
        double *T;
        T = (double*)malloc(M*K * sizeof(T[0]));;

        if(T == 0)
        {
                fprintf(stderr, "! host memory allocation error: T\n");
                return EXIT_FAILURE;
        }

        double *P;
        P = (double*)malloc(N*K * sizeof(P[0]));;
        if(P == 0)
        {
                fprintf(stderr, "! host memory allocation error: P\n");
                return EXIT_FAILURE;
        }

        double *R;
        R = (double*)malloc(M*N * sizeof(R[0]));;
        if(R == 0)
        {
                fprintf(stderr, "! host memory allocation error: R\n");
                return EXIT_FAILURE;
        }

        dtime = ((double)clock()-start)/CLOCKS_PER_SEC;
        printf("\nTime for data allocation: %f\n", dtime);

        // call gs_pca_cublas

        start=clock();

        memcpy(R, X, M*N * sizeof(X[0]));

        gs_pca_cublas(M, N, K, T, P, R);
        dtime = ((double)clock()-start)/CLOCKS_PER_SEC;

        printf("\nTime for device GS-PCA computation: %f\n", dtime);
        // the results are in T, P, R
        print_results(M, N, K, X, T, P, R);

        // clean up memory
        free(P);
        free(T);
        free(X);

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

