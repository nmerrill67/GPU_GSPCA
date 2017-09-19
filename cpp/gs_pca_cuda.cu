#include "gs_pca_cuda.h"


double* gs_pca_cuda(int M, int N, int K, double *R)
	{
	// PCA model: X = TP’ + R
	//
	//
	//
	//
	// input: X, MxN matrix
	// input: M = number of
	// input: N = number of
	// input: K = number of (data) rows in X columns in X components (K<=N)
	// output: T, MxK scores matrix
	// output: P, NxK loads matrix
	// output: R, MxN residual matrix

        // initialize cublas
        cublasStatus status;
        status = cublasInit();

        if(status != CUBLAS_STATUS_SUCCESS)
        {
                fprintf(stderr, "! CUBLAS initialization error\n");
                return EXIT_FAILURE;
        }



	// maximum number of iterations
	int J = 10000;

	// max error
	double er = 1.0e-7;
	int n, j, k;

	// transfer the host matrix X to device matrix dR
	double *dR = 0;
	status = cublasAlloc(M*N, sizeof(dR[0]), (void**)&dR);

	if(status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf (stderr, "! device memory allocation error (dR)\n");
		return EXIT_FAILURE;
	}

	status = cublasSetMatrix(M, N, sizeof(R[0]), R, M, dR, M);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf (stderr, "! device access error (write dR)\n");
		return EXIT_FAILURE;
	}

	// allocate device memory for T, P
	double *dT = 0;
	status = cublasAlloc(M*K, sizeof(dT[0]), (void**)&dT);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf (stderr, "! device memory allocation error (dT)\n");
		return EXIT_FAILURE;
	}

	double *dP = 0;
	status = cublasAlloc(N*K, sizeof(dP[0]), (void**)&dP);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf (stderr, "! device memory allocation error (dP)\n");
		return EXIT_FAILURE;
	}

	// allocate memory for eigenvalues
	double *L;
	L = (double*)malloc(K * sizeof(L[0]));;
	if(L == 0)
	{
		fprintf (stderr, "! host memory allocation error: T\n");
		return EXIT_FAILURE;
	}

	// mean center the data
	double *dU = 0;
	status = cublasAlloc(M, sizeof(dU[0]), (void**)&dU);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf (stderr, "! device memory allocation error (dU)\n");
		return EXIT_FAILURE;
	}

	cublasDcopy(M, &dR[0], 1, dU, 1);
	for(n=1; n<N; n++)
	{
		cublasDaxpy (M, 1.0, &dR[n*M], 1, dU, 1);
	}

	for(n=0; n<N; n++)
	{
		cublasDaxpy (M, -1.0/N, dU, 1, &dR[n*M], 1);
	}
	
	// GS-PCA
	double a;
	for(k=0; k<K; k++)
	{
		cublasDcopy (M, &dR[k*M], 1, &dT[k*M], 1);
		a = 0.0;
		for(j=0; j<J; j++)
		{
			cublasDgemv ('t', M, N, 1.0, dR, M, &dT[k*M], 1, 0.0, &dP[k*N], 1);
			if(k>0)
			{
				cublasDgemv ('t', N, k, 1.0, dP, N, &dP[k*N], 1, 0.0, dU, 1);
				cublasDgemv ('n', N, k, -1.0, dP, N, dU, 1, 1.0, &dP[k*N], 1);
			}
			cublasDscal (N, 1.0/cublasDnrm2(N, &dP[k*N], 1), &dP[k*N], 1);
			cublasDgemv ('n', M, N, 1.0, dR, M, &dP[k*N], 1, 0.0, &dT[k*M], 1);
			if(k>0)
			{
				cublasDgemv ('t', M, k, 1.0, dT, M, &dT[k*M], 1, 0.0, dU, 1);
				cublasDgemv ('n', M, k, -1.0, dT, M, dU, 1, 1.0, &dT[k*M], 1);
			}

			L[k] = cublasDnrm2(M, &dT[k*M], 1);
			cublasDscal(M, 1.0/L[k], &dT[k*M], 1);

			if(fabs(a - L[k]) < er*L[k]) break;
				a = L[k];
			}
	
			cublasDger (M, N, - L[k], &dT[k*M], 1, &dP[k*N], 1, dR, M);
		}
	for(k=0; k<K; k++)
	{
		cublasDscal(M, L[k], &dT[k*M], 1);
	}

        double *T;
        T = (double*)malloc(M*K * sizeof(T[0]));;

        if(T == 0)
        {
                fprintf(stderr, "! host memory allocation error: T\n");
                return EXIT_FAILURE;
        }


	// transfer device dT to host T
	cublasGetMatrix (M, K, sizeof(dT[0]), dT, M, T, M);

	// clean up memory
	free(R)
	free(L);
	status = cublasFree(dP);
	status = cublasFree(dT);
	status = cublasFree(dR);

        // shutdown
        status = cublasShutdown(); 
        if(status != CUBLAS_STATUS_SUCCESS) 
        { 
                fprintf (stderr, "! cublas shutdown error\n"); 
                return EXIT_FAILURE; 
        } 


	return T;

}



int print_results(int M, int N, int K,
	double *X, double *T, double *P, double *R)
{
	int m, n, k;
	// If M < 13 print the results on screen
	if(M > 12) return EXIT_SUCCESS;
	
	printf("\nX\n");
	for(m=0; m<M; m++)
	{
		for(n=0; n<N; n++)
		{
		printf("%+f ", X[id( m, n,M)]);
		}
		printf("\n");
	}

	printf("\nT\n");

	for(m=0; m<M; m++)
	{
		for(n=0; n<K; n++)
		{
		printf("%+f ", T[id(m, n, M)]);
		}
		printf("\n");
	}


	double a;
	printf("\nT’ * T\n");

	for(m = 0; m<K; m++)
	{
		for(n=0; n<K; n++)
		{
			a=0;
			for(k=0; k<M; k++)
			{
			a = a + T[id(k, m, M)] * T[id(k, n, M)];
			}
			printf("%+f ", a);
		}
		printf("\n");
	}

	printf("\nP\n");
	for(m=0; m<N; m++)
	{
		for(n=0; n<K; n++)
		{
			printf("%+f ", P[id(m, n, N)]);
		}
		printf("\n");
	}
	printf("\nP’ * P\n");


	for(m = 0; m<K; m++)
	{
		for(n=0; n<K; n++)
		{
			a=0; for(k=0; k<N; k++) a = a + P[id(k, m, N)] * P[id(k, n, N)];
			
			printf("%+f ", a);
		}
	
		printf("\n");
	}

	printf("\nR\n");

	for(m=0; m<M; m++)
		{
		for(n=0; n<N; n++)
		{
			printf("%+f ", R[id( m, n,M)]);
		}
		printf("\n");
	}


	return EXIT_SUCCESS;

}
