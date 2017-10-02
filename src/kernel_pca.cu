#include "kernel_pca.h"
#include <stdio.h> /* for fprintf and stderr */





double* dev_fit_transform(cublasHandle_t h, int M, int N, double *dR, int K)
{


	cudaError_t status;

	// maximum number of iterations
	int J = 10000;

	// max error
	double er = 1.0e-7;

        // if no K specified, or K > min(M, N)
        int K_;
        K_ = min(M, N);
        if (K == -1 || K > K_) K = K_;

	

	int n, j, k;

	// allocate device memory for T, P
	double *dT = 0;
	status = cudaMalloc(&dT, M*K*sizeof(dT[0]));
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "! cuda memory allocation error (dT)\n");
	}

	double *dP = 0;
	status = cudaMalloc(&dP, N*K*sizeof(dP[0]));
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "! cuda memory allocation error (dP)\n");
	}

	// allocate memory for eigenvalues
	double *L;
	L = (double*)malloc(K * sizeof(L[0]));;
	if(L == 0)
	{
		fprintf(stderr, "! memory allocation error: T\n");
	}

	// mean center the data
	double *dU = 0;
	status = cudaMalloc(&dU, M*sizeof(dU[0]));
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "! cuda memory allocation error (dU)\n");
	}

	cublasDcopy(h, M, &dR[0], 1, dU, 1);

	double one = 1.0;
	double n_one = -1.0;
	for(n=1; n<N; n++)
	{
		cublasDaxpy(h, M, &one, &dR[n*M], 1, dU, 1);
	}

	double neg_one_n = -1.0/N;

	for(n=0; n<N; n++)
	{
		cublasDaxpy(h, M, &neg_one_n, dU, 1, &dR[n*M], 1);
	}

	double zero = 0.0;
	double *norm;
	double one_over_norm;	
	double one_Lk;
	double n_Lk;
	// GS-PCA
	double a;
	for(k=0; k<K; k++)
	{
		cublasDcopy (h, M, &dR[k*M], 1, &dT[k*M], 1);
		a = 0.0;
		for(j=0; j<J; j++)
		{
			cublasDgemv (h, CUBLAS_OP_T, M, N, &one, dR, M, &dT[k*M], 1, &zero, &dP[k*N], 1);
			if(k>0)
			{
				cublasDgemv (h, CUBLAS_OP_T, N, k, &one, dP, N, &dP[k*N], 1, &zero, dU, 1);
				cublasDgemv (h, CUBLAS_OP_N, N, k, &n_one, dP, N, dU, 1, &one, &dP[k*N], 1);
			}
	
			cublasDnrm2(h, N, &dP[k*N], 1, norm);	
			one_over_norm = 1.0/(*norm);
			cublasDscal (h, N, &one_over_norm , &dP[k*N], 1);
			cublasDgemv (h, CUBLAS_OP_N, M, N, &one, dR, M, &dP[k*N], 1, &zero, &dT[k*M], 1);
			if(k>0)
			{
				cublasDgemv (h, CUBLAS_OP_T, M, k, &one, dT, M, &dT[k*M], 1, &zero, dU, 1);
				cublasDgemv (h, CUBLAS_OP_N, M, k, &n_one, dT, M, dU, 1, &one, &dT[k*M], 1);
			}

			cublasDnrm2(h, M, &dT[k*M], 1, &L[k]);
			one_Lk = 1.0/L[k];
			cublasDscal(h, M, &one_Lk, &dT[k*M], 1);

			if(fabs(a - L[k]) < er*L[k]) break;
			
			a = L[k];
			
		}
		n_Lk = - L[k];
			
		cublasDger (h, M, N, &n_Lk, &dT[k*M], 1, &dP[k*N], 1, dR, M);

	}

	for(k=0; k<K; k++)
	{
		cublasDscal(h, M, &L[k], &dT[k*M], 1);
	}

	// clean up memory
	free(L);
	status = cudaFree(dP);
	status = cudaFree(dU);

	return dT;

}



float* dev_fit_transform(cublasHandle_t h, int M, int N, float *dR, int K)
{

	cudaError_t status;

	// maximum number of iterations
	int J = 10000;

	// max error
	float er = 1.0e-7;

        // if no K specified, or K > min(M, N)
        int K_;
        K_ = min(M, N);
        if (K == -1 || K > K_) K = K_;

	int n, j, k;

	// allocate device memory for T, P
	float *dT = 0;
	status = cudaMalloc(&dT, M*K*sizeof(dT[0]));
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "! cuda memory allocation error (dT)\n");
	}

	float *dP = 0;
	status = cudaMalloc(&dP, N*K*sizeof(dP[0]));
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "! cuda memory allocation error (dP)\n");
	}

	// allocate memory for eigenvalues
	float *L;
	L = (float*)malloc(K * sizeof(L[0]));;
	if(L == 0)
	{
		fprintf(stderr, "! memory allocation error: T\n");
	}

	// mean center the data
	float *dU = 0;
	status = cudaMalloc(&dU, M*sizeof(dU[0]));
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "! cuda memory allocation error (dU)\n");
	}

	float one = 1.0;
	float n_one = -1.0;

	cublasScopy(h, M, &dR[0], 1, dU, 1);
	for(n=1; n<N; n++)
	{
		cublasSaxpy (h, M, &one, &dR[n*M], 1, dU, 1);
	}

	float neg_one_n = -1.0/N;
	for(n=0; n<N; n++)
	{
		cublasSaxpy (h, M, &neg_one_n, dU, 1, &dR[n*M], 1);
	}
	
	float zero = 0.0;
	float *norm;
	float one_over_norm;	
	float one_Lk;
	float n_Lk;
	// GS-PCA
	float a;
	for(k=0; k<K; k++)
	{
		cublasScopy (h, M, &dR[k*M], 1, &dT[k*M], 1);
		a = 0.0;
		for(j=0; j<J; j++)
		{
			cublasSgemv (h, CUBLAS_OP_T, M, N, &one, dR, M, &dT[k*M], 1, &zero, &dP[k*N], 1);
			if(k>0)
			{
				cublasSgemv (h, CUBLAS_OP_T, N, k, &one, dP, N, &dP[k*N], 1, &zero, dU, 1);
				cublasSgemv (h, CUBLAS_OP_N, N, k, &n_one, dP, N, dU, 1, &one, &dP[k*N], 1);
			}
			cublasSnrm2(h, N, &dP[k*N], 1, norm);
			one_over_norm = 1.0/(*norm);
			cublasSscal (h, N, &one_over_norm, &dP[k*N], 1);
			cublasSgemv (h, CUBLAS_OP_N, M, N, &one, dR, M, &dP[k*N], 1, &zero, &dT[k*M], 1);
			if(k>0)
			{
				cublasSgemv (h, CUBLAS_OP_T, M, k, &one, dT, M, &dT[k*M], 1, &zero, dU, 1);
				cublasSgemv (h, CUBLAS_OP_N, M, k, &n_one, dT, M, dU, 1, &one, &dT[k*M], 1);
			}

			cublasSnrm2(h, M, &dT[k*M], 1, &L[k]);
			one_Lk = 1.0/L[k];
			cublasSscal(h, M, &one_Lk, &dT[k*M], 1);

			if(fabs(a - L[k]) < er*L[k]) break;
			
			a = L[k];
			
		}
			
		n_Lk = - L[k];
		cublasSger (h, M, N, &n_Lk, &dT[k*M], 1, &dP[k*N], 1, dR, M);
	

	}

	for(k=0; k<K; k++)
	{
		cublasSscal(h, M, &L[k], &dT[k*M], 1);
	}

	// clean up memory
	free(L);
	status = cudaFree(dP);
	status = cudaFree(dU);

	return dT;

}






