#include "kernel_pca.h"
#include "gtest/gtest.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

KernelPCA* test_pca;


int M = 1000, m;
int N = 100, n;


TEST(kernel_pca, default_consructor_test)
{
	test_pca = new KernelPCA;
	EXPECT_TRUE(test_pca) << "Default constructor failed"; // make sure its not a null pointer
}


TEST(kernel_pca, get_k_negative_one_test)
{
	EXPECT_EQ(test_pca->get_n_components(), -1) << "Default constructor does not set K to -1";
}

double *Td; // results matrix

TEST(kernel_pca, fit_transform_all_double_test)
{
	// initialize srand and clock

	srand(time(NULL));


	// initiallize some random test data X
	double *Xd;


        Xd = (double*)malloc(M*N * sizeof(Xd[0]));

        for(m = 0; m < M; m++)
        {
                for(n = 0; n < N; n++)
                {
        	        Xd[ind_f(m, n, M)] = rand() / (double)RAND_MAX;
                }
        }

	

        Td = test_pca->fit_transform(M, N, Xd, 1);

	EXPECT_TRUE(Td) << "double-precision fit_transform for all components returned a null pointer";

	free(Xd);	

} 

int gsl_status;

TEST(kernel_pca, double_all_orth_test)
{
	// check  that the bases are orthagonal
	gsl_matrix* T_mat = gsl_matrix_alloc(M, N);


	for (m=0; m<M; m++)
	{
		for (n=0; n<N; n++)
		{
			try
			{
				gsl_matrix_set(T_mat, m, n, Td[ind_f(m,n,M)]);
			}	
			catch(...) 
			{
				FAIL() << "Error getting data from double precision principal component score at (" << m << ", " << n << ")\n"; 
			}
		}
	}

	
	double dot_product;

	double max_dot_allowed = 0.0000001; // dot product of the orthagonal eigenvectors should be  close to zero. So lets compare it

	for (int i=1; i<N; i++) // check for each pair of consecutive cols
	{
		const gsl_vector T0 = gsl_matrix_column(T_mat, i-1).vector;
		const gsl_vector T1 = gsl_matrix_column(T_mat, i).vector;

		gsl_status = gsl_blas_ddot(&T0, &T1, &dot_product);
		
		EXPECT_EQ(gsl_status, 0) << "GSL double precision dot product failed";

		EXPECT_LT(dot_product, max_dot_allowed) << "double precision fit_transform for all components created weakly orthogonal eigenvectors";

	}

	gsl_matrix_free(T_mat);

	free(Td);	

}

float *Tf; // results matrix

TEST(kernel_pca, fit_transform_all_float_test)
{

	// initiallize some random test data X
	float *Xf;


        Xf = (float*)malloc(M*N * sizeof(Xf[0]));

        for(m = 0; m < M; m++)
        {
                for(n = 0; n < N; n++)
                {
        	        Xf[ind_f(m, n, M)] = rand() / (float)RAND_MAX;
                }
        }

	

        Tf = test_pca->fit_transform(M, N, Xf, 1);

	EXPECT_TRUE(Tf) << "single precision fit_transform for all components returned a null pointer";

	free(Xf);	

} 

TEST(kernel_pca, float_all_orth_test)
{
	// check  that the bases are orthagonal
	gsl_matrix_float* T_mat = gsl_matrix_float_alloc(M, N);


	for (m=0; m<M; m++)
	{
		for (n=0; n<N; n++)
		{
			try
			{
				gsl_matrix_float_set(T_mat, m, n, Tf[ind_f(m,n,M)]);

			}
                        catch(...)
                        {
                                FAIL() << "Error getting data from single precision principal component score at (" << m << ", " << n << ")\n";                  
                        }

		}
	}

	
	float dot_product;

	float max_dot_allowed = 0.0001;	

	for (int i=1; i<N; i++) // check for each pair of consecutive cols
	{
		const gsl_vector_float T0 = gsl_matrix_float_column(T_mat, i-1).vector;
		const gsl_vector_float T1 = gsl_matrix_float_column(T_mat, i).vector;

		gsl_blas_sdot(&T0, &T1, &dot_product);


		EXPECT_LT(dot_product, max_dot_allowed) << "single precision fit_transform for all components created weakly orthogonal eigenvectors";

	}

	gsl_matrix_float_free(T_mat);

	free(Tf);	

	delete test_pca;

}


KernelPCA* test_pca2;

int K = 2;

TEST(kernel_pca, constructor_test)
{
	test_pca2 = new KernelPCA(4);
	EXPECT_TRUE(test_pca2); // make sure its not a null pointer
	EXPECT_EQ(test_pca2->get_n_components(), 4);
}

TEST(kernel_pca, set_get_n_comp_test)
{

	test_pca2->set_n_components(K);

	EXPECT_EQ(test_pca2->get_n_components(), K); 

}


TEST(kernel_pca, fit_transform_double_test)
{

	// initiallize some random test data X
	double *Xd;


        Xd = (double*)malloc(M*N * sizeof(Xd[0]));

        for(m = 0; m < M; m++)
        {
                for(n = 0; n < N; n++)
                {
        	        Xd[ind_f(m, n, M)] = rand() / (double)RAND_MAX;
                }
        }

	

        Td = test_pca2->fit_transform(M, N, Xd, 1);

	EXPECT_TRUE(Td) << "double-precision fit_transform for 2 components returned a null pointer";

	free(Xd);	

} 


TEST(kernel_pca, double_orth_test)
{
	// check  that the bases are orthagonal
	gsl_matrix* T_mat = gsl_matrix_alloc(M,K);


	for (m=0; m<M; m++)
	{
		for (n=0; n<K; n++)
		{
			try
			{
				gsl_matrix_set(T_mat, m, n, Td[ind_f(m,n,M)]);
			}	
			catch(...) 
			{
				FAIL() << "Error getting data from double precision principal component score at (" << m << ", " << n << ")\n"; 
			}
		}
	}

	
	double dot_product;

	double max_dot_allowed = 0.0000001; // dot product of the orthagonal eigenvectors should be  close to zero. So lets compare it


	const gsl_vector T0 = gsl_matrix_column(T_mat, 0).vector;
	const gsl_vector T1 = gsl_matrix_column(T_mat, 1).vector;

	gsl_status = gsl_blas_ddot(&T0, &T1, &dot_product);
	
	EXPECT_EQ(gsl_status, 0) << "GSL double precision dot product failed";

	EXPECT_LT(dot_product, max_dot_allowed) << "double precision fit_transform for two components created weakly orthogonal eigenvectors";


	gsl_matrix_free(T_mat);

	free(Td);	

}


TEST(kernel_pca, fit_transform_float_test)
{

	// initiallize some random test data X
	float *Xf;


        Xf = (float*)malloc(M*N * sizeof(Xf[0]));

        for(m = 0; m < M; m++)
        {
                for(n = 0; n < N; n++)
                {
        	        Xf[ind_f(m, n, M)] = rand() / (float)RAND_MAX;
                }
        }

	

        Tf = test_pca2->fit_transform(M, N, Xf, 1);

	EXPECT_TRUE(Tf) << "single precision fit_transform for all components returned a null pointer";

	free(Xf);	

} 

TEST(kernel_pca, float_orth_test)
{
	// check  that the bases are orthagonal
	gsl_matrix_float* T_mat = gsl_matrix_float_alloc(M, K);


	for (m=0; m<M; m++)
	{
		for (n=0; n<K; n++)
		{
			try
			{
				gsl_matrix_float_set(T_mat, m, n, Tf[ind_f(m,n,M)]);

			}
                        catch(...)
                        {
                                FAIL() << "Error getting data from single precision principal component score at (" << m << ", " << n << ")\n";                  
                        }

		}
	}

	
	float dot_product;

	float max_dot_allowed = 0.0001;	


	const gsl_vector_float T0 = gsl_matrix_float_column(T_mat, 0).vector;
	const gsl_vector_float T1 = gsl_matrix_float_column(T_mat, 1).vector;

	gsl_blas_sdot(&T0, &T1, &dot_product);


	EXPECT_LT(dot_product, max_dot_allowed) << "single precision fit_transform for all components created weakly orthogonal eigenvectors";

	gsl_matrix_float_free(T_mat);

	free(Tf);	

	delete test_pca2;

}


