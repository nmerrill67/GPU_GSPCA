#include "kernel_pca.h"
#include "gtest/gtest.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

KernelPCA* test_pca;

int K = 2;

TEST(kernel_pca, consructor)
{
	test_pca = new KernelPCA(K);
	EXPECT_TRUE(test_pca); // make sure its not a null pointer

}







