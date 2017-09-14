
import numpy as np
from pycuda import gpuarray, autoinit, cumath
from skcuda import cublas, misc

#Look into http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process for a glimpse into this method
# Also in http://arxiv.org/pdf/0811.1081.pdf
# run preprocessmat! before running this
def CUDA_GSPCA(X, num_pcs, epsilon=0.0001, max_iter=10000):
	
	h = cublas.cublasCreate() # create a handle to the c library

	misc.init()

	R = gpuarray.to_gpu(X) # move data to gpu

	n = R.shape[0] # num samples

	p = R.shape[1] # num features

	Lambda = gpuarray.zeros((num_pcs,1), np.float32)

	P = gpuarray.zeros((X.shape[1],num_pcs), np.float32)

	T = gpuarray.zeros((X.shape[0],num_pcs), np.float32)


	# mean centering data
	U = misc.sum(R,axis=1) # sum the columns of R

	print p
	print U.shape

	for i in xrange(p):
		R[:,i] -= U / p # Subtract the mean from each row. Data is now mean centered 	

	for k in xrange(num_pcs):

		print k,"th PC loop"
		mu = 0.0

		T[:,k] = R[:,k]

		for j in xrange(max_iter):

			
			cublas.cublasSgemv(h, 't', n, p, 1.0, R.gpudata, n, T[:,k].gpudata, 1, 0.0, P[:,k].gpudata, 1)

						
			if k > 0:

				cublas.cublasSgemv(h,'t', p, k, P.gpudata, p, P[:,k].gpudata, 1, 0.0, U, 1)  

				cublas.cublasSgemv (h, 'n', p, k, -1.0, P, p, U, 1, 1.0, P[:,k].gpudata, 1)

			cublas.cublasSscal(h, p, 1.0/cublas.cublasSnrm2(h, p, P[:,k].gpudata, 1), P[:,k].gpudata, 1)

			cublas.cublasSgemv(h, 'n', n, p, 1.0, R.gpudata, n, P[:,k].gpudata, 1, 0.0, T[:,k].gpudata, 1)

			if k > 0:

				cublas.cublasSgemv(h, 't', n, k, 1.0, T.gpudata, n, T[:,k].gpudata, 1, 0.0, U.gpudata, 1)
				cublas.cublasSgemv (h, 'n', n, k, -1.0, T.gpudata, n, U.gpudata, 1, 1.0, T[:,k].gpudata, 1)
		

			Lambda[k].gpudata = cublas.cublasSnrm2(h, n, T[:,k].gpudata, 1)
			cublas.cublasSscal(n, 1.0/L[k], T[:,k].gpudata, 1)
		


			if cummath.fabs(Lambda[k]-mu) < epsilon:
				break

			mu = Lambda[k]

		cublas.cublasSger (h, n, p, -Lambda[k], T[:,k].gpudata, 1, P[:,k].gpudata, 1, R.gpudata, n)

	for k in xrange(num_pcs):
		cublas.cublasDscal(h, n, Lambda[k], T[:,k].gpudata, 1) 

	cublasDestroy(h)

	return T.get() # return the cpu data

if __name__=='__main__':
	X = np.random.rand(400,56).astype(np.float32)
	T = CUDA_GSPCA(X,4)
	print T.shape
