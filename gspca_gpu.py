
import numpy as np
from pycuda import gpuarray, autoinit, cumath
from skcuda import cublas, misc

#Look into http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process for a glimpse into this method
# Also in http://arxiv.org/pdf/0811.1081.pdf
# run preprocessmat! before running this
def CUDA_GSPCA(X, num_pcs, epsilon=0.0001, max_iter=10000):

	"""
	Compute the first <num_pcs> principal components of X using the pycuda library.

	n: number of samples in X
	p: number of dimmensions of X
	num_pcs: number of principal components to compute


	Input:
	X - nxp numpy.array(dtype=float32): data matrix that needs to be reduced
	num_pcs - 

	
	h = cublas.cublasCreate() # create a handle to the c library

	misc.init()

	R = gpuarray.to_gpu(X) # nxp move data to gpu

	n = R.shape[0] # num samples

	p = R.shape[1] # num features

	Lambda = np.zeros((num_pcs,1), dtype=np.float32) # kx1

	P = gpuarray.zeros((p, num_pcs), np.float32) # pxk

	T = gpuarray.zeros((n, num_pcs), np.float32) # nxk


	# mean centering data
	U = gpuarray.zeros((n,1), np.float32)
	U = misc.sum(R,axis=1) # nx1 sum the columns of R

	for i in xrange(p):
		cublas.cublasSaxpy(h, n, -1.0/p, U.gpudata, 1, R[:,i].gpudata, 1) 	


	for k in xrange(num_pcs):

		mu = 0.0

		cublas.cublasScopy(h, n, R[:,k].gpudata, 1, T[:,k].gpudata, 1)

		for j in xrange(max_iter):

		
			cublas.cublasSgemv(h, 't', n, p, 1.0, R.gpudata, n, T[:,k].gpudata, 1, 0.0, P[:,k].gpudata, 1)
	
						
			if k > 0:

				cublas.cublasSgemv(h,'t', p, k, 1.0, P.gpudata, p, P[:,k].gpudata, 1, 0.0, U.gpudata, 1)  

				cublas.cublasSgemv (h, 'n', p, k, -1.0, P.gpudata, p, U.gpudata, 1, 1.0, P[:,k].gpudata, 1)


			l2 = cublas.cublasSnrm2(h, p, P[:,k].gpudata, 1)
			cublas.cublasSscal(h, p, 1.0/l2, P[:,k].gpudata, 1)

			cublas.cublasSgemv(h, 'n', n, p, 1.0, R.gpudata, n, P[:,k].gpudata, 1, 0.0, T[:,k].gpudata, 1)

			if k > 0:

				cublas.cublasSgemv(h, 't', n, k, 1.0, T.gpudata, n, T[:,k].gpudata, 1, 0.0, U.gpudata, 1)
				cublas.cublasSgemv (h, 'n', n, k, -1.0, T.gpudata, n, U.gpudata, 1, 1.0, T[:,k].gpudata, 1)
		

			Lambda[k,:] = np.array([cublas.cublasSnrm2(h, n, T[:,k].gpudata, 1)], dtype=np.float32)

			cublas.cublasSscal(h, n, 1.0/Lambda[k], T[:,k].gpudata, 1)
		
			
			

			if abs(Lambda[k] - mu) < epsilon:
				break

			mu = Lambda[k]

		# end for j

		cublas.cublasSger(h, n, p, (0-Lambda[k]), T[:,k].gpudata, 1, P[:,k].gpudata, 1, R.gpudata, n)

	# end for k

	for k in xrange(num_pcs):
		cublas.cublasSscal(h, n, Lambda[k], T[:,k].gpudata, 1) 

	T_cpu = T.get()

	cublas.cublasDestroy(h)

	return T_cpu # return the cpu data

if __name__=='__main__':
	X = np.random.rand(400,56).astype(np.float32)
	T = CUDA_GSPCA(X,4)
	print T.shape
