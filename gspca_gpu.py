
import numpy as np
from pycuda import gpuarray, cumath, autoinit
from skcuda import cublas, misc


class KernelPCA():

	
	def __init__(self, n_components=None, epsilon=0.0000001, max_iter=10000):
		
		self.n_components = n_components
		self.epsilon = epsilon
		self.max_iter = max_iter	
		misc.init()
	
		self.h = cublas.cublasCreate() # create a handle to initialize cublas

	def fit_transform(self, X):

		"""
		Compute the first <n_components> principal components of X using the pycuda library.

		n: number of samples in X
		p: number of dimmensions of X
		n_components: number of principal components to compute

		Note that this function takes as input and returns numpy arrays, not gpuarrays. The gpu processing is all done internally in the function.

		Input:
		X - nxp numpy.array(dtype=float32): data matrix that needs to be reduced
		n_components - int: number of principal components to use
		epsilon - float: max error tolerance for eigen value calculation
		max_iter - int: maximum iterations to compute eigenvector 
		
		Outputs:
		T - nxk numpy.array: the first k principal components.
		"""
	

		R = gpuarray.to_gpu(X) # nxp move data to gpu

		n = R.shape[0] # num samples

		p = R.shape[1] # num features

		n_components = self.n_components

		if n_components == None or n_components > n or n_components > p:
			n_components = min(n, p)	


		Lambda = np.zeros((n_components,1), dtype=np.float32) # kx1

		P = gpuarray.zeros((p, n_components), np.float32) # pxk

		T = gpuarray.zeros((n, n_components), np.float32) # nxk


		# mean centering data
		U = gpuarray.zeros((n,1), np.float32)
		U = misc.sum(R,axis=1) # nx1 sum the columns of R

		for i in xrange(p):
			cublas.cublasSaxpy(self.h, n, -1.0/p, U.gpudata, 1, R[:,i].gpudata, 1) 	


		for k in xrange(n_components):

			mu = 0.0

			cublas.cublasScopy(self.h, n, R[:,k].gpudata, 1, T[:,k].gpudata, 1)

			for j in xrange(self.max_iter):

			
				cublas.cublasSgemv(self.h, 't', n, p, 1.0, R.gpudata, n, T[:,k].gpudata, 1, 0.0, P[:,k].gpudata, 1)
		
							
				if k > 0:

					cublas.cublasSgemv(self.h,'t', p, k, 1.0, P.gpudata, p, P[:,k].gpudata, 1, 0.0, U.gpudata, 1)  

					cublas.cublasSgemv (self.h, 'n', p, k, 0.0-1.0, P.gpudata, p, U.gpudata, 1, 1.0, P[:,k].gpudata, 1)


				l2 = cublas.cublasSnrm2(self.h, p, P[:,k].gpudata, 1)
				cublas.cublasSscal(self.h, p, 1.0/l2, P[:,k].gpudata, 1)

				cublas.cublasSgemv(self.h, 'n', n, p, 1.0, R.gpudata, n, P[:,k].gpudata, 1, 0.0, T[:,k].gpudata, 1)

				if k > 0:

					cublas.cublasSgemv(self.h, 't', n, k, 1.0, T.gpudata, n, T[:,k].gpudata, 1, 0.0, U.gpudata, 1)
					cublas.cublasSgemv(self.h, 'n', n, k, 0.0-1.0, T.gpudata, n, U.gpudata, 1, 1.0, T[:,k].gpudata, 1)
			

				Lambda[k] = cublas.cublasSnrm2(self.h, n, T[:,k].gpudata, 1)

				cublas.cublasSscal(self.h, n, 1.0/Lambda[k], T[:,k].gpudata, 1)
							

				if abs(Lambda[k] - mu) < self.epsilon*Lambda[k]:
					break

				if j==self.max_iter-1:
					print "max_iter reached"

				mu = Lambda[k]

			# end for j

			cublas.cublasSger(self.h, n, p, (0.0-Lambda[k]), T[:,k].gpudata, 1, P[:,k].gpudata, 1, R.gpudata, n)

		# end for k

		for k in xrange(n_components):
			cublas.cublasSscal(self.h, n, Lambda[k], T[:,k].gpudata, 1) 

		return T.get() # return the cpu data

if __name__=='__main__':

	from sklearn.decomposition import KernelPCA as KernelPCA_cpu
	from time import time

	X = np.random.rand(2000,100).astype(np.float32)

	pca_gpu = KernelPCA(n_components=4)

	t0 = time()
	T = pca_gpu.fit_transform(X)
	t1 = time()

	print "gpu pca done"

	#	print "GPU compute time: ", t_gpu
	"""	
	t_gpu = t1-t0

	pca_cpu = KernelPCA_cpu(n_components=174, n_jobs=-1)
	
	t2 = time()
	T2 = pca_cpu.fit_transform(X)
	t3 = time()

	t_cpu = t3-t2

	print "PCA for 2000x100, 4 components"
	print "CPU compute time: ", t_cpu

	"""

