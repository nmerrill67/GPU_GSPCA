import numpy as np
from pycuda import gpuarray, cumath, autoinit
from skcuda import cublas, misc, linalg


class KernelPCA():

	
	def __init__(self, n_components=None, handle=None, epsilon=0.0000001, max_iter=10000):
		
		self.n_components = n_components
		self.epsilon = epsilon
		self.max_iter = max_iter	
		misc.init()
		
		if handle is None:
			self.h = misc._global_cublas_handle # create a handle to initialize cublas
		else:	
			self.h = handle
			

	def fit_transform(self, R_gpu):

		"""
		Principal Component Analysis.

		Compute the first K principal components of R_gpu using the
		Gram-Schmidt orthogonalization algorithm provided by [Andrecut, 2008].

		Parameters
		----------
		R_gpu: pycuda.gpuarray.GPUArray
			NxP (N = number of samples, P = number of variables) data matrix that needs 
			to be reduced. R_gpu can be of type numpy.float32 or numpy.float64.
			Note that if R_gpu is not instantiated with the kwarg 'order="F"', 
			specifying a fortran-contiguous (row-major) array structure,
			fit_transform will throw an error.
		n_components: int
			The number of principal component column vectors to compute in the output 
			matrix.
		epsilon: float	
			The maximum error tolerance for eigen value approximation.
		max_iter: int
			The maximum number of iterations in approximating each eigenvalue  
		

		Returns
		-------
		T_gpu: pycuda.gpuarray.GPUArray
			`NxK` matrix of the first K principal components of R_gpu. 

		References
		----------
		`[Andrecut, 2008] <https://arxiv.org/pdf/0811.1081.pdf>`_
		

		Notes
		-----
		If n_components was not set, then `K = min(N, P)`. Otherwise, `K = min(n_components, N, P)`
		
		"""

		if R_gpu.flags.c_contiguous:
			raise ValueError("Array must be fortran-contiguous. Please instantiate with 'order=\"F\"'")
	
		n = R_gpu.shape[0] # num samples

		p = R_gpu.shape[1] # num features

		# choose either single or doubel precision cublas functions
		if R_gpu.dtype == 'float32':

			cuAxpy = cublas.cublasSaxpy
			cuCopy = cublas.cublasScopy
			cuGemv = cublas.cublasSgemv
			cuNrm2 = cublas.cublasSnrm2
			cuScal = cublas.cublasSscal
			cuGer =	cublas.cublasSger

		elif R_gpu.dtype == 'float64':

			cuAxpy = cublas.cublasDaxpy
			cuCopy = cublas.cublasDcopy
			cuGemv = cublas.cublasDgemv
			cuNrm2 = cublas.cublasDnrm2
			cuScal = cublas.cublasDscal
			cuGer =	cublas.cublasDger



		else:
			raise ValueError("Array must be of type numpy.float32 or numpy.float64, not '" + R_gpu.dtype + "'") 

		n_components = self.n_components

		if n_components == None or n_components > n or n_components > p:
			n_components = min(n, p)	


		Lambda = np.zeros((n_components,1), dtype=np.float32, order="F") # kx1

		P_gpu = gpuarray.zeros((p, n_components), np.float32, order="F") # pxk

		T_gpu = gpuarray.zeros((n, n_components), np.float32, order="F") # nxk


		# mean centering data
		U_gpu = gpuarray.zeros((n,1), np.float32, order="F")
		U_gpu = misc.sum(R_gpu,axis=1) # nx1 sum the columns of R

		for i in xrange(p):
			cuAxpy(self.h, n, -1.0/p, U_gpu.gpudata, 1, R_gpu[:,i].gpudata, 1) 	


		for k in xrange(n_components):

			mu = 0.0

			cuCopy(self.h, n, R_gpu[:,k].gpudata, 1, T_gpu[:,k].gpudata, 1)

			for j in xrange(self.max_iter):

			
				cuGemv(self.h, 't', n, p, 1.0, R_gpu.gpudata, n, T_gpu[:,k].gpudata, 1, 0.0, P_gpu[:,k].gpudata, 1)
		
							
				if k > 0:

					cuGemv(self.h,'t', p, k, 1.0, P_gpu.gpudata, p, P_gpu[:,k].gpudata, 1, 0.0, U_gpu.gpudata, 1)  

					cuGemv (self.h, 'n', p, k, 0.0-1.0, P_gpu.gpudata, p, U_gpu.gpudata, 1, 1.0, P_gpu[:,k].gpudata, 1)


				l2 = cuNrm2(self.h, p, P_gpu[:,k].gpudata, 1)
				cuScal(self.h, p, 1.0/l2, P_gpu[:,k].gpudata, 1)

				cuGemv(self.h, 'n', n, p, 1.0, R_gpu.gpudata, n, P_gpu[:,k].gpudata, 1, 0.0, T_gpu[:,k].gpudata, 1)

				if k > 0:

					cuGemv(self.h, 't', n, k, 1.0, T_gpu.gpudata, n, T_gpu[:,k].gpudata, 1, 0.0, U_gpu.gpudata, 1)
					cuGemv(self.h, 'n', n, k, 0.0-1.0, T_gpu.gpudata, n, U_gpu.gpudata, 1, 1.0, T_gpu[:,k].gpudata, 1)
			

				Lambda[k] = cuNrm2(self.h, n, T_gpu[:,k].gpudata, 1)

				cuScal(self.h, n, 1.0/Lambda[k], T_gpu[:,k].gpudata, 1)
							

				if abs(Lambda[k] - mu) < self.epsilon*Lambda[k]:
					break


				mu = Lambda[k]

			# end for j

			cuGer(self.h, n, p, (0.0-Lambda[k]), T_gpu[:,k].gpudata, 1, P_gpu[:,k].gpudata, 1, R_gpu.gpudata, n)

		# end for k

		for k in xrange(n_components):
			cuScal(self.h, n, Lambda[k], T_gpu[:,k].gpudata, 1) 

		# free gpu memory
		P_gpu.gpudata.free()
		U_gpu.gpudata.free()

		return T_gpu # return the gpu array

if __name__=='__main__':

	from sklearn.decomposition import KernelPCA as KernelPCA_cpu
	from time import time

	X = np.random.rand(2000,100).astype(np.float32)
	
	# notice how it has to be Fortran contiguous
	dX = gpuarray.GPUArray((2000,100), np.float32, order="F")

	dX.set(X)

	pca_gpu = KernelPCA(n_components=4)

	t0 = time()
	T_gpu = pca_gpu.fit_transform(dX)
	t1 = time()

	print "gpu pca done\n"

	print "output shape : ", T_gpu.shape
	
	t_gpu = t1-t0

	print "GPU compute time: ", t_gpu

	dot_product = linalg.dot(T_gpu[:,0], T_gpu[:,1])

	print "T0 . T1 = ", dot_product

	"""
	pca_cpu = KernelPCA_cpu(n_components=174, n_jobs=-1)
	
	t2 = time()
	T2 = pca_cpu.fit_transform(X)
	t3 = time()

	t_cpu = t3-t2

	print "PCA for 2000x100, 4 components"
	print "CPU compute time: ", t_cpu

	"""

