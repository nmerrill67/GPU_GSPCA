
import numpy as np
from pycuda import gpuarray, autoinit, cumath
from skcuda import cublas

def preprocessmat(X):
	X -= X.mean()

#Look into http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process for a glimpse into this method
# Also in http://arxiv.org/pdf/0811.1081.pdf
# run preprocessmat! before running this
def CUDA_GSPCA(X, num_pcs, epsilon=0.0001, max_iter=10000):
	
	cublas = cublas.cublasCreate() # intialize the gpu linalg library

	R = gpuarray.to_gpu(X) # move data to gpu

	Lambda = gpuarray.zeros((R.shape[0],1), np.float32)

	P = gpuarray.zeros((X.shape[1],X.shape[1]), np.float32)

	T = gpuarray.zeros((X.shape[0],num_pcs), np.float32)

	for k in xrange(num_pcs):

		print k,"th PC loop"
		mu = 0.0
		T[:,k] = R[:,k]

		for j in xrange(max_iter):

			
			P[:,k] = linalg.mdot(linalg.transpose(R),T[:,k])

						
			if k > 0:
				A = linalg.dot(linalg.transpose(U[:,k]), U[:,k]) 
				U[:,k] -= U[:,k] * A

			L2 = linalg.norm(U[:,k])
			U[:,k] /= L2
			V[:,k] = linalg.dot(R, U[:,k])

			if k>0:
				B = linalg.dot(linalg.transpose(V[:,k]), V[:,k])
				V[:,k] -= V[:,k] * B

			Lambda[k] = linalg.norm(V[:,k])
			V[:,k] /= Lambda[k]

			if cummath.fabs(Lambda[k]-mu) < epsilon:
				break

			mu = Lambda[k]

		R -= Lambda[k] * linalg.dot(V[:,k], linalg.transpose(U[:,k]))
	T = linalg.mdot(V, Lambda)
	P = U

	cublasDestroy(cublas)

	return T.get() # return the cpu data

if __name__=='__main__':
	X = np.random.rand(400,56).astype(np.float32)
	preprocessmat(X)
	T = CUDA_GSPCA(X,4,10e-5)
	print T.shape
