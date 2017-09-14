
import numpy as np
from pycuda import gpuarray, autoinit, cumath
from skcuda import cublas, misc

#Look into http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process for a glimpse into this method
# Also in http://arxiv.org/pdf/0811.1081.pdf
# run preprocessmat! before running this
def CUDA_GSPCA(X, num_pcs, epsilon=0.0000001, max_iter=10000):
	
	h = cublas.cublasCreate() # create a handle to the c library

	misc.init()

	R = gpuarray.to_gpu(X) # move data to gpu

	n = R.shape[0] # num samples

	p = R.shape[1] # num features

	Lambda = gpuarray.zeros((num_pcs,1), np.float32)

	P = gpuarray.zeros((X.shape[1],num_pcs), np.float32)

	T = gpuarray.zeros((X.shape[0],num_pcs), np.float32)


	# mean centering data
	U = misc.sum(R,axis=0) # sum the columns of R



	for i in xrange(p):
		cublas.cublasSaxpy(h, n, -1.0/p, U.gpudata, 1, R[:,i].gpudata, 1) 	

	Lk = 0.0

	for k in xrange(num_pcs):

		print k," PC loop"

		mu = 0.0

		cublas.cublasDcopy(h, n, R[:,k].gpudata, 1, T[:,k].gpudata, 1)

		for j in xrange(max_iter):

		
			print j, 'inner loop'
	
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


			Lk = Lambda[k].get()[0]
			cublas.cublasSscal(h, n, 1.0/Lk, T[:,k].gpudata, 1)
		
			
			

			if abs(Lk-mu) < epsilon:
				break

			mu = Lk

		# end for j

		print 'before update R', P.gpudata, '\n'
		cublas.cublasSger(h, n, p, (0-Lk), T[:,k].gpudata, 1, P[:,k].gpudata, 1, R.gpudata, n)

		print 'after update R', P.gpudata,'\n'

	# end for k

	for k in xrange(num_pcs):
		cublas.cublasDscal(h, n, Lk, T[:,k].gpudata, 1) 

	T_cpu = T.get()

	cublas.cublasDestroy(h)

	return T_cpu # return the cpu data

if __name__=='__main__':
	X = np.random.rand(400,56).astype(np.float32)
	T = CUDA_GSPCA(X,4)
	print T.shape
