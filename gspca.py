
import numpy as np



def preprocessmat(X):
	meanvec = np.matrix(np.mean(X,1)).T
	X -= meanvec

#Look into http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process for a glimpse into this method
# Also in http://arxiv.org/pdf/0811.1081.pdf
# run preprocessmat! before running this
def GramSchmidtPCA(X,n,epsilon):
	R = np.copy(X)
	V = np.zeros_like(R)
	Lambda = np.zeros((R.shape[0],1))
	U = np.zeros((X.shape[1],X.shape[1]))
	for k in xrange(n):
		print k,"th PC loop"
		mu = 0
		V[k,:] = R[k,:]
		while True:
			U[k,:] = np.dot(R.T,V)[k,:]
			if k>0:
				intermediate = (U[k-1,:]*(U[k,:]).T)[0] / np.linalg.norm(U[k,:])
				A = np.dot( intermediate , U[k,:]) 
				U[k,:] = U[k-1,:] - A
			L2 = np.linalg.norm(U[k,:])
			U[k,:] = U[k,:]/L2
			V[k,:] = np.dot(R,U)[k,:]
			if k>0:
				B = (V[k-1,:]*(V[k,:]).T)[0] / np.linalg.norm(V[k,:]) * V[k,:] 
				V[k,:] = V[k-1,:] - B
			Lambda[k] = np.linalg.norm(V[k,:])
			V[k,:] = V[k,:]/Lambda[k]
			if np.abs(Lambda[k]-mu) < epsilon:
				break
			mu = Lambda[k]
		R = R - np.dot(Lambda[k],np.dot(U[k,:],(V[k,:]).T))
	T = V * Lambda
	P = U
	return T,P,R

if __name__=='__main__':
	A = np.random.rand(400,56)
	preprocessmat(A)
	print A
	T = GramSchmidtPCA(A,4,10e-5)[0]
	print T[:4,:].T.shape