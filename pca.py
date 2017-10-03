from ctypes import *
import sys
from skcuda.cublas import cublasCreate, cublasDestroy, _types
from pycuda.gpuarray import GPUArray, to_gpu
from pycuda import autoinit
import numpy as np





# Load kernel_pca library:
if 'linux' in sys.platform:
	_libkernel_pca_libname = 'build/libkernel_pca.so'
	'''
	elif sys.platform == 'darwin':
	    _libcublas_libname_list = ['libcublas.dylib']
	elif sys.platform == 'win32':
	    if sys.maxsize > 2**32:
		_libcublas_libname_list = ['cublas.dll'] + \
					  ['cublas64_%s.dll' % int(10*v) for v in _version_list]
	    else:
		_libcublas_libname_list = ['cublas.dll'] + \
					  ['cublas32_%s.dll' % int(10*v) for v in _version_list]
	'''
else:
	raise RuntimeError('unsupported platform. KernelPCA currently only supports linux')

# Print understandable error message when library cannot be found:

try:
	'''
	if sys.platform == 'win32':
	    _libcublas = ctypes.windll.LoadLibrary(_libcublas_libname)
	else:
	'''
	_libkernel_pca = cdll.LoadLibrary(_libkernel_pca_libname)

except OSError:
	raise OSError('kernel_pca library not found')

# notice how all float and double pointers are being called ints here
_libkernel_pca.dev_fit_transform_d.restype = int 
_libkernel_pca.dev_fit_transform_d.argtypes = [_types.handle, c_int, c_int, c_int, c_int]

_libkernel_pca.dev_fit_transform_f.restype = int
_libkernel_pca.dev_fit_transform_f.argtypes = [_types.handle, c_int, c_int, c_int, c_int]

_libkernel_pca.c_strided_to_f_contiguous_f.restype = None
_libkernel_pca.c_strided_to_f_contiguous_f.argtypes = [c_int, c_int, POINTER(c_int), c_int]

_libkernel_pca.f_to_c_contiguous_f.restype = None
_libkernel_pca.f_to_c_contiguous_f.argtypes = [c_int, c_int, POINTER(c_int), c_int]

_libkernel_pca.c_strided_to_f_contiguous_d.restype = None
_libkernel_pca.c_strided_to_f_contiguous_d.argtypes = [c_int, c_int, POINTER(c_int), c_int]

_libkernel_pca.f_to_c_contiguous_d.restype = None
_libkernel_pca.f_to_c_contiguous_d.argtypes = [c_int, c_int, POINTER(c_int), c_int]


class KernelPCA:
	
	def __init__(self, cublas_handle, n_components=-1):
		

		self.K = n_components

		self.h = h

	def fit_transform(self, X_gpu):
		
		if not X_gpu.flags.c_contiguous:	
			raise ValueError("Array must me C contiguous")	

		is_float = False
		if X_gpu.dtype == 'float32' and X_gpu.dtype != 'float64':
			is_float = True		
		else:
			raise ValueError("Array must be type float32 or float64, not '" + X_gpu.dtype + "'") 

		if len(X_gpu.shape) != 2:
			raise ValueError("PCA can only be performed on a 2D array")	
		
		M = X_gpu.shape[0]
		N = X_gpu.shape[1]

		# need the array strides to index the array internally
		strides = np.array([X_gpu.strides[0], X_gpu.strides[1]], dtype=np.intc)


		print "int gpudata", c_int(X_gpu.gpudata)

		if is_float:
			_libkernel_pca.c_strided_to_f_contiguous_f(M, N, strides.ctypes.data_as(POINTER(c_int)), int(X_gpu.gpudata))
	
			print "made it"	
			_fit_transform = _libkernel_pca.dev_fit_transform_f

		else: 
			_libkernel_pca.c_strided_to_f_contiguous_d(M, N, strides.ctypes.data_as(POINTER(c_int)), int(X_gpu.gpudata))

			_fit_transform = _libkernel_pca.dev_fit_transform_d

		# check if K is too big
        	K_ = min(M, N)
        	if self.K == -1 or self.K > K_:
			 self.K = K_

		T_gpu_ptr = _fit_transform(self.h, M, N, X_gpu.gpudata, self.K) 	
		
		# instantiate a new gpu array object from the returned F-contiguous gpu pointer
		T_gpu = GPUArray((M, self.K), X_gpu.dtype, gpudata=T_gpu_ptr)

		return T_gpu

	def get_n_components(self):
		return self.K

	def set_n_components(self, n_components):
		self.K = n_components


if __name__=='__main__':
	
	h = cublasCreate()	

	pca = KernelPCA(h, n_components=4)
	X = np.ones((500,10)).astype(np.float32)
	print "IN PYTHON\n\n"
	print "X[0] = ", X[0,0]
	dX = to_gpu(X)
	print "dX[0] = ", dX[0,0]
	dT = pca.fit_transform(dX)
	print dT
	cublasDestroy(h)
