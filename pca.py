from ctypes import *
import sys
from skcuda.cublas import cublasCreate, cublasDestroy, _types
from pycuda.gpuarray import GPUArray
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


_libkernel_pca.dev_fit_transform_d.restype = POINTER(c_double) 
_libkernel_pca.dev_fit_transform_d.argtypes = [_types.handle, c_int, c_int, POINTER(c_double), c_int]

_libkernel_pca.dev_fit_transform_f.restype = POINTER(c_float) 
_libkernel_pca.dev_fit_transform_f.argtypes = [_types.handle, c_int, c_int, POINTER(c_float), c_int]

_libkernel_pca.c_strided_to_f_contiguous_f.restype = None
_libkernel_pca.c_strided_to_f_contiguous_f.argtypes = [c_int, c_int, POINTER(c_int), POINTER(c_float)]

_libkernel_pca.f_to_c_contiguous_f.restype = None
_libkernel_pca.f_to_c_contiguous_f.argtypes = [c_int, c_int, POINTER(c_int), POINTER(c_float)]

_libkernel_pca.c_strided_to_f_contiguous_d.restype = None
_libkernel_pca.c_strided_to_f_contiguous_d.argtypes = [c_int, c_int, POINTER(c_int), POINTER(c_double)]

_libkernel_pca.f_to_c_contiguous_d.restype = None
_libkernel_pca.f_to_c_contiguous_d.argtypes = [c_int, c_int, POINTER(c_int), POINTER(c_double)]


class KernelPCA:
	
	def __init__(self, n_components=-1, h):
		

		self.n_components = n_components

		self.h = h

	def fit_transform(self, X_gpu):
		
		is_c_contiguous = False
		if X_gpu.flags.c_contiguous:	
			is_c_contiguous = True

		is_float = False
		if X_gpu.dtype == 'float32' and X_gpu.dtype != 'foat64':
			is_float = True		
		else:
			raise ValueError("Array must be type float32 or float64, not '" + X_gpu.dtype + "'") 

		if len(X_gpu.shape) != 2:
			raise ValueError("PCA can only be performed on a 2D array")	
		
		M = X_gpu.shape[0]
		N = X_gpu.shape[1]

		# need the array strides to index the array internally
		strides = np.array([X_gpu.strides[0], X_gpu.strides[1]], dtype=np.intc)

		if is_c_contiguous:
			_libkernel_pca.	





	def get_n_components(self):
		return self.n_components

	def set_n_components(self, n_components):
		self.n_components = n_components


if __name__=='__main__':
	
	pca = KernelPCA()
	print 'comps: ', pca.get_n_components()

