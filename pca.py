from ctypes import *
import sys
from skcuda.cublas import cublasCreate, cublasDestroy, _types
from pycuda.gpuarray import GPUArray






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
_libkernel_pca.dev_fit_transform_d.argtypes = [_types.handle, c_int, c_int, POINTER(c_float), c_int]

class KernelPCA:
	
	def __init__(self, n_components=-1):
		

		self.n_components = n_components

		self.h = cublasCreate()

	def __del__(self):

		cublasDestroy(self.h)

	#def fit_transform(self, X_gpu):
		
		

	def get_n_components(self):
		return self.n_components

	def set_n_components(self, n_components):
		self.n_components = n_components


if __name__=='__main__':
	
	pca = KernelPCA()
	print 'comps: ', pca.get_n_components()


