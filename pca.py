import ctypes
import ctypes.util
import sys


class KernelPCA:
	
	def __init__(self, n_components=-1):
		
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
			_libkernel_pca = ctypes.cdll.LoadLibrary(_libkernel_pca_libname)

		except OSError:
			raise OSError('kernel_pca library not found')


		_libkernel_pca.initFun.restype = 

		self._KernelPCA = _libkernel_pca.initFun(n_components)



if __name__=='__main__':
	
	pca = KernelPCA()



