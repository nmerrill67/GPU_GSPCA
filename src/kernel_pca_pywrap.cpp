#include "kernel_pca_pywrap.hpp"
#include <iostream>



// indexing for c contiguous arrays. This is only used if  a numpy array is c contiguous, then it needs to be converted to fortran contiguous for KernelPCA.
#define ind_c(m, n, num_cols) (((m) * (num_cols)) + (n))

void PyKernelPCA::CheckNpArray(PyObject* arr)
{

	if (!PyArray_IS_C_CONTIGUOUS(reinterpret_cast<PyArrayObject*>(arr)))
	{
		throw std::runtime_error("array must be C contiguous (did you use numpy.array.T?)");
	}
  	if (PyArray_TYPE(arr) != NPY_FLOAT32) 
	{
    		throw std::runtime_error("numpy array must be of type float32");
	}
 	
  	if (PyArray_NDIM(arr) != 2)
	{
  		throw std::runtime_error("numpy array must be 2 dimensional for PCA");
	}
	
}

PyKernelPCA::PyKernelPCA(int n_components) : KernelPCA::KernelPCA(n_components){}
 

PyObject* PyKernelPCA::fit_transform(PyObject* R_, bool verbose=0)
{


	CheckNpArray(R_); // check: is the input array c-coontiguous, is it float32 type and also is it 2D



	int M, N;
	M = PyArray_DIMS(R_)[0]; // first dimension of array	
	N = PyArray_DIMS(R_)[1]; 


	float* R; // C array from numpy array
	
        R = (float*)malloc(M*N * sizeof(R[0]));
	if (R == 0)
	{
		throw std::runtime_error("Cannot allocate memory for C array R");
	}
	

	npy_intp* strides  = PyArray_STRIDES(R_); // strides for data gaps
	int s0, s1;
	s0 = strides[0]; s1 = strides[1];


	char* R_data = (char*)PyArray_DATA(R_);

	

	// switch to fortran contiguous for KernelPCA, and at the same time switch to a c array from the PyObject 
	for (int m = 0; m < M; m++)
	{
		for (int n = 0; n < N; n++)
		{
			R[ind_f(m,n,M)] = *(float*)&R_data[ m*s0 + n*s1 ];
		}	
	}	


	float* T;

	T = KernelPCA::fit_transform(M, N, R, verbose); // run fit_transform on the raw float data, and put it in a float array	

	
	int K, m, k;

	K =  KernelPCA::get_n_components();

	// SimpleNewFromData can only handle a c-contiguous array, so convert T to c contiguous

	float* T_ret;
	T_ret = (float*)malloc(M*K * sizeof(T_ret[0]));

	// switch back to C contiguous for numpy
	for (m = 0; m < M; m++)
	{
		for (k = 0; k < K; k++)
		{

			T_ret[ind_c(m,k,K)] = T[ind_f(m, k, M)];

		}	
	}
	
	free(T);
 	
	
	npy_intp dims[2] = {M,K};


	PyObject* T_PyArr;
 	T_PyArr = PyArray_SimpleNewFromData(2 /* = number of array dims */, dims, NPY_FLOAT32, reinterpret_cast<void*>(T_ret));
	

	return T_PyArr;

}


boost::shared_ptr<PyKernelPCA> initWrapper(int n_components)
{

	if (n_components < -1 || n_components == 0) throw std::runtime_error("Invalid n_components. Must be 0 < n_components < min matrix dimension, or n_components=-1 to return all components");
	

	boost::shared_ptr<PyKernelPCA> ptr( new PyKernelPCA(n_components) );

	return ptr;

}


#if PY_VERSION_HEX >= 0x03000000
void *
#else
void
#endif
initialize()
{
  import_array();
}

// Use boosts' macro to make the python module "kernel_pca_pywrap"
// This wraps the PyKernelPCA class, which extends the KernelPCA class to be able to accept numpy arrays as input, and return numpy arrays from fit_transform
BOOST_PYTHON_MODULE(py_kernel_pca) 
{

	initialize();

	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

	boost::python::class_< PyKernelPCA, boost::shared_ptr< PyKernelPCA >, boost::noncopyable>("KernelPCA",
		boost::python::no_init)
		.def("__init__", boost::python::make_constructor(&initWrapper))
		.def("fit_transform", &PyKernelPCA::fit_transform, (boost::python::arg("verbose")=0) )
		.def("get_n_components", &KernelPCA::get_n_components)	
		.def("set_n_components", &KernelPCA::set_n_components)	
	;


}



















