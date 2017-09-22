#include "kernel_pca_pywrap.hpp"
#include <iostream>



// indexing for c contiguous arrays. This is only used if  a numpy array is c contiguous, then it needs to be converted to fortran contiguous for KernelPCA.
#define ind_c(m, n, num_cols) (((m) * (num_cols) + (n)))



void PyKernelPCA::CheckNpArray(PyObject* arr)
{

	if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject*)arr))
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

PyKernelPCA::PyKernelPCA(int n_components) : KernelPCA::KernelPCA(n_components){        std::cout << "n_comp " << this->KernelPCA::get_n_components() << std::endl;
}
 

PyObject* PyKernelPCA::fit_transform(PyObject* R_)
{


	CheckNpArray(R_); // check: is the input array c-coontiguous, is it float32 type and also is it 2D


	std::cout << "R_ ref cnt = " << R_->ob_refcnt << "\n";


	int M, N;
	M = PyArray_DIMS(R_)[0]; // first dimension of array	
	N = PyArray_DIMS(R_)[1]; 

	float* R; // C array from numpy array
	
        R = (float*)malloc(M*N * sizeof(R[0]));

	npy_intp* strides  = PyArray_STRIDES(R_); // strides for data gaps

	char* R_data = (char*)PyArray_DATA(R_);

	// switch to fortran contiguous for KernelPCA, and at the same time switch to a c array from the PyObject 
	for (int m = 0; m < M; m++)
	{
		for (int n = 0; n < N; n++)
		{
			R[ind_f(m,n,M)] = *(float*)&R_data[ m*strides[0] + n*strides[1] ];
		}	
	}	

	Py_DECREF(R_);
	std::cout << "R_ ref cnt = " << R_->ob_refcnt << "\n";



	float* T;
		
	T = KernelPCA::fit_transform(M, N, R); // run fit_transform on the raw float data, and put it in a float array	

	std::cout << "n_comp " << this->KernelPCA::get_n_components() << std::endl;

	
	int dims[2] = {M, KernelPCA::get_n_components()};


	// SimpleNewFromData can only handle a c-contiguous array, so convert T to c contiguous
	float *T_ret; // there is no way to do this without making a copy (I think)
	T_ret = (float*)malloc(M*N * sizeof(T_ret[0]));

	// switch back to C contiguous for numpy
	for (int m = 0; m < M; m++)
	{
		for (int n = 0; n < N; n++)
		{

			T_ret[ind_c(m, n, N)] = T[ind_f(m, n, M)];

		}	
	}
	
	free(T);
 
	
	return PyArray_SimpleNewFromData(2, (npy_intp*)dims, NPY_FLOAT32, (void*)T_ret);

}


boost::shared_ptr<PyKernelPCA> initWrapper(int n_components)
{
	
	// if (!n_components) n_components = new int(-1);

	boost::shared_ptr<PyKernelPCA> ptr( new PyKernelPCA(n_components) );

	return ptr;

}


// Use boosts' macro to make the python module "kernel_pca_pywrap"
// This wraps the PyKernelPCA class, which extends the KernelPCA class to be able to accept numpy arrays as input, and return numpy arrays from fit_transform
BOOST_PYTHON_MODULE(py_kernel_pca) 
{

	boost::python::class_< PyKernelPCA, boost::shared_ptr< PyKernelPCA >, boost::noncopyable>("KernelPCA",
		boost::python::no_init)
		.def("__init__", boost::python::make_constructor(&initWrapper))
		.def("fit_transform", &PyKernelPCA::fit_transform, boost::python::return_value_policy<boost::python::manage_new_object>())
		.def("set_n_components", &KernelPCA::set_n_components)	
	;

}



















