#include "kernel_pca_pywrap.hpp"



void PyKernelPCA::CheckNpArray(PyArrayObject* arr)
{
	/*
	if (!(PyArray_FLAGS(arr) & NPY_ARRAY_F_CONTIGUOUS)) 
    		throw std::runtime_error("numpy array must be Fortran contiguous (column-major). Try numpy.asfortranarray.");
  	*/
  	if (PyArray_TYPE(arr) != NPY_FLOAT32) 
    		throw std::runtime_error("numpy array must be of type float32");
  
  	if (PyArray_NDIM(arr) != 2)
  		throw std::runtime_error("numpy array must be 2 dimensional for PCA");
}

PyKernelPCA::PyKernelPCA(int n_components) : KernelPCA::KernelPCA(n_components){}
 

PyArrayObject* PyKernelPCA::fit_transform(PyArrayObject* R_)
{

	
	CheckNpArray(R_);

	float* R; // C array from numpy array
	R = (float*)PyArray_GetPtr(R_, (npy_intp*) 0); // This macro (or is it a function?) returns a void*, so just cast it to float* inplace

	int M, N;
	M = PyArray_DIMS(R_)[0]; // first dimension of array	
	N = PyArray_DIMS(R_)[1]; 

	float* T;
		
	T = KernelPCA::fit_transform(M, N, R); // run fit_transform on the raw float data, and put it in a float array	

	

	int dims[2] = {M, KernelPCA::get_n_components()};
	
	return (PyArrayObject*)PyArray_SimpleNewFromData(2, (npy_intp*)dims, NPY_FLOAT32, (void*)T);

}


boost::shared_ptr<PyKernelPCA> initWrapper(int n_components=-1)
{

	boost::shared_ptr<PyKernelPCA> ptr( new PyKernelPCA(n_components) );

	return ptr;

}


// Use boosts' macro to make the python module "kernel_pca_pywrap"
// This wraps the PyKernelPCA class, which extends the KernelPCA class to be able to accept numpy arrays as input, and return numpy arrays from fit_transform
BOOST_PYTHON_MODULE(py_kernel_pca) 
{

	boost::python::class_< PyKernelPCA, boost::shared_ptr< PyKernelPCA > >("KernelPCA",
		boost::python::no_init)
		.def("__init__", boost::python::make_constructor(&initWrapper))
		.def("fit_transform", &PyKernelPCA::fit_transform, boost::python::return_value_policy<boost::python::manage_new_object>())
		.def("set_n_components", &KernelPCA::set_n_components)	
	;

}



















