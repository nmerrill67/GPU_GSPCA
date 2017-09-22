#include "kernel_pca_pywrap.hpp"
#include <iostream>


void PyKernelPCA::CheckNpArray(PyObject* arr)
{
	
	std::cout << "Before first case\n";
	if ( !PyArray_IS_F_CONTIGUOUS(arr) ) 
	{
		std::cout << "here 1\n";
		Py_INCREF(((PyArrayObject*)arr)->descr); // CastToType decreases the ref count for arr, and we dont want that
		
		std::cout << "here 2\n";
		PyObject *tmp;
    		tmp = PyArray_CastToType((PyArrayObject*)arr, ((PyArrayObject*)arr)->descr, 1); // cast the array to fortran contiguous 
	
		std::cout << "here 3\n";
		arr = tmp;


	}
	std::cout << "Passed first case\n";
  	if (PyArray_TYPE(arr) != NPY_FLOAT32) 
	{
    		throw std::runtime_error("numpy array must be of type float32");
	}
 	
	std::cout << "Passed 2nd case\n";
  	if (PyArray_NDIM(arr) != 2)
	{
  		throw std::runtime_error("numpy array must be 2 dimensional for PCA");
	}
	
	std::cout << "Passed third case\n";
}

PyKernelPCA::PyKernelPCA(int n_components) : KernelPCA::KernelPCA(n_components){}
 

PyObject* PyKernelPCA::fit_transform(PyObject* R_)
{

	
	CheckNpArray(R_);

	std::cout << "R_ is f contiguous: " << PyArray_IS_F_CONTIGUOUS(R_);

	float* R; // C array from numpy array
	R = (float*)PyArray_GetPtr((PyArrayObject*)R_, (npy_intp*) 0); // This macro (or is it a function?) returns a void*, so just cast it to float* inplace

	int M, N;
	M = PyArray_DIMS(R_)[0]; // first dimension of array	
	N = PyArray_DIMS(R_)[1]; 

	float* T;
		
	T = KernelPCA::fit_transform(M, N, R); // run fit_transform on the raw float data, and put it in a float array	

	
	int dims[2] = {M, KernelPCA::get_n_components()};

	
	return PyArray_SimpleNewFromData(2, (npy_intp*)dims, NPY_FLOAT32, (void*)T);

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



















