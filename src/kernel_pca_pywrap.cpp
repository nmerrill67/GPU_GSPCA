#include "kernel_pca_pywrap.hpp"

void PyKernelPCA::CheckNpArray(PyArrayObject* arr)
{
	if (!(PyArray_FLAGS(arr) & NPY_ARRAY_F_CONTIGUOUS)) 
    		throw std::runtime_error("numpy array must be Fortran contiguous (column-major). Try numpy.asfortranarray.");
  	
  	if (PyArray_TYPE(arr) != NPY_FLOAT32) 
    		throw std::runtime_error("numpy array must be of type float32");
  
  	if (PyArray_NDIM(arr) != 2)
  		throw std::runtime_error("numpy array must be 2 dimensional for PCA");
}

PyKernelPCA::PyKernelPCA(int num_pcs=-1) : KernelPCA::KernelPCA(num_pcs){}
 





PyArrayObject* PyKernelPCA::fit_transform(PyArrayObject* R)
{

	


	return R;

}

void PyKernelPCA::set_n_components(int K_) 
{

	this->KernelPCA::set_n_components(K_);

}

BOOST_PYTHON_MODULE(kernel_pca) 
{

	boost::python::class_<PyKernelPCA>("KernelPCA",
		boost::python::init<int>())
		.def("fit_transform", &PyKernelPCA::fit_transform, boost::python::return_value_policy<boost::python::manage_new_object>())
		.def("set_n_components", &PyKernelPCA::set_n_components)	
	;

}



















