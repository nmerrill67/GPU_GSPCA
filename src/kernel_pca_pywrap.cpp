#include "kernel_pca_pywrap.hpp"

void CheckContiguousArray(PyArrayObject* arr)
{
  if (!(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS)) {
    throw std::runtime_error("nummpy array must be C contiguous");
  }
  if (PyArray_TYPE(arr) != NPY_FLOAT32) {
    throw std::runtime_error("numpy array must be float32");
  }
 
}

PyKernelPCA::PyKernelPCA(int num_pcs=-1) : KernelPCA::KernelPCA(num_pcs){}
 





PyArrayObject* PyKernelPCA::fit_transform(PyArrayObject* R)
{

	return R;

}
 

BOOST_PYTHON_MODULE(kernel_pca) 
{

	boost::python::class_<PyKernelPCA>("KernelPCA",
		boost::python::init<int>())
		.def("fit_transform", &PyKernelPCA::fit_transform)
		.def("set_n_components", &KernelPCA::set_n_components)	
	;

}



















