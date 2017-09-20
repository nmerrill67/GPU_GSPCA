#include <boost/python.hpp>
#include "kernel_pca.h"

BOOST_PYTHON_MODULE(kernel_pca) 
{

	boost::python::class_<KernelPCA>("KernelPCA",
		boost::python::init<int>())
		.def("fit_transform", &KernelPCA::fit_transform)
		.def("set_n_components", &KernelPCA::set_n_components)	
	;

}



















