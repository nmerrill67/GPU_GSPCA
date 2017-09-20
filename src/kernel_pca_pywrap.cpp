#include <boost/python.hpp>
#include "kernel_pca.h"

BOOST_PYTHON_MODULE(kernel_pca) 
{

	boost::python::class_<KernelPCA>("KernelPCA",
		init<int>())
		.def("fit_transform", &KernelPCA::fit_transform)
		.add_property("K", &KernelPCA::set_n_components)	
	;

}



















