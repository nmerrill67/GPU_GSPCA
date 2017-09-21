#include <boost/python.hpp>
#include <numpy/arrayobject.h>

#include "kernel_pca.h"


// Python wrapper class for KernelPCA
class PyKernelPCA : public KernelPCA 
{

public:


PyKernelPCA(int num_pcs); 

// overload KernelPCA::fit_transform
PyArrayObject* fit_transform(PyArrayObject* R);

};













