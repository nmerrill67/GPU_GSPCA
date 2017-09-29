#include <Python.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>

#include "kernel_pca.h"




// Python wrapper class for KernelPCA
class PyKernelPCA : public KernelPCA 
{

private:

bool CheckNpArray(PyObject* arr);


public:


PyKernelPCA(int n_components);



// overload KernelPCA::fit_transform
PyObject* fit_transform(PyObject* R, bool verbose);



};



boost::shared_ptr<PyKernelPCA> initWrapper(int n_components);











