#include <Python.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>

#include "kernel_pca.h"




// Python wrapper class for KernelPCA
class PyKernelPCA : public KernelPCA 
{

private:

bool CheckNpArray(PyObject* arr);

// Copy a c-contiguous strided numpy array to a fortran-contiguous float or double array
float* c_cont_npy_to_f_cont_float_ptr(int M, int N, PyObject* R_);
double* c_cont_npy_to_f_cont_double_ptr(int M, int N, PyObject* R_);

public:


PyKernelPCA(int n_components);



// overload KernelPCA::fit_transform
PyObject* fit_transform(PyObject* R, bool verbose);

};


boost::shared_ptr<PyKernelPCA> initWrapper(int n_components);











