#include <Python.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>

#include "kernel_pca.h"


// Python wrapper class for KernelPCA
class PyKernelPCA : public KernelPCA 
{

private:

void CheckNpArray(PyArrayObject* arr);


public:


PyKernelPCA(int n_components);

// overload KernelPCA::fit_transform
PyArrayObject* fit_transform(PyArrayObject* R);

// get access to set_n_components
void set_n_components(int K_);

int get_n_components();




};













