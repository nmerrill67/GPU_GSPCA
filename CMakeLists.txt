cmake_minimum_required(VERSION 3.0)

project(kernel_pca)

set(CMAKE_BUILD_TYPE RELEASE)


set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -fPIC")



# CUDA PACKAGE
find_package(CUDA REQUIRED)
# nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS ON)

SET(CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")




cuda_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CUDA_INCLUDE_DIRECTORIES}
)

cuda_add_library(kernel_pca SHARED
	${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_pca.cu
)


CUDA_ADD_CUBLAS_TO_TARGET(kernel_pca ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_pca.cu)

CUDA_COMPILE(kernel_pca ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_pca.cu)

target_link_libraries(kernel_pca
	cuda
)



### Example compilation ################

find_package(GSL REQUIRED)

# COMPILE AND LINK
add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

include_directories(
	${GSL_INCLUDE_DIRS}
)

target_link_libraries(main 
	kernel_pca
	${GSL_LIBRARIES}
)





#### Python wrapper ############################

find_package( PythonLibs 2.7 REQUIRED )

find_package( Boost COMPONENTS python REQUIRED )

message("${PYTHON_INCLUDE_DIRS}")

execute_process( COMMAND export CPLUS_INCLUDE_PATH="${PYTHON_INCLUDE_DIRS}" )

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${PYTHON_INCLUDE_DIRS}
	${Boost_INCLUDE_DIR}
)

add_library(py_kernel_pca SHARED
	${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_pca_pywrap.cpp
)



target_link_libraries(py_kernel_pca
	kernel_pca
	${Python_LIBRARIES}
	${Boost_LIBRARIES}
)

# don't prepend wrapper library name with lib
set_target_properties( py_kernel_pca PROPERTIES PREFIX "" )












