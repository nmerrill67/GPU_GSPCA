cmake_minimum_required(VERSION 3.0)

project(kernel_pca C)

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE RELEASE)
	
endif()

message("BUID TYPE = ${CMAKE_BUILD_TYPE}")


# colors, because why not
string(ASCII 27 Esc)
set(RED "${Esc}[31m")
set(GREEN "${Esc}[32m")
set(ENDL "${Esc}[m")

# CUDA PACKAGE
find_package(CUDA REQUIRED)
# nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--compiler-options;-Wall;--compiler-options;-fPIC)
SET(CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")



cuda_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
i	${CUDA_INCLUDE_DIRS}
)

cuda_add_library(kernel_pca SHARED
	${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_pca.cu
)

set_target_properties(kernel_pca PROPERTIES LINKER_LANGUAGE C ) 

CUDA_ADD_CUBLAS_TO_TARGET(kernel_pca ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_pca.cu)

CUDA_COMPILE(kernel_pca ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_pca.cu)

target_link_libraries(kernel_pca ${CUDA_LIBRARIES})





	
#### Tests #######################################

option(BUILD_TESTS "Build python tests (if python wrappers are built)" OFF)

if (BUILD_TESTS)
	
	message("${GREEN} Python tests built ${ENDL}")
		
	
        add_custom_target(pytest COMMAND cd ../test && python -m unittest -v py_kernel_pca_test.TestKernelPCA WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
	
endif ()
























