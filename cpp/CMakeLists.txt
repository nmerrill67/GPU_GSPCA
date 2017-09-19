cmake_minimum_required(VERSION 3.0)

project(kernel_pca)

set(CMAKE_BUILD_TYPE RELEASE)


if (UNIX)
    set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
else ()
    set(CMAKE_CXX_FLAGS "/c /Zi /GS- /GL /W3 /Gy /Gm- /Ox /EHsc /W3 /O2 /MD")
    set(CMAKE_EXE_LINKER_FLAGS "/LTCG /RELEASE")
endif ()



# CUDA PACKAGE
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)


SET(CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")

# COMPILE CU FILES
file(GLOB CUDA_FILES "src/" *.cu)

set(CUDA_ARCH "sm_50")



# SETUP FOR CPP FILES
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
cuda_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CUDA_INCLUDE_DIRECTORIES}
)
include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CUDA_INCLUDE_DIRECTORIES}
)
cuda_add_library(kernel_pca
	${CUDA_FILES}
)


CUDA_ADD_CUBLAS_TO_TARGET(kernel_pca ${CUDA_FILES})

CUDA_COMPILE(kernel_pca ${CUDA_FILES})

target_link_libraries(kernel_pca
	cuda
)

### Example compilation ################

find_package(GSL REQUIRED)

# COMPILE AND LINK
add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

include_directories(
	${GSL_INCLUDE_DIRS}
)

target_link_libraries(main 
	kernel_pca
	${GSL_LIBRARIES}
)




