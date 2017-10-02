cmake_minimum_required(VERSION 3.0)

project(kernel_pca)

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE RELEASE)
	
endif()

message("BUID TYPE = ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -fPIC")

# colors, because why not
string(ASCII 27 Esc)
set(RED "${Esc}[31m")
set(GREEN "${Esc}[32m")
set(ENDL "${Esc}[m")

# CUDA PACKAGE
find_package(CUDA REQUIRED)
# nvcc flags
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS ON)

SET(CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")


# Progress bar

find_package(Curses REQUIRED) # progressbar is included in kernel_pca.cu, so it it needs to compile

include_directories(	
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(progressbar SHARED
	 ${CMAKE_CURRENT_SOURCE_DIR}/src/progressbar.c
)

target_link_libraries(progressbar ${CURSES_LIBRARIES})

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
	progressbar
)



### Example compilation ################

find_package(GSL QUIET)

if (GSL_FOUND)

	message("\n\n\n ${GREEN} GSL is installed. Building c++ demo ${ENDL} \n\n\n")

	# COMPILE AND LINK
	add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

	include_directories(
		${GSL_INCLUDE_DIRS}
	)

	target_link_libraries(main 
		kernel_pca
		${GSL_LIBRARIES}
	)
else ()

	message( "\n\n\n ${RED} GSL not installed. C++ demo not being built ${ENDL}\n\n\n ")

endif ()




#### Python wrapper ############################

set(PYTHON_BUILT 0)

find_package( PythonLibs 2.7 QUIET)

if (PYTHONLIBS_FOUND)

	set(PYTHON_BUILT 1)

        message("\n\n\n ${GREEN} Python 2.7 is installed. Building python wrappers ${ENDL} \n\n\n")


	find_package( Boost COMPONENTS python REQUIRED )


	execute_process( COMMAND export CPLUS_INCLUDE_PATH="${PYTHON_INCLUDE_DIRS}" )

	include_directories(
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
else ()

        message( "\n\n\n ${RED} Python 2.7 is not installed. Python wrappers will not be built! ${ENDL}\n\n\n ")


endif ()

	
#### Tests #######################################

option(BUILD_TESTS "Build gtest module and python tests (if python wrappers are built)" ON)

if (BUILD_TESTS)

	message( "\n\n\n ${GREEN} Building Tests \n\n\n")

	include(ExternalProject)

	set(GTEST_FORCE_SHARED_CRT ON)
	set(GTEST_DISABLE_PTHREADS OFF)

	ExternalProject_Add(googletest
	    GIT_REPOSITORY https://github.com/google/googletest.git
	    CMAKE_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
	    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
	    -Dgtest_force_shared_crt=${GTEST_FORCE_SHARED_CRT}
	    -Dgtest_disable_pthreads=${GTEST_DISABLE_PTHREADS}
	    -DBUILD_GTEST=ON
	    PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
	    # Disable install step
	    INSTALL_COMMAND ""
	)

	# Specify include dir
	ExternalProject_Get_Property(googletest source_dir)
	set(GTEST_INCLUDE_DIRS ${source_dir}/googletest/include)

	# Specify MainTest's link libraries
	ExternalProject_Get_Property(googletest binary_dir)
	set(GTEST_LIBS_DIR ${binary_dir}/googlemock/gtest)



	enable_testing()

	include_directories(${GTEST_INCLUDE_DIRS})	

	set(PROJECT_TEST_NAME ${PROJECT_NAME}_test)

	file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/test/*.cpp)
	add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})
	add_dependencies(${PROJECT_TEST_NAME} googletest)

	target_link_libraries(${PROJECT_TEST_NAME}
                kernel_pca
                ${GSL_LIBRARIES}
        	${GTEST_LIBS_DIR}/libgtest.a
        	${GTEST_LIBS_DIR}/libgtest_main.a
        )

	target_link_libraries(${PROJECT_TEST_NAME} ${CMAKE_THREAD_LIBS_INIT})


	add_test(kernel_pca_test ${PROJECT_TEST_NAME})	

	if(PYTHON_BUILT)
	
		message("${GREEN} Python tests built ${ENDL}")
		
	
                add_custom_target(tests COMMAND make test && cd ../test && python -m unittest -v py_kernel_pca_test.TestKernelPCA WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
	
	else()

		
                add_custom_target(tests COMMAND make test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

	endif()

endif ()
























