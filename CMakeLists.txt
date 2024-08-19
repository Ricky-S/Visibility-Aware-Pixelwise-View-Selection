cmake_minimum_required (VERSION 2.8)
project (MVS)

find_package(CUDA 6.0 REQUIRED ) # For Cuda Managed Memory and c++11
find_package(OpenCV REQUIRED )
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(.)

# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_30,code=sm_30 -gencode arch=compute_52,code=sm_52)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_86,code=sm_86 -gencode arch=compute_75,code=sm_75)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_86,code=sm_86 -gencode arch=compute_75,code=sm_75)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_86,code=sm_86)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_86,code=sm_86)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_61,code=sm_61)
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++11)
    add_definitions(-pthread)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    # add_definitions(-g)
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native") # extend release-profile with fast-math
    # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ffast-math -march=native") # extend release-profile with fast-math
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    MVS
    cameraGeometryUtils.h
    camera.h
    mathUtils.h
    globalstate.h
    linestate.h
    cameraparameters.h
    algorithmparameters.h
    displayUtils.h
    fileIoUtils.h
    point_cloud.h
    point_cloud_list.h
    main.h
    vector_operations.h
    ambc.cu
    consistency.cu
    main.cpp
    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(MVS
    ${OpenCV_LIBS}
    )
