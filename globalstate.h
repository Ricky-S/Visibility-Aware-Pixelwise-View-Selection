#pragma once

#include "algorithmparameters.h"
#include "cameraparameters.h"
//#include "camera.h"
#include "linestate.h"
#include "managed.h"
#include "point_cloud.h"

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <cuda_texture_types.h>

// includes, cuda
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

class GlobalState : public Managed {
public:
    CameraParameters_cu *cameras;
    CameraParameters_cu *cameras_orig;
    //Camera_cu *cameras;
    AlgorithmParameters *params;
    LineState *lines;
    LineState_consistency *lines_consistency;
    curandState *cs;
    float4 ** foods_all;
    float ** fitness_all;
    int max_subset_num = 1;
    int rows;
    int cols;
    int limit = 8;  //A food source which could not be improved through "limit" trials is abandoned by its employed bee.
    float lower_bound = -1.0; //lower bound of the parameters.
    float upper_bound = 1.0; // upper bound of the parameters.can be defined as arrays for the problems of which parameters have different bounds. 
    float depth_lower_bound = 0.3;
    float depth_upper_bound = 1.2;
    int current_image = 0;
    int cycles = 0;
    //int iters = 0;
    int num_images = 0;

    PointCloud *pc;
    // int box_hsize = 11;
    // int box_vsize = 11;
    // int image_number = 32;
    //float* GlobalMin;
    // vector<Camera> cameras;
    //AlgorithmParameters *params;
    
    // Variables used for consistency checks
    float4 *norm4_1d_consistency_ref;
    float4 **norm4_1d_consistency_src;

    cudaTextureObject_t imgs  [MAX_IMAGES];
    cudaTextureObject_t normals_depths  [MAX_IMAGES]; // first 3 values normal, fourth depth
    cudaArray *cuArray[MAX_IMAGES];
    float *dataArray[MAX_IMAGES];
    float *dataArray2[MAX_IMAGES];


    void resize(int n)
    {
        printf("Resizing globalstate to %d\n", n);
        cudaMallocManaged (&lines_consistency,     sizeof(LineState_consistency) * (n));//n+1
    }
    void gs_free_lines_consistency()
    {
        printf("cudafree");
        cudaFree (&lines_consistency);
    }
    //cudaTextureObject_t gradx [MAX_IMAGES];
    //cudaTextureObject_t grady [MAX_IMAGES];
    GlobalState() {
        //printf("GlobalState constructor\n");
        cameras = new CameraParameters_cu;
        cameras_orig = new CameraParameters_cu;
        lines = new LineState;
        // lines = (LineState *)malloc(sizeof(LineState));
        // lines_consistency = new LineState_consistency;
    }
    ~GlobalState() {
        //printf("GlobalState destructor\n");
        delete cameras;
        delete lines;
        delete cameras_orig;
        // delete lines_consistency;
    }

};
