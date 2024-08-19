#ifndef MAIN_H
#define MAIN_H
#include <fstream>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "vector_types.h"
#include "config.h"
#include "globalstate.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>

#include "camera.h"
#include "mathUtils.h"
#include "point_cloud_list.h"

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

using namespace std;
using namespace cv;

    

//pathes to input images (camera images, ground truth, ...)
struct InputFiles {
    string results_folder; 
    string gt_filename; // ground truth image
    string gt_nocc_filename; // non-occluded ground truth image (as provided e.g. by Kitti)
    string occ_filename; // occlusion mask (binary map of all the points that are occluded) (as provided e.g. by Middleburry)
    string gt_normal_filename; // ground truth normal map (for Strecha)
    string calib_filename; // calibration file containing camera matrices (P) (as provided e.g. by Kitti)
    string images_folder; // path to camera input images
    string p_folder; // path to camera projection matrix P (Strecha)
    string in_ex_folder;
    string camera_folder; // path to camera calibration matrix K (Strecha)
    string krt_file; // path to camera matrixes in middlebury format
    string bounding_folder; //path to bounding volume (Strecha)
    string seed_file; // path to bounding volume (Strecha)
    string pmvs_folder; // path to pmvs folder
    int num_images; // number of images
    InputFiles () : gt_filename ( "" ), gt_nocc_filename ( "" ), occ_filename ( "" ), gt_normal_filename ( "" ), calib_filename ( "" ), images_folder ( "" ), p_folder ( "" ), camera_folder ( "" ),krt_file(""), pmvs_folder("") {}
    vector<string> img_filenames; // input camera images (only filenames, path is set in images_folder), names can also be used for calibration data (e.g. for Strecha P, camera)

};

//pathes to output files
struct OutputFiles {
    OutputFiles () : parentFolder ( "results" ), disparity_filename ( 0 ) {}
    const char* parentFolder;
    char* disparity_filename;
};

//parameters for camera geometry setup (assuming that K1 = K2 = K, P1 = K [I | 0] and P2 = K [R | t])
struct CameraParameters {
    CameraParameters () : rectified ( false ), idRef ( 0 ) {}
    Mat_<float> K; //if K varies from camera to camera: K and f need to be stored within Camera
    Mat_<float> K_inv; //if K varies from camera to camera: K and f need to be stored within Camera
    
    float f;
    bool rectified;
    vector<Camera> cameras;
    int idRef;
    vector<int> viewSelectionSubset;
    vector<vector <int>> selectedViews;
};

int runcuda(GlobalState &gs);
// int runconsistency(GlobalState &gs, PointCloudList &pc_list, int num_views, int i);
int runconsistency(GlobalState &gs, PointCloudList &pc_list, int num_views);
// class ABC 
// {
// public:
//     ABC();
//     ~ABC();

//     //void addKernel(int * c, const int *a, const int *b);
//     // int runKernel();
//     // void readImages();
//     // int runTexture();
//     void getHomography_cu ( const Camera_cu &from, const Camera_cu &to,float * __restrict__ K1_inv, const float * __restrict__ K2,const float4 &n, const float &d, float * __restrict__ H );
//     void getCorrespondingPoint_cu ( const int2 &p, const float * __restrict__ H, float4 * __restrict__ ptf );
//     float l1_norm(float f);
//     float l1_norm(float4 f);
//     void ambc(GlobalState &gs);

// private:
//     // cudaTextureObjects texture_objects_host;
//     // cudaTextureObjects *texture_objects_cuda;
//     // std::vector<cv::Mat> images;
//     // cudaArray *cuArray[256];
// };

#endif