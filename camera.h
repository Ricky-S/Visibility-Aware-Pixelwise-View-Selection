#ifndef CAMERA_H
#define CAMERA_H
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
//#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>

#include "mathUtils.h"
#include "globalstate.h"
#include "managed.h"
#include "cameraparameters.h"

using namespace std;
using namespace cv;



struct Camera {
    Camera () : P ( Mat::eye ( 3,4,CV_32F ) ),  R ( Mat::eye ( 3,3,CV_32F ) ),baseline (0.54f), reference ( false ), depthMin ( 2.0f ), depthMax ( 20.0f ) {}
    Mat_<float> P;
    Mat_<float> P_inv;
    Mat_<float> M;
    Mat_<float> M_inv;
    //Mat_<float> K;
    Mat_<float> R;
    Mat_<float> R_orig_inv;
    Mat_<float> t;
    float4 t4;
    float* R_array;
    Vec3f C;
    float baseline;
    bool reference;
    float depthMin; //this could be figured out from the bounding volume (not done right now, but that's why this parameter is here as well and not only in AlgorithmParameters)
    float depthMax; //this could be figured out from the bounding volume (not done right now, but that's why this parameter is here as well and not only in AlgorithmParameters)
    //int id; //corresponds to the image name id (eg. 0-10), independent of order in argument list, just dependent on name
    string id;
    Mat_<float> K;
    Mat_<float> K_inv;
    //float f;
};

extern vector<Camera> cameras;


class Camera_cu : public Managed {
   public:
    float* P;
    float4 P_col34;
    float* P_inv;
    float* M; 
    float* M_inv;
    float* R;
    float* R_orig_inv;
    float* R_orig;
    float4 t4_orig;
    float4 t4;
    float4 C4;
    float fx;
    float fy;
    float f;
    float alpha;
    float baseline;
    bool reference;
    float depthMin;  // this could be figured out from the bounding volume (not
                     // done right now, but that's why this parameter is here as
                     // well and not only in AlgorithmParameters)
    float depthMax;  // this could be figured out from the bounding volume (not
                     // done right now, but that's why this parameter is here as
                     // well and not only in AlgorithmParameters)
    char* id;  // corresponds to the image name id (eg. 0-10), independent of
               // order in argument list, just dependent on name
    float* K;
    float* K_inv;
    Camera_cu() {
        baseline = 0.54f;
        reference = false;
        depthMin = 2.0f;  // this could be figured out from the bounding volume
                          // (not done right now, but that's why this parameter
                          // is here as well and not only in
                          // AlgorithmParameters)
        depthMax = 20.0f;  // this could be figured out from the bounding volume
                           // (not done right now, but that's why this parameter
                           // is here as well and not only in
                           // AlgorithmParameters)

        checkCudaErrors(cudaMallocManaged(&P, sizeof(float) * 4 * 4));
        checkCudaErrors(cudaMallocManaged(&P_inv, sizeof(float) * 4 * 4));
        checkCudaErrors(cudaMallocManaged (&M, sizeof(float) * 4 * 4)); 
        checkCudaErrors(cudaMallocManaged(&M_inv, sizeof(float) * 4 * 4));
        checkCudaErrors(cudaMallocManaged(&K, sizeof(float) * 4 * 4));
        checkCudaErrors(cudaMallocManaged(&K_inv, sizeof(float) * 4 * 4));
        checkCudaErrors(cudaMallocManaged(&R, sizeof(float) * 4 * 4));
        checkCudaErrors(cudaMallocManaged(&R_orig_inv, sizeof(float) * 4 * 4));
        checkCudaErrors(cudaMallocManaged(&R_orig, sizeof(float) * 4 * 4));
        checkCudaErrors(cudaMallocManaged(&R_orig_inv, sizeof(float) * 4 * 4));
    }
    ~Camera_cu() {
        cudaFree(P);
        cudaFree(P_inv);
        cudaFree(M);
        cudaFree(M_inv);
        cudaFree(K);
        cudaFree(K_inv);
        cudaFree(R);
        cudaFree(R_orig_inv);
        cudaFree(R_orig);
    }
};




//change tex3D to Get_2D_gray
inline int Get_2D_gray(Mat_<uchar> image, float x_location, float y_location)
{
   int x = (int)x_location;
   int y = (int)y_location;
   int pixel = image.at<uchar>(y,x);

   return pixel;
}



inline Mat_<float> getTransformationMatrix ( Mat_<float> R, Mat_<float> t ) {
    Mat_<float> transMat = Mat::eye ( 4,4, CV_32F );
    //Mat_<float> Rt = - R * t;
    R.copyTo ( transMat ( Range ( 0,3 ),Range ( 0,3 ) ) );
    t.copyTo ( transMat ( Range ( 0,3 ),Range ( 3,4 ) ) );

    return transMat;
}

inline Mat_<float> getColSubMat ( Mat_<float> M, int* indices, int numCols ) {
    Mat_<float> subMat = Mat::zeros ( M.rows,numCols,CV_32F );
    for ( int i = 0; i < numCols; i++ ) {
        M.col ( indices[i] ).copyTo ( subMat.col ( i ) );
    }
    return subMat;
}

inline Mat_<float> getCameraCenter ( Mat_<float> &P ) {
    Mat_<float> C = Mat::zeros ( 4,1,CV_32F );

    Mat_<float> M = Mat::zeros ( 3,3,CV_32F );

    int xIndices[] = { 1, 2, 3 };
    int yIndices[] = { 0, 2, 3 };
    int zIndices[] = { 0, 1, 3 };
    int tIndices[] = { 0, 1, 2 };

    // x coordinate
    M = getColSubMat ( P,xIndices,sizeof ( xIndices )/sizeof ( xIndices[0] ) );
    C ( 0,0 ) = ( float )determinant ( M );

    // y coordinate
    M = getColSubMat ( P,yIndices,sizeof ( yIndices )/sizeof ( yIndices[0] ) );
    C ( 1,0 ) = - ( float )determinant ( M );

    // z coordinate
    M = getColSubMat ( P,zIndices,sizeof ( zIndices )/sizeof ( zIndices[0] ) );
    C ( 2,0 ) = ( float )determinant ( M );

    // t coordinate
    M = getColSubMat ( P,tIndices,sizeof ( tIndices )/sizeof ( tIndices[0] ) );
    C ( 3,0 ) = - ( float )determinant ( M );

    return C;
}

inline void transformCamera ( Mat_<float> R,Mat_<float> t, Mat_<float> transform, Camera &cam, Mat_<float> K ) {
    // create rotation translation matrix
    Mat_<float> transMat_original = getTransformationMatrix ( R,t );

    //transform
    Mat_<float> transMat_t = transMat_original * transform;

    // compute translated P (only consider upper 3x4 matrix)
    cam.P = K * transMat_t ( Range ( 0,3 ),Range ( 0,4 ) );
    // set R and t
    cam.R = transMat_t ( Range ( 0,3 ),Range ( 0,3 ) );
    cam.t = transMat_t ( Range ( 0,3 ),Range ( 3,4 ) );
    // set camera center C
    Mat_<float> C = getCameraCenter ( cam.P );

    C = C / C ( 3,0 );
    cam.C = Vec3f ( C ( 0,0 ),C ( 1,0 ),C ( 2,0 ) );
}

inline Mat_<float> getTransformationReferenceToOrigin ( Mat_<float> R,Mat_<float> t ) {
    // create rotation translation matrix
    Mat_<float> transMat_original = getTransformationMatrix ( R,t );

    // get transformation matrix for [R1|t1] = [I|0]
    return transMat_original.inv ();
}

#endif