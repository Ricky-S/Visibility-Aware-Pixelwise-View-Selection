#pragma once

#include "camera.h"
#include "managed.h"
#include "config.h"

class __align__(128) CameraParameters_cu : public Managed {
public:
    float f; // used only for dummy depth -> disparity conversion
    bool rectified;
    Camera_cu cameras[MAX_IMAGES];
    int idRef;
    int cols;
    int rows;
    int* viewSelectionSubset;
    int viewSelectionSubsetNumber;
    float3** vectors_camera; 

    int * selectedViewsNum; 
    int ** selectedViews; 
    int * selectedViewsNum_ambc;
    int ** selectedViews_ambc; 
    CameraParameters_cu()
    {
		rectified = true;
		idRef = 0;
        cudaMallocManaged (&viewSelectionSubset, sizeof(int) * MAX_IMAGES);
        cudaMallocManaged (&selectedViewsNum, sizeof(int) * MAX_IMAGES);
        cudaMallocManaged (&selectedViewsNum_ambc, sizeof(int) * MAX_IMAGES);
        cudaMallocManaged (&selectedViews, sizeof(int*) * MAX_IMAGES);
        cudaMallocManaged (&selectedViews_ambc, sizeof(int*) * MAX_IMAGES);
        cudaMallocManaged (&vectors_camera, sizeof(float3*) * MAX_IMAGES);
        for (int i = 0; i < MAX_IMAGES; i++)
        {
            cudaMallocManaged (&selectedViews[i], sizeof(int*) * MAX_IMAGES);
            cudaMallocManaged (&selectedViews_ambc[i], sizeof(int*) * MAX_IMAGES);
            cudaMallocManaged (&vectors_camera[i], sizeof(float3) * MAX_IMAGES);
        }
    }
    ~CameraParameters_cu()
    {
        cudaFree (viewSelectionSubset);
        cudaFree (selectedViewsNum);
        cudaFree (selectedViewsNum_ambc);
        
        for (int i = 0; i < MAX_IMAGES; i++)
        {
            cudaFree (selectedViews[i]);
            cudaFree (selectedViews_ambc[i]);
            cudaFree (vectors_camera[i]);
        }
        cudaFree (selectedViews);
        cudaFree (selectedViews_ambc);
        cudaFree (vectors_camera);
 

    }
};
