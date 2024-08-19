#pragma once
#include <string.h> // memset()
//#include "algorithmparameters.h"
//#include "cameraparameters.h"
#include "managed.h"

#include <vector_types.h> // float4

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




class __align__(128) LineState : public Managed {
public:

    // float4 **foods; // Foods is the population of food sources.
    float4 *foods_1d;
    // float **fitness;//fitness is a vector holding fitness (quality) values associated with food sources.
    float *fitness_1d;
    // char **trial; //trial is a vector holding trial numbers through which solutions can not be improved.
    char *trial_1d;
    // int * numBests;
    float4 * norm4;

    //host
    float4 *h_foods_1d; 
    float *h_fitness_1d;
    char *h_trial_1d;
    float4 *h_norm4;

    int iters = 0;

    // Control Parameters of ABC algorithm
    // int NP = 40; // The number of colony size (employed bees+onlooker bees).
    int FoodNumber = 5; //The number of food sources equals the half of the colony size.
    
    int maxCycle = 3000; /*The number of cycles for foraging {a stopping criteria}*/
    int n;

    void resize(int n) //n  = 640*480
    {
        // Allocate input in host memory
        // h_foods_1d = (float4*)malloc(sizeof(float4) *n * FoodNumber);
        // h_fitness_1d = (float*)malloc(sizeof(float) *n * FoodNumber);
        // h_trial_1d = (char*)malloc(sizeof(char) *n * FoodNumber);
        // h_norm4 = (float4*)malloc(sizeof(float4) *n);
        // // Allocate vectors in device memory
        // cudaMallocManaged (&norm4,          sizeof(float4) *n);
        // cudaMallocManaged (&fitness_1d, sizeof(float) *n * FoodNumber);
        // cudaMallocManaged (&trial_1d, sizeof(char) *n * FoodNumber);
        // cudaMalloc (&foods_1d, sizeof(float4) *n * FoodNumber);
        // // Copy vectors from host memory to device memory
        // cudaMemcpy(norm4, h_norm4, sizeof(float4) *n, cudaMemcpyHostToDevice);
        // cudaMemcpy(fitness_1d, h_fitness_1d, sizeof(float) *n * FoodNumber, cudaMemcpyHostToDevice);
        // cudaMemcpy(trial_1d, h_trial_1d, sizeof(char) *n * FoodNumber, cudaMemcpyHostToDevice);
        // cudaMemcpy(foods_1d, h_foods_1d, sizeof(float4) *n * FoodNumber, cudaMemcpyHostToDevice);



        cudaMallocManaged (&norm4,          sizeof(float4) *n);
        // cudaMallocManaged (&numBests,          sizeof(int) *n);
        cudaMallocManaged (&fitness_1d, sizeof(float) *n * FoodNumber);
        cudaMallocManaged (&trial_1d, sizeof(char) *n * FoodNumber);
        cudaMalloc (&foods_1d, sizeof(float4) *n * FoodNumber);

        cudaMemset             (norm4,      0, sizeof(float4) * n);
        // cudaMemset             (numBests,      0, sizeof(int) * n);
        cudaMemset             (trial_1d, 0, sizeof(char) *n * FoodNumber);
        cudaMemset             (foods_1d, 0, sizeof(float4) *n * FoodNumber);
        cudaMemset             (fitness_1d, 0, sizeof(float) *n * FoodNumber);
    }
    void free()
    {
        // Free device memory
        cudaFree (norm4);
        // cudaFree (numBests);
        cudaFree (foods_1d);
        cudaFree (trial_1d);
        cudaFree (fitness_1d);

        // Free host memory

    }
    ~LineState()
    {
        cudaFree (norm4);
        // cudaFree (numBests);
        cudaFree (foods_1d);
        cudaFree (trial_1d);
        cudaFree (fitness_1d);
    }
};


class __align__(128) LineState_consistency : public Managed {
public:
  
   
 
    char *used_pixels; 
    float4 *suggestions; // suggestions from points that satisties strong/weak consistency check 
    // float4 *suggestions_1; // suggestions from points that satisties strong/weak consistency check 
    // float4 *norm4;
    int n;

    void resize(int n) //n  = 640*480
    {
        
        
        cudaMallocManaged (&used_pixels,                sizeof(char) * n);
        cudaMallocManaged (&suggestions,                sizeof(float4) * n);
        // cudaMallocManaged (&suggestions_1,              sizeof(float4) * n);
        // cudaMallocManaged (&norm4,                      sizeof(float4) * n);

        cudaMemset         (used_pixels,        0, sizeof(char) * n);
        cudaMemset         (suggestions,        0, sizeof(float4) * n);
        // cudaMemset         (suggestions_1,      0, sizeof(float4) * n);
        // cudaMemset         (norm4,              0, sizeof(float4) * n);
    }

    void reset(int n)
    {

        cudaMemset         (used_pixels,      0, sizeof(char) * n);
        cudaMemset         (suggestions,       0, sizeof(float4) * n);
        // cudaMemset         (suggestions_1,       0, sizeof(float4) * n);
    }

    ~LineState_consistency()
    {
        
        cudaFree (used_pixels);
        cudaFree (suggestions);
        // cudaFree (suggestions_1);
        // cudaFree (norm4);
    }
};
