#include "main.h"

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <vector_types.h>
#include <algorithm>
// CUDA helper functions
//#include "helper_cuda.h"         // helper functions for CUDA error check

#include "globalstate.h"
#include "algorithmparameters.h"
#include "vector_operations.h"


#include <sys/time.h>
#include <time.h>


#define idx(i,j,lda) ( (j) + ((i)*(lda)) )
#define MAXCOST 431.f // 1000.0f
// #define SHARED

#ifndef SHARED_HARDCODED
__managed__ int SHARED_SIZE_W_m;
__constant__ int SHARED_SIZE_W;
__managed__ int SHARED_SIZE_H;
__managed__ int SHARED_SIZE = 0;
__managed__ int WIN_RADIUS_W;
__managed__ int WIN_RADIUS_H;
__managed__ int TILE_W;
__managed__ int TILE_H;
#endif

__device__ void getViewVector_cu(float4 *__restrict__ v, const Camera_cu &camera, const int2 &p);
__device__ FORCEINLINE_GIPUMA static void vecOnHemisphere_cu(float4 *__restrict__ v, const float4 &viewVector);

/**
 * @brief Get the time object
 *
 * @return double
 */
double get_time(void)
{
    struct timeval tv;
    double t;
    gettimeofday(&tv, (struct timezone *)0);
    t = tv.tv_sec + (float)tv.tv_usec * 1e-6;
    return t;
}



__device__ void NormalizeVec3 (float4 *vec)
{
    const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
    const float inverse_sqrt = rsqrtf (normSquared);
    vec->x *= inverse_sqrt;
    vec->y *= inverse_sqrt;
    vec->z *= inverse_sqrt;
}


__device__ FORCEINLINE_GIPUMA static float getD_cu(const float4 &normal,
                                                   const int2 &p,
                                                   const float &depth,
                                                    const Camera_cu &cam)
{
    /*float4 pt;*/
    /*get3Dpoint_cu ( &pt, cam, (float)x0, (float)y0, depth );*/
    float4 pt, ptX;
    pt.x = depth * (float)(p.x) - cam.P_col34.x;
    pt.y = depth * (float)(p.y) - cam.P_col34.y;
    pt.z = depth - cam.P_col34.z;

    matvecmul4(cam.M_inv, pt, (&ptX));

    return -(dot4(normal, ptX));
    /*return getPlaneDistance_cu (normal, ptX);*/
}

__device__ void getHomography_cu(const Camera_cu &from, const Camera_cu &to,
                                 float *__restrict__ K1_inv, const float *__restrict__ K2,
                                 const float4 &n, const float &d, float *__restrict__ H)
{
    float tmp[16];
    outer_product4(to.t4, n, H); // tmp = t * n
    matdivide(H, d);             // tmp / d
    matmatsub2(to.R, H);         // tmp = R - tmp;
    matmul_cu(H, K1_inv, tmp);   // tmp2=tmp*Kinv
    matmul_cu(K2, tmp, H);       // H = tmp * K2

    return;
}



__device__ float2 ComputeCorrespondingPoint(const float *H, const int2 p)
{
    float3 pt;
    pt.x = H[0] * p.x + H[1] * p.y + H[2];
    pt.y = H[3] * p.x + H[4] * p.y + H[5];
    pt.z = H[6] * p.x + H[7] * p.y + H[8];
    return make_float2(pt.x / pt.z, pt.y / pt.z);
}

__device__ float l1_norm(float4 fa)
{
    return (fabsf(fa.x) +
            fabsf(fa.y) +
            fabsf(fa.z)) *
           0.3333333f;
}

__device__ float ComputeBilateralWeight(const float x_dist, const float y_dist, const float pix, const float center_pix, const float sigma_spatial, const float sigma_color)
{
    const float spatial_dist = sqrt(x_dist * x_dist + y_dist * y_dist);
    const float color_dist = fabs(pix - center_pix);
    return exp(-spatial_dist / (2.0f * sigma_spatial * sigma_spatial) - color_dist / (2.0f * sigma_color * sigma_color));
}

__device__ float ComputeBilateralWeight(const float x_dist, const float y_dist, const float4 pix, const float4 center_pix, const float sigma_spatial, const float sigma_color)
{
    const float spatial_dist = sqrt(x_dist * x_dist + y_dist * y_dist);
    const float color_dist = l1_norm(pix - center_pix);
    return exp(-spatial_dist / (2.0f * sigma_spatial * sigma_spatial) - color_dist / (2.0f * sigma_color * sigma_color));
}




__device__ float getDepthFromPlane3_cu(const Camera_cu &cam,
                                       const float4 &n,
                                       const float &d,
                                       const int2 &p)
{
    return -d * cam.fx / ((n.x * (p.x - cam.K[2])) + (n.y * (p.y - cam.K[2 + 3])) * cam.alpha + n.z * cam.fx);
}


/**
 * @brief generate random number from the given min and max
 *
 */
__device__ float curand_between(curandState *cs, const float &min, const float &max)
{
    return (curand_uniform(cs) * (max - min) + min);
}



__device__ float3 Get3DPointonWorld_cu(const float x, const float y, const float depth, const Camera_cu &camera)
{
    float3 pointX;
    float3 tmpX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    // Rotation
    tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
    tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
    tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

    // Transformation
    float3 C;
    C.x = -(camera.R[0] * camera.t4.x + camera.R[3] * camera.t4.y + camera.R[6] * camera.t4.z);
    C.y = -(camera.R[1] * camera.t4.x + camera.R[4] * camera.t4.y + camera.R[7] * camera.t4.z);
    C.z = -(camera.R[2] * camera.t4.x + camera.R[5] * camera.t4.y + camera.R[8] * camera.t4.z);
    pointX.x = tmpX.x + C.x;
    pointX.y = tmpX.y + C.y;
    pointX.z = tmpX.z + C.z;

    return pointX;
}



__device__ float ComputeDepthfromPlaneHypothesis(const Camera_cu & camera, const float4 plane_hypothesis, const int2 p)
{
    return -plane_hypothesis.w * camera.K[0] / ((p.x - camera.K[2]) * plane_hypothesis.x + (camera.K[0] / camera.K[4]) * (p.y - camera.K[5]) * plane_hypothesis.y + camera.K[0] * plane_hypothesis.z);
}


__device__ void ProjectonCamera_cu(const float3 PointX, const Camera_cu &camera, float2 &point, float &depth)
{
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t4.x;
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t4.y;
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t4.z;

    depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}





// Pixel-wise view selection
template <typename T>
__device__ float pmCost_viewsel(
    const cudaTextureObject_t &l, // eference image
    const cudaTextureObject_t &r, // source_image
    const int x,                  // location of the pixel
    const int y,
    const float4 &normal, 
    const int &vRad,      // calculate the local area
    const int &hRad,
    const CameraParameters_cu &cpc,
    const int &camTo,
    const int cost_choice,
    const int current_image,
    const GlobalState &gs,
    bool print_opt)
{

    float H[16] = {0};
    getHomography_cu(cpc.cameras[current_image], cpc.cameras[camTo],
                     cpc.cameras[current_image].K_inv,
                     cpc.cameras[camTo].K,
                     normal, normal.w, H);
    
    
    bool occluded = false;
    bool over_rad = false;
    // if (gs.cycles != -1)
    // { 
        
        // get the transformed depth
        int2 ref_pt_center = make_int2(x, y);
    
        float depth = ComputeDepthfromPlaneHypothesis(cpc.cameras[current_image], normal, ref_pt_center);
        float3 forward_point = Get3DPointonWorld_cu(ref_pt_center.x, ref_pt_center.y, depth, cpc.cameras[current_image]);
        float maximum_angle_radians = 80 * M_PI / 180.0f;
        float4 vec_ref; // view vector between the 3D point and the reference camera
        float4 vec_source;  // view vector between the 3D point and the souce camera
        // vec_ref.x = cpc.cameras[current_image].C4.x - forward_point.x;
        // vec_ref.y = cpc.cameras[current_image].C4.y - forward_point.y;
        // vec_ref.z = cpc.cameras[current_image].C4.z - forward_point.z;
        vec_ref.x = normal.x;
        vec_ref.y = normal.y;
        vec_ref.z = normal.z;
        NormalizeVec3(&vec_ref);
        vec_source.x = cpc.cameras[camTo].C4.x - forward_point.x;
        vec_source.y = cpc.cameras[camTo].C4.y - forward_point.y;
        vec_source.z = cpc.cameras[camTo].C4.z - forward_point.z;
        NormalizeVec3(&vec_source);
        
        float angle = acosf ( vec_ref.x * vec_source.x + vec_ref.y * vec_source.y + vec_ref.z * vec_source.z);

        if (angle > maximum_angle_radians)
        {
            over_rad = true;
        }        

        // get the source pixel and the depth
        float2 source_pixel;
        float src_d;
        ProjectonCamera_cu(forward_point, cpc.cameras[camTo], source_pixel, src_d);

        if (source_pixel.x >=  1&&
            source_pixel.x < gs.cols -1 &&
            source_pixel.y >=  1&&
            source_pixel.y < gs.rows -1)
        {
    
            float4 source; // first 3 components normal, fourth depth
            source = tex2D<float4>(gs.normals_depths[camTo], source_pixel.x + 0.5f, source_pixel.y + 0.5f);

            if (src_d - source.w > 2) // the pixel is occluded
            {
                
                // printf("a%d, \n", ((int)(source_pixel.y + 0.5f)) * gs.cols + (int)(source_pixel.x + 0.5f));
                if (gs.lines_consistency[camTo].used_pixels[((int)(source_pixel.y + 0.5f)) * gs.cols + (int)(source_pixel.x + 0.5f)] >= 1) // occluded pixel satisfies strong consistency check
                    occluded = true;
            }
        }
        else
        {
            occluded = true;
        }
    
    // }
    
    // float2 src_pt_center = ComputeCorrespondingPoint(H, ref_pt_center);

    if (occluded) // if comment->nopixsel
    {
        return 9999;
    }
    
    float cost = 0.0f;
    int radius = hRad / 2;
    float sigma_spatial = 5.0f;
    float sigma_color = 3.0f;
    const float cost_max = 2.0f;
    {
        float sum_ref = 0.0f;
        float sum_ref_ref = 0.0f;
        float sum_src = 0.0f;
        float sum_src_src = 0.0f;
        float sum_ref_src = 0.0f;
        float bilateral_weight_sum = 0.0f;
        const float ref_center_pix = tex2D<float>(l, x + 0.5f, y + 0.5f);

        for (int i = -radius; i < radius + 1; i += 2)
        {
            float sum_ref_row = 0.0f;
            float sum_src_row = 0.0f;
            float sum_ref_ref_row = 0.0f;
            float sum_src_src_row = 0.0f;
            float sum_ref_src_row = 0.0f;
            float bilateral_weight_sum_row = 0.0f;

            for (int j = -radius; j < radius + 1; j += 2)
            {
                const int2 ref_pt = make_int2(x + i, y + j);
                const float ref_pix = tex2D<float>(l, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
                float2 src_pt = ComputeCorrespondingPoint(H, ref_pt);                
                const float src_pix = tex2D<float>(r, src_pt.x + 0.5f, src_pt.y + 0.5f);

                float weight = ComputeBilateralWeight(i, j, ref_pix, ref_center_pix, sigma_spatial, sigma_color);

                sum_ref_row += weight * ref_pix;
                sum_ref_ref_row += weight * ref_pix * ref_pix;
                sum_src_row += weight * src_pix;
                sum_src_src_row += weight * src_pix * src_pix;
                sum_ref_src_row += weight * ref_pix * src_pix;
                bilateral_weight_sum_row += weight;
            }

            sum_ref += sum_ref_row;
            sum_ref_ref += sum_ref_ref_row;
            sum_src += sum_src_row;
            sum_src_src += sum_src_src_row;
            sum_ref_src += sum_ref_src_row;
            bilateral_weight_sum += bilateral_weight_sum_row;
        }
        const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
        sum_ref *= inv_bilateral_weight_sum;
        sum_ref_ref *= inv_bilateral_weight_sum;
        sum_src *= inv_bilateral_weight_sum;
        sum_src_src *= inv_bilateral_weight_sum;
        sum_ref_src *= inv_bilateral_weight_sum;

        const float var_ref = sum_ref_ref - sum_ref * sum_ref;
        const float var_src = sum_src_src - sum_src * sum_src;

        const float kMinVar = 1e-5f;
        if (var_ref < kMinVar || var_src < kMinVar)
        {
            cost = cost_max; 
        }
        else
        {
            const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
            const float var_ref_src = sqrt(var_ref * var_src);
            cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
        }
        // if (over_rad || occluded) // if (over_rad || occluded)->nopixsel
        if (over_rad)
            return -cost;

       
        return cost;
    }
    

}

//0.299R + 0.587G + 0.114B
__device__ float compress(float4 a)
{
    return (0.333f * a.x + 0.333f * a.y + 0.333f * a.z);
    
}


// Pixel-wise view selection
template <typename T>
__device__ float pmCost_viewsel_float4(
    const cudaTextureObject_t &l, // reference image
    const cudaTextureObject_t &r, // source_image
    const int x,                  
    const int y,
    const float4 &normal,
    const int &vRad,      
    const int &hRad,
    const CameraParameters_cu &cpc,
    const int &camTo,
    const int cost_choice,
    const int current_image,
    const GlobalState &gs,
    bool print_opt)
{

    float H[16] = {0};
    getHomography_cu(cpc.cameras[current_image], cpc.cameras[camTo],
                     cpc.cameras[current_image].K_inv,
                     cpc.cameras[camTo].K,
                     normal, normal.w, H);
    
    
    bool occluded = false;
    bool over_rad = false;
    // if (gs.cycles != -1)
    // { 
        
        const CameraParameters_cu &camParams = *(gs.cameras_orig);
        // get the transformed depth
        int2 ref_pt_center = make_int2(x, y);
        
    
        float depth = ComputeDepthfromPlaneHypothesis(cpc.cameras[current_image], normal, ref_pt_center);
        float3 forward_point = Get3DPointonWorld_cu(ref_pt_center.x, ref_pt_center.y, depth, cpc.cameras[current_image]);
        float maximum_angle_radians = 80 * M_PI / 180.0f;
        float4 vec_ref; // view vector between the 3D point and the reference camera
        float4 vec_source;  // view vector between the 3D point and the souce camera
        // vec_ref.x = cpc.cameras[current_image].C4.x - forward_point.x;
        // vec_ref.y = cpc.cameras[current_image].C4.y - forward_point.y;
        // vec_ref.z = cpc.cameras[current_image].C4.z - forward_point.z;
        vec_ref.x = normal.x;
        vec_ref.y = normal.y;
        vec_ref.z = normal.z;
        NormalizeVec3(&vec_ref);
        vec_source.x = cpc.cameras[camTo].C4.x - forward_point.x;
        vec_source.y = cpc.cameras[camTo].C4.y - forward_point.y;
        vec_source.z = cpc.cameras[camTo].C4.z - forward_point.z;
        NormalizeVec3(&vec_source);
        
        float angle = acosf ( vec_ref.x * vec_source.x + vec_ref.y * vec_source.y + vec_ref.z * vec_source.z);

        if (angle > maximum_angle_radians)
        {
            over_rad = true;
        }        

        // get the source pixel and the depth
        float2 source_pixel;
        float src_d;
        ProjectonCamera_cu(forward_point, cpc.cameras[camTo], source_pixel, src_d);

        if (source_pixel.x >=  1&&
            source_pixel.x < gs.cols -1 &&
            source_pixel.y >=  1&&
            source_pixel.y < gs.rows -1)
        {
    
            float4 source; // first 3 components normal, fourth depth
            source = tex2D<float4>(gs.normals_depths[camTo], source_pixel.x + 0.5f, source_pixel.y + 0.5f);

            if (src_d - source.w > 2) // the pixel is occluded
            {
                
                // printf("a%d, \n", ((int)(source_pixel.y + 0.5f)) * gs.cols + (int)(source_pixel.x + 0.5f));
                if (gs.lines_consistency[camTo].used_pixels[((int)(source_pixel.y + 0.5f)) * gs.cols + (int)(source_pixel.x + 0.5f)] >= 1) // occluded pixel satisfies strong consistency check
                    occluded = true;
            }
        }
        else
        {
            occluded = true;
        }
    
    // }
    
    // float2 src_pt_center = ComputeCorrespondingPoint(H, ref_pt_center);

    if (occluded)
    {
        return 9999;
    }
    
    float cost = 0.0f;
    int radius = hRad / 2;
    float sigma_spatial = 5.0f;
    float sigma_color = 3.0f;
    const float cost_max = 2.0f;
    {
        float sum_ref = 0.0f;
        float sum_ref_ref = 0.0f;
        float sum_src = 0.0f;
        float sum_src_src = 0.0f;
        float sum_ref_src = 0.0f;
        float bilateral_weight_sum = 0.0f;
        const float4 ref_center_pix = tex2D<float4>(l, x + 0.5f, y + 0.5f);

        for (int i = -radius; i < radius + 1; i += 2)
        {
            float sum_ref_row = 0.0f;
            float sum_src_row = 0.0f;
            float sum_ref_ref_row = 0.0f;
            float sum_src_src_row = 0.0f;
            float sum_ref_src_row = 0.0f;
            float bilateral_weight_sum_row = 0.0f;

            for (int j = -radius; j < radius + 1; j += 2)
            {
                const int2 ref_pt = make_int2(x + i, y + j);
                const float4 ref_pix = tex2D<float4>(l, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
                float2 src_pt = ComputeCorrespondingPoint(H, ref_pt);          
                const float4 src_pix = tex2D<float4>(r, src_pt.x + 0.5f, src_pt.y + 0.5f);
                float weight = ComputeBilateralWeight(i, j, ref_pix, ref_center_pix, sigma_spatial, sigma_color);

                sum_ref_row += compress(weight * ref_pix);
                sum_ref_ref_row += compress(weight * ref_pix * ref_pix);
                sum_src_row += compress(weight * src_pix);
                sum_src_src_row += compress(weight * src_pix * src_pix);
                sum_ref_src_row += compress(weight * ref_pix * src_pix);
                bilateral_weight_sum_row += weight;
            }

            sum_ref += sum_ref_row;
            sum_ref_ref += sum_ref_ref_row;
            sum_src += sum_src_row;
            sum_src_src += sum_src_src_row;
            sum_ref_src += sum_ref_src_row;
            bilateral_weight_sum += bilateral_weight_sum_row;
        }
        const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
        sum_ref *= inv_bilateral_weight_sum;
        sum_ref_ref *= inv_bilateral_weight_sum;
        sum_src *= inv_bilateral_weight_sum;
        sum_src_src *= inv_bilateral_weight_sum;
        sum_ref_src *= inv_bilateral_weight_sum;

        const float var_ref = sum_ref_ref - sum_ref * sum_ref;
        const float var_src = sum_src_src - sum_src * sum_src;

        const float kMinVar = 1e-5f;
        if (var_ref < kMinVar || var_src < kMinVar)
        {
            cost = cost_max; 
        }
        else
        {
            const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
            const float var_ref_src = sqrt(var_ref * var_src);
            cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
        }
        if (over_rad)
            return -cost;

        return cost;
    }
    

}


// f-fitness w-foods
__device__ void sort_large_by_fitness(float *f, float4 *w, int n)
{
    int j;

    for (int i = 1; i < n; i++) {
        float tmp = f[i];
        float4 tmp_w = w[i];
        for (j = i; j >= 1 && tmp > f[j - 1]; j--) {
            f[j] = f[j - 1];
            w[j] = w[j - 1];
        }
        f[j] = tmp;
        w[j] = tmp_w;
    }
    
}

// f-fitness w-foods
__device__ void sort_large_by_fitness_1d(float *f, float4 *w, int n, int center)
{
    int h = 0;
    float h_f = 0;
    for (int i = 0; i < n; i++)
    {
        if (f[idx(center, i, n)] > h_f)
        {
            h = i;
            h_f = f[idx(center, i, n)];
        } 
      
    }

    // switch highest
    float4 tmp = w[idx(center, 0, n)];
    w[idx(center, 0, n)] = w[idx(center, h, n)];
    w[idx(center, h, n)] = tmp;

    f[idx(center, h, n)] = f[idx(center, 0, n)];
    f[idx(center, 0, n)] = h_f;
    
}




// via https://stackoverflow.com/questions/2786899/fastest-sort-of-fixed-length-6-int-array
__device__ void sort_small(float *d, int n)
{
    int j;
    for (int i = 1; i < n; i++)
    {
        float tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j - 1]; j--)
            d[j] = d[j - 1];
        d[j] = tmp;
    }
}


template <typename T>
__device__ float functionABC_singlepoint(int x, int y,
                                         float4 solution,            // const CameraParameters_cu &cpc,
                                         const GlobalState &gs,
                                         bool print_opt) // double sol[D])// width=480, D=4
{
    float top[800] = {9999.f};
    for (int i = 0; i < 800; i++)    top[i] = 9999.f;

  
    // number of valid views that would be cosidered for final cost caluclation
    int numValidViews = gs.cameras->selectedViewsNum_ambc[gs.current_image];

    for (int i = 0; i < gs.cameras->selectedViewsNum_ambc[gs.current_image]; i++) 
    {
        int idxCurr = gs.cameras->selectedViews_ambc[gs.current_image][i];
        
        top[i] = pmCost_viewsel<T>(
        gs.imgs[gs.current_image],
        gs.imgs[idxCurr],
        x,
        y,
        solution,
        gs.params->box_vsize,
        gs.params->box_hsize,
        *(gs.cameras),
        idxCurr,
        gs.params->cost_choice,
        gs.current_image,
        gs,
        print_opt);
        
        if (top[i] >= MAXCOST)
        {
            numValidViews = numValidViews - 1;
            // if (numValidViews <= 4)
            // {
            //     numValidViews ++;
            //     top[i] = -top[i];
            // }
            // else
            // {
                top[i] = 9999.f; 
            // }
            

        }
        // TODO
        else if (top[i] < 0)
        {
            numValidViews = numValidViews - 1;// pixel
            // if (numValidViews <= 5)
            // {
            //     numValidViews ++;
                top[i] = -top[i];// nopixel
            // }
            // else
            // {
                top[i] = 9999.f;// pixel
            // }
            
        }         
    }

    sort_small(top, gs.cameras->selectedViewsNum_ambc[gs.current_image]); 

    float cost = 0;
    for (int i = 0; i < numValidViews; i++)
    {
        cost += top[i];
    }

    // gs.lines->numBests[y * gs.cols + x] = numValidViews;
    if (numValidViews <= 2)
        return MAXCOST;
    else
    {
        cost = cost / ((float)numValidViews - 2);
        return cost;
    }
}

// CHECKED
__device__ void get3Dpoint_cu1(float4 *__restrict__ ptX, const Camera_cu &cam, const int2 &p)
{
    // in case camera matrix is not normalized: see page 162, then depth might not be the real depth but w and depth needs to be computed from that first
    float4 pt;
    pt.x = (float)p.x - cam.P_col34.x;
    pt.y = (float)p.y - cam.P_col34.y;
    pt.z = 1.0f - cam.P_col34.z;

    matvecmul4(cam.M_inv, pt, ptX);
}


/**
 * @brief Get the Random Normal object by random method from the Gipuma paper. Then store it in the foods array.
 *
 * @param gs GlobalState &gs
 * @param center center index of the pixel
 * @param index food index of the colony
 * @param p int2 representing the pixel
 * @param localState used for random method
 */
__device__ void getRandomNormal(GlobalState &gs, const int center, const int index, int2 p)
{
    float q1 = 0.0, q2 = 0.0;
    bool bigger_than_1 = true;
    curandState localState = gs.cs[center];
    // curand_init(clock64(), p.y, p.x, &localState );
    // calculate random value q1 and q2 from Massively Parallel Multiview Stereopsis by Surface Normal Diffusion.
    while (bigger_than_1 == true)
    {
        // curand_init(clock64(), p.y, p.x, &localState);
        q1 = curand_between(&localState, -1.0f, 1.0f); //-1~1
        gs.cs[center] = localState;
        // curand_init(clock64(), p.y, p.x, &localState);

        q2 = curand_between(&localState, -1.0f, 1.0f); //-1~1
        gs.cs[center] = localState;

        if ((q1 * q1 + q2 * q2) < 0.999)
        {
            float S = q1 * q1 + q2 * q2;
            gs.lines->foods_1d[idx(center, index, gs.lines->FoodNumber)].x = 1 - 2 * S;
            gs.lines->foods_1d[idx(center, index, gs.lines->FoodNumber)].y = 2 * q1 * sqrt(1 - S);
            gs.lines->foods_1d[idx(center, index, gs.lines->FoodNumber)].z = 2 * q2 * sqrt(1 - S);
            bigger_than_1 = false;
        }
    }
}

__device__ void getViewVector_cu(float4 *v, const Camera_cu &camera, const int2 &p)
{
    get3Dpoint_cu1(v, camera, p);
    sub((*v), camera.C4);
    (*v).x = camera.C4.x - (*v).x;
    (*v).y = camera.C4.y - (*v).y;
    (*v).z = camera.C4.z - (*v).z;
    NormalizeVec3(v);
}

__device__ FORCEINLINE_GIPUMA static void vecOnHemisphere_cu(float4 *__restrict__ v, const float4 &viewVector)
{
    const float dp = dot4((*v), viewVector);
    if (dp < 0.0f)
    {
        negate4(v);
    }
    return;
}

template <typename T>
__global__ void ambc_calc_cost(GlobalState &gs)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= gs.cols)
        return; 
    if (p.y >= gs.rows)
        return; 
    const int center = p.y * gs.cols + p.x;


    // initial all food sources
    for (int index = 0; index < gs.lines->FoodNumber; index++) // means the food sources.
    {
        // float cost = functionABC_singlepoint<T>(p.x, p.y, gs.lines->foods[center][index], gs, false); // calculate the cost of the plane on current pixel
        // gs.lines->fitness[center][index] = cost >= 0 ? (1 / (cost + 1)) : (1 - cost); // fitness =  1 / (cost + 1) if cost >=0, which is most of the case
        
        gs.lines->trial_1d[idx(center, index, gs.lines->FoodNumber)] = 0;
    }
    sort_large_by_fitness_1d(gs.lines->fitness_1d, gs.lines->foods_1d, gs.lines->FoodNumber, center);


    
    // Propagation between views    
    float4 norm_transformed_back;
    float4 viewVector;

    matvecmul4(gs.cameras->cameras[gs.current_image].R_orig, gs.lines_consistency[gs.current_image].suggestions[center], (&norm_transformed_back));
    norm_transformed_back.w = getD_cu(norm_transformed_back, p, gs.lines_consistency[gs.current_image].suggestions[center].w, gs.cameras->cameras[gs.current_image]);

    
    
    getViewVector_cu(&viewVector, gs.cameras->cameras[gs.current_image], p);
    vecOnHemisphere_cu(&norm_transformed_back, viewVector);
    NormalizeVec3(&norm_transformed_back);

    float cost1 = functionABC_singlepoint<T>(p.x, p.y, norm_transformed_back, gs, false);
    //smooth

    //TODO: select the worst foodsource
    float fitness1 = cost1 >= 0 ? (1 / (cost1 + 1)) : (1 - cost1);
    if (fitness1 > gs.lines->fitness_1d[idx(center, gs.lines->FoodNumber-1, gs.lines->FoodNumber)])
    {

        
        gs.lines->foods_1d[idx(center, gs.lines->FoodNumber-1, gs.lines->FoodNumber)].x = norm_transformed_back.x;
        gs.lines->foods_1d[idx(center, gs.lines->FoodNumber-1, gs.lines->FoodNumber)].y = norm_transformed_back.y;
        gs.lines->foods_1d[idx(center, gs.lines->FoodNumber-1, gs.lines->FoodNumber)].z = norm_transformed_back.z;
        gs.lines->foods_1d[idx(center, gs.lines->FoodNumber-1, gs.lines->FoodNumber)].w = norm_transformed_back.w;
        gs.lines->fitness_1d[idx(center, gs.lines->FoodNumber-1, gs.lines->FoodNumber)] = fitness1;

       
        
    }

    sort_large_by_fitness_1d(gs.lines->fitness_1d, gs.lines->foods_1d, gs.lines->FoodNumber, center);

    return;
}




/**
 * @brief Initialization function for all food sources, depth is sampled in depth space for now.
 *
 * @param gs GlobalState &gs
 */
template <typename T>
__global__ void ambc_init_cu(GlobalState &gs)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= gs.cols)
        return; 
    if (p.y >= gs.rows)
        return; 
    const int center = p.y * gs.cols + p.x;
    // initial all food sources
    curand_init(clock64(), p.y, p.x, &gs.cs[center]);
    curandState localState = gs.cs[center];
    for (int index = 0; index < gs.lines->FoodNumber; index++) // means the food sources.
    {
        // Random normal and make sure it faces towards the camera
        getRandomNormal(gs, center, index, p);
        float4 viewVector;

        getViewVector_cu(&viewVector, gs.cameras->cameras[gs.current_image], p);
        vecOnHemisphere_cu(&gs.lines->foods_1d[idx(center, index, gs.lines->FoodNumber)], viewVector);
        NormalizeVec3(&gs.lines->foods_1d[idx(center, index, gs.lines->FoodNumber)]);
        // TODO: sample in the disparity space
        // Random depth in the boundary

        float disp = curand_between(&localState, gs.depth_lower_bound, gs.depth_upper_bound);
        gs.cs[center] = localState;

        gs.lines->foods_1d[idx(center, index, gs.lines->FoodNumber)].w = getD_cu(gs.lines->foods_1d[idx(center, index, gs.lines->FoodNumber)], p, disp, gs.cameras->cameras[gs.current_image]); // Transform depth to the distance between center and the plane
        
        float cost = functionABC_singlepoint<T>(p.x, p.y, gs.lines->foods_1d[idx(center, index, gs.lines->FoodNumber)], gs, false); // calculate the cost of the plane on current pixel

        gs.lines->fitness_1d[idx(center, index, gs.lines->FoodNumber)] = cost >= 0 ? (1 / (cost + 1)) : (1 - cost); // fitness =  1 / (cost + 1) if cost >=0, which is most of the case
    }
    
    // sort_large_by_fitness(gs.lines->fitness[center], gs.lines->foods[center], gs.lines->FoodNumber);

    return;
}

/**
 * @brief
 *
 * @param gs
 * @param iter
 */
template <typename T>
__global__ void SendScoutBees_cu(GlobalState &gs, int iter)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= gs.cols)
        return; 
    if (p.y >= gs.rows)
        return; 
    const int center = p.y * gs.cols + p.x;


    int maxtrialindex = 3;
    for (int i = 1; i < gs.lines->FoodNumber; i++) // Skip top-1 food sources
    {
        if (gs.lines->trial_1d[idx(center, i, gs.lines->FoodNumber)] > gs.lines->trial_1d[idx(center, maxtrialindex, gs.lines->FoodNumber)])
            maxtrialindex = i;
    }
    curandState localState = gs.cs[center];
    // A food source which could not be improved through "limit" trials is abandoned by its employed bee except best food source.
    if (gs.lines->trial_1d[idx(center, maxtrialindex, gs.lines->FoodNumber)] >= gs.limit )    
    //TODO: limit is set to 10 currently, which is useless after warmup runs.
    {

        getRandomNormal(gs, center, maxtrialindex, p);
        float4 viewVector;
        // getViewVector_cu(&viewVector, gs.cameras->cameras[REFERENCE], p);
        getViewVector_cu(&viewVector, gs.cameras->cameras[gs.current_image], p);
        vecOnHemisphere_cu(&gs.lines->foods_1d[idx(center, maxtrialindex, gs.lines->FoodNumber)], viewVector);
        NormalizeVec3(&gs.lines->foods_1d[idx(center, maxtrialindex, gs.lines->FoodNumber)]);
        // curand_init(clock64(), p.y, p.x, &localState);
        float disp = curand_between(&localState, gs.depth_lower_bound, gs.depth_upper_bound); // r;// (ub_depth-lb_depth)+lb_depth;;//r*0.5+0.5;  //0~1
        gs.cs[center] = localState;
        // disp = disparityDepthConversion_cu ( gs.cameras->cameras[REFERENCE].fx, gs.cameras->cameras[REFERENCE].baseline, disp);
        gs.lines->foods_1d[idx(center, maxtrialindex, gs.lines->FoodNumber)].w = getD_cu(gs.lines->foods_1d[idx(center, maxtrialindex, gs.lines->FoodNumber)], p, disp, gs.cameras->cameras[gs.current_image]);
        
        // gs.lines->foods[center][maxtrialindex].w = curand_between(&localState, gs.depth_lower_bound, gs.depth_upper_bound); // depth from 0 to 1.
        float cost = functionABC_singlepoint<T>(p.x, p.y, gs.lines->foods_1d[idx(center, maxtrialindex, gs.lines->FoodNumber)], gs, false);
        gs.lines->fitness_1d[idx(center, maxtrialindex, gs.lines->FoodNumber)] = cost >= 0 ? (1 / (cost + 1)) : (1 - cost);
        gs.lines->trial_1d[idx(center, maxtrialindex, gs.lines->FoodNumber)] = 0;
        
    }
    // sort_large_by_fitness(gs.lines->fitness[center], gs.lines->foods[center], gs.lines->FoodNumber);
    sort_large_by_fitness_1d(gs.lines->fitness_1d, gs.lines->foods_1d, gs.lines->FoodNumber, center);
}


template <typename T>
__device__ void calOnlookerBees_cu_best(GlobalState &gs, int2 p, const int center_other, int food_number_index, int other_x, int other_y)
{

    const int center = p.y * gs.cols + p.x;
    float4 solution = gs.lines->foods_1d[idx(center_other, 0, gs.lines->FoodNumber)]; // food with highest fitness

    // float4 viewVector;
    // getViewVector_cu(&viewVector, gs.cameras->cameras[gs.current_image], p);
    // vecOnHemisphere_cu(&solution, viewVector);
    // NormalizeVec3(&solution);

    // float D_lower_bound = getD_cu(solution, p, gs.depth_lower_bound, gs.cameras->cameras[gs.current_image]); // Transform depth to the distance between center and the plane
    // float D_upper_bound = getD_cu(solution, p, gs.depth_upper_bound, gs.cameras->cameras[gs.current_image]); // Transform depth to the distance between center and the plane


    // if (solution.w < D_lower_bound)
    //     solution.w = D_lower_bound;
    // if (solution.w > D_upper_bound)
    //     solution.w = D_upper_bound;

    float ObjValSol = functionABC_singlepoint<T>(p.x, p.y, solution, gs, false);
   

   

    float FitnessSol = ObjValSol >= 0 ? (1 / (ObjValSol + 1)) : (1 - ObjValSol);
    //smooth
    if (gs.lines_consistency[gs.current_image].used_pixels[center_other] == 1)
    {
        FitnessSol = FitnessSol + 0.15;
    }
  
    if (FitnessSol > gs.lines->fitness_1d[idx(center, food_number_index, gs.lines->FoodNumber)])
    {
        // If the mutant solution is better than the current solution i,
        // replace the solution with the mutant and reset the trial counter of solution i.
        gs.lines->trial_1d[idx(center, food_number_index, gs.lines->FoodNumber)] = 0;
        gs.lines->foods_1d[idx(center, food_number_index, gs.lines->FoodNumber)] = solution;
        gs.lines->fitness_1d[idx(center, food_number_index, gs.lines->FoodNumber)] = FitnessSol;
    }
    else
    { // if the solution i can not be improved, increase its trial counter
        gs.lines->trial_1d[idx(center, food_number_index, gs.lines->FoodNumber)] = gs.lines->trial_1d[idx(center, food_number_index, gs.lines->FoodNumber)] + 1;
    }
}


template <typename T>
__global__ void SendOnlookerBees_cu_black_best(GlobalState &gs)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0)
        p.y = p.y * 2;
    else
        p.y = p.y * 2 + 1;
    if (p.x >= gs.cols - 2)
        return; 
    if (p.y >= gs.rows - 2)
        return;
    
  
    int offset = 3;
    float r; // random number
    const int center = p.y * gs.cols + p.x;

    if (false)//gs.lines_consistency[gs.current_image].used_pixels[center]>=1) 
    {
        return;
    }
    else
    {
        curandState localState = gs.cs[center];
        // Left
        int left = center - 1;
        if (left < offset) return;//left = offset;
        int left5 = center - 5;
        if (left5 < offset) return;//left5 = offset;
        int left9 = center - 15;
        if (left9 < offset) return;//left9 = offset;
        // Up
        int up = center - gs.cols;
        if (up < offset) return;//up = offset;
        int up5 = center - gs.cols*5;
        if (up5 < offset) return;//up5 = offset;
        int up9 = center - gs.cols*15;
        if (up9 < offset) return;//up9 = offset;
        int up2l = center - gs.cols*2 - 1;
        if (up2l < offset) return;//up2l = offset;
        int up2r = center - gs.cols*2 + 1;
        if (up2r < offset) return;//up2r = offset;
        int up1l = center - gs.cols - 2;
        if (up1l < offset) return;//up1l = offset;
        int up1r = center - gs.cols + 2;
        if (up1r < offset) return;//up1r = offset;

        // Down
        int down = center + gs.cols;
        if (down > gs.rows * gs.cols - offset) return;//down = gs.rows * gs.cols - offset;
        int down5 = center + gs.cols*5;
        if (down5 > gs.rows * gs.cols - offset) return;//down5 = gs.rows * gs.cols - offset;
        int down9 = center + gs.cols*15;
        if (down9 > gs.rows * gs.cols - offset) return;//down9 = gs.rows * gs.cols - offset;   
        int down2l = center + gs.cols*2 - 1;
        if (down2l > gs.rows * gs.cols - offset) return;//down2l = gs.rows * gs.cols - offset;
        int down2r = center + gs.cols*2 + 1;
        if (down2r > gs.rows * gs.cols - offset) return;//down2r = gs.rows * gs.cols - offset;
        int down1l = center + gs.cols - 2;
        if (down1l > gs.rows * gs.cols - offset) return;//down1l = gs.rows * gs.cols - offset;
        int down1r = center + gs.cols + 2;
        if (down1r > gs.rows * gs.cols - offset) return;//down1r = gs.rows * gs.cols - offset;

        // Right
        int right = center + 1;
        if (right > gs.rows * gs.cols - offset) return;//right = gs.rows * gs.cols - offset;
        int right5 = center + 5;
        if (right5 > gs.rows * gs.cols - offset) return;//right5 = gs.rows * gs.cols - offset;
        int right9 = center + 15;
        if (right9 > gs.rows * gs.cols - offset) return;//right9 = gs.rows * gs.cols - offset;


        // curandState localState = gs.cs[center];
        for (int i = 0; i < gs.lines->FoodNumber; i++)
        {

            // curand_init(clock64(), p.y, p.x, &localState);
            int rand = curand_between(&localState, 0.0f, 8); // get number from 0~sum
            gs.cs[center] = localState;

            float rand3 = curand_between(&localState, 0.0f, 3); // get number from 0~3
            gs.cs[center] = localState;
            // printf("rand: %d, id: %d\n", rand, center);
                if (rand < 1)
                {
                    if (rand3<1)        calOnlookerBees_cu_best<T>(gs, p, up, i, p.x, p.y-1);
                    else if(rand3 < 2)  calOnlookerBees_cu_best<T>(gs, p, up5, i, p.x, p.y - 5);
                    else                calOnlookerBees_cu_best<T>(gs, p, up9, i, p.x, p.y -9);
                }
                else if (rand < 2)
                {
                    if (rand3<1)        calOnlookerBees_cu_best<T>(gs, p, down, i, p.x, p.y+1);
                    else if(rand3 < 2)  calOnlookerBees_cu_best<T>(gs, p, down5, i, p.x, p.y+5);
                    else                calOnlookerBees_cu_best<T>(gs, p, down9, i, p.x, p.y+9);
                }
                else if (rand < 3)
                {
                    if (rand3<1)        calOnlookerBees_cu_best<T>(gs, p, left, i,p.x-1, p.y);
                    else if(rand3 < 2)  calOnlookerBees_cu_best<T>(gs, p, left5, i,p.x-5, p.y);
                    else                calOnlookerBees_cu_best<T>(gs, p, left9, i,p.x-9, p.y);
                }
                else if (rand < 4)
                {
                    if (rand3<1)        calOnlookerBees_cu_best<T>(gs, p, right, i,p.x+1, p.y);
                    else if(rand3 < 2)  calOnlookerBees_cu_best<T>(gs, p, right5, i,p.x+5, p.y);
                    else                calOnlookerBees_cu_best<T>(gs, p, right9, i,p.x+9, p.y);
                }
                else if (rand < 5)
                {
                    if (rand3<1.5)        calOnlookerBees_cu_best<T>(gs, p, up2l, i, p.x - 1, p.y-2);
                    else                  calOnlookerBees_cu_best<T>(gs, p, up1l, i, p.x -2, p.y-1);                    
                }
                else if (rand < 6)
                {
                    if (rand3<1.5)        calOnlookerBees_cu_best<T>(gs, p, up2r, i, p.x + 1, p.y -2);
                    else                  calOnlookerBees_cu_best<T>(gs, p, up1r, i, p.x + 2, p.y-1);                    
                }
                else if (rand < 7)
                {
                    if (rand3<1.5)        calOnlookerBees_cu_best<T>(gs, p, down2l, i, p.x-1, p.y+2);
                    else                  calOnlookerBees_cu_best<T>(gs, p, down1l, i, p.x-2, p.y+1);                    
                }
                else if (rand <= 8)
                {
                    if (rand3<1.5)        calOnlookerBees_cu_best<T>(gs, p, down2r, i, p.x+1, p.y+2);
                    else                  calOnlookerBees_cu_best<T>(gs, p, down1r, i, p.x+2, p.y+1);                    
                }
        }

        sort_large_by_fitness_1d(gs.lines->fitness_1d, gs.lines->foods_1d, gs.lines->FoodNumber, center);
        
    }
}


template <typename T>
__global__ void SendOnlookerBees_cu_red_best(GlobalState &gs)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0)
        p.y = p.y * 2 + 1;
    else
        p.y = p.y * 2;
    if (p.x >= gs.cols - 2)
        return; 
    if (p.y >= gs.rows - 2)
        return; 

    int offset = 3;
    const int center = p.y * gs.cols + p.x;


    if (false)//gs.lines_consistency[gs.current_image].used_pixels[center]>=1) 
    {
        return;
    }
    else
    {        
        curandState localState = gs.cs[center];

       // Left
        int left = center - 1;
        if (left < offset) return;//left = offset;
        int left5 = center - 5;
        if (left5 < offset) return;//left5 = offset;
        int left9 = center - 15;
        if (left9 < offset) return;//left9 = offset;
        // Up
        int up = center - gs.cols;
        if (up < offset) return;//up = offset;
        int up5 = center - gs.cols*5;
        if (up5 < offset) return;//up5 = offset;
        int up9 = center - gs.cols*15;
        if (up9 < offset) return;//up9 = offset;
        int up2l = center - gs.cols*2 - 1;
        if (up2l < offset) return;//up2l = offset;
        int up2r = center - gs.cols*2 + 1;
        if (up2r < offset) return;//up2r = offset;
        int up1l = center - gs.cols - 2;
        if (up1l < offset) return;//up1l = offset;
        int up1r = center - gs.cols + 2;
        if (up1r < offset) return;//up1r = offset;

        // Down
        int down = center + gs.cols;
        if (down > gs.rows * gs.cols - offset) return;//down = gs.rows * gs.cols - offset;
        int down5 = center + gs.cols*5;
        if (down5 > gs.rows * gs.cols - offset) return;//down5 = gs.rows * gs.cols - offset;
        int down9 = center + gs.cols*15;
        if (down9 > gs.rows * gs.cols - offset) return;//down9 = gs.rows * gs.cols - offset;   
        int down2l = center + gs.cols*2 - 1;
        if (down2l > gs.rows * gs.cols - offset) return;//down2l = gs.rows * gs.cols - offset;
        int down2r = center + gs.cols*2 + 1;
        if (down2r > gs.rows * gs.cols - offset) return;//down2r = gs.rows * gs.cols - offset;
        int down1l = center + gs.cols - 2;
        if (down1l > gs.rows * gs.cols - offset) return;//down1l = gs.rows * gs.cols - offset;
        int down1r = center + gs.cols + 2;
        if (down1r > gs.rows * gs.cols - offset) return;//down1r = gs.rows * gs.cols - offset;

        // Right
        int right = center + 1;
        if (right > gs.rows * gs.cols - offset) return;//right = gs.rows * gs.cols - offset;
        int right5 = center + 5;
        if (right5 > gs.rows * gs.cols - offset) return;//right5 = gs.rows * gs.cols - offset;
        int right9 = center + 15;
        if (right9 > gs.rows * gs.cols - offset) return;//right9 = gs.rows * gs.cols - offset;


        for (int i = 0; i < gs.lines->FoodNumber; i++)
        {
            // curandState localState = gs.cs[center];
            //  curand_init(clock64(), p.y, p.x, &localState);
            float rand = curand_between(&localState, 0.0f, 8); // get number from 0~sum
            gs.cs[center] = localState;
            float rand3 = curand_between(&localState, 0.0f, 3); // get number from 0~3
            gs.cs[center] = localState;
            // printf("rand: %d, id: %d\n", rand, center);
                if (rand < 1)
                {
                    if (rand3<1)        calOnlookerBees_cu_best<T>(gs, p, up, i, p.x, p.y-1);
                    else if(rand3 < 2)  calOnlookerBees_cu_best<T>(gs, p, up5, i, p.x, p.y - 5);
                    else                calOnlookerBees_cu_best<T>(gs, p, up9, i, p.x, p.y -9);
                }
                else if (rand < 2)
                {
                    if (rand3<1)        calOnlookerBees_cu_best<T>(gs, p, down, i, p.x, p.y+1);
                    else if(rand3 < 2)  calOnlookerBees_cu_best<T>(gs, p, down5, i, p.x, p.y+5);
                    else                calOnlookerBees_cu_best<T>(gs, p, down9, i, p.x, p.y+9);
                }
                else if (rand < 3)
                {
                    if (rand3<1)        calOnlookerBees_cu_best<T>(gs, p, left, i,p.x-1, p.y);
                    else if(rand3 < 2)  calOnlookerBees_cu_best<T>(gs, p, left5, i,p.x-5, p.y);
                    else                calOnlookerBees_cu_best<T>(gs, p, left9, i,p.x-9, p.y);
                }
                else if (rand < 4)
                {
                    if (rand3<1)        calOnlookerBees_cu_best<T>(gs, p, right, i,p.x+1, p.y);
                    else if(rand3 < 2)  calOnlookerBees_cu_best<T>(gs, p, right5, i,p.x+5, p.y);
                    else                calOnlookerBees_cu_best<T>(gs, p, right9, i,p.x+9, p.y);
                }
                else if (rand < 5)
                {
                    if (rand3<1.5)        calOnlookerBees_cu_best<T>(gs, p, up2l, i, p.x - 1, p.y-2);
                    else                  calOnlookerBees_cu_best<T>(gs, p, up1l, i, p.x -2, p.y-1);                    
                }
                else if (rand < 6)
                {
                    if (rand3<1.5)        calOnlookerBees_cu_best<T>(gs, p, up2r, i, p.x + 1, p.y -2);
                    else                  calOnlookerBees_cu_best<T>(gs, p, up1r, i, p.x + 2, p.y-1);                    
                }
                else if (rand < 7)
                {
                    if (rand3<1.5)        calOnlookerBees_cu_best<T>(gs, p, down2l, i, p.x-1, p.y+2);
                    else                  calOnlookerBees_cu_best<T>(gs, p, down1l, i, p.x-2, p.y+1);                    
                }
                else if (rand <= 8)
                {
                    if (rand3<1.5)        calOnlookerBees_cu_best<T>(gs, p, down2r, i, p.x+1, p.y+2);
                    else                  calOnlookerBees_cu_best<T>(gs, p, down1r, i, p.x+2, p.y+1);                    
                }
        }
        sort_large_by_fitness_1d(gs.lines->fitness_1d, gs.lines->foods_1d, gs.lines->FoodNumber, center);
       

    }
}




/**
 * @brief Every pixel has <FoodNumber> sources.
 * For every source, randomly choose one param from four params and find a neighbour different from current source.
 * change one param.
 * randomly change that param by probability and calculate the cost and Fitness.
 * if the new solution is better, update it.
 *
 * @param gs GlobalState &gs
 * @param iter iterations
 */
template <typename T>
__global__ void SendEmployedBees_cu(GlobalState &gs, int iter)
{

    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= gs.cols - 2)
        return; 
    if (p.y >= gs.rows - 2)
        return; 
    const int center = p.y * gs.cols + p.x;
    
    


    // Employed Bee Phase
    float r=0.1f; // random number
    // neighbour corresponds to k in equation v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij})*/
    int neighbour = 0;
    float param2change = 0.0; // param2change choose one param to change.
    
    curandState localState = gs.cs[center];
    for (int i = 0; i < gs.lines->FoodNumber; i++) // 
    {
        // The parameter to be changed is determined randomly
        // a random number in the range [0,1)
        r = curand_between(&localState, 0.0f, 2.0f); // generate random number between 0 and 2.
        gs.cs[center] = localState;

        param2change = r; // Demension = 4

        // A randomly chosen solution is used in producing a mutant solution of the solution i
        r = curand_between(&localState, 0.0f, 1.0f); // generate random number between 0 and 1.
        gs.cs[center] = localState;

        neighbour = (int)(r * gs.lines->FoodNumber); // gs.lines->FoodNumber

        

        // Randomly selected solution must be different from the solution i*/
        while (neighbour == i)
        {
            r = curand_between(&localState, 0.0f, 1.0f); // generate random number between 0 and 1.
            gs.cs[center] = localState;

            neighbour = (int)(r * gs.lines->FoodNumber);
        }

        
        float4 solution = gs.lines->foods_1d[idx(center, i, gs.lines->FoodNumber)];

        // v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij})
        r = curand_between(&localState, 0.0f, 1.0f); // generate random number between 0 and 1.
        
        if (param2change < 0.3333333f)
        {
            solution.x = gs.lines->foods_1d[idx(center, i, gs.lines->FoodNumber)].x + (gs.lines->foods_1d[idx(center, i, gs.lines->FoodNumber)].x - gs.lines->foods_1d[idx(center, neighbour, gs.lines->FoodNumber)].x) * (r - 0.5) * 2;
        }
        else if (param2change < 2 * 0.3333333f)
        {
            solution.y = gs.lines->foods_1d[idx(center, i, gs.lines->FoodNumber)].y + (gs.lines->foods_1d[idx(center, i, gs.lines->FoodNumber)].y - gs.lines->foods_1d[idx(center, neighbour, gs.lines->FoodNumber)].y) * (r - 0.5) * 2;
        }
        else if (param2change < 3 * 0.3333333f)
        {
            solution.z = gs.lines->foods_1d[idx(center, i, gs.lines->FoodNumber)].z + (gs.lines->foods_1d[idx(center, i, gs.lines->FoodNumber)].z - gs.lines->foods_1d[idx(center, neighbour, gs.lines->FoodNumber)].z) * (r - 0.5) * 2;
        }
        else
        {
            solution.w = gs.lines->foods_1d[idx(center, i, gs.lines->FoodNumber)].w + (gs.lines->foods_1d[idx(center, i, gs.lines->FoodNumber)].w - gs.lines->foods_1d[idx(center, neighbour, gs.lines->FoodNumber)].w) * (r - 0.5) * 2;
        }

        float4 viewVector;
        getViewVector_cu(&viewVector, gs.cameras->cameras[gs.current_image], p);
        NormalizeVec3(&solution);
        vecOnHemisphere_cu(&solution, viewVector);
        
        float D_lower_bound = getD_cu(solution, p, gs.depth_lower_bound, gs.cameras->cameras[gs.current_image]); // Transform depth to the distance between center and the plane
        float D_upper_bound = getD_cu(solution, p, gs.depth_upper_bound, gs.cameras->cameras[gs.current_image]); // Transform depth to the distance between center and the plane


        if (solution.w < D_lower_bound)
            solution.w = D_lower_bound;
        if (solution.w > D_upper_bound)
            solution.w = D_upper_bound;


        float ObjValSol = functionABC_singlepoint<T>(p.x, p.y, solution, gs, false);
        float FitnessSol = ObjValSol >= 0 ? (1 / (ObjValSol + 1)) : (1 - ObjValSol);

        // a greedy selection is applied between the current solution i and its mutant
        if (FitnessSol >  gs.lines->fitness_1d[idx(center, i, gs.lines->FoodNumber)])
        {
            // If the mutant solution is better than the current solution i,
            // replace the solution with the mutant and reset the trial counter of solution i.
            gs.lines->trial_1d[idx(center, i, gs.lines->FoodNumber)] = 0;
            gs.lines->foods_1d[idx(center, i, gs.lines->FoodNumber)] = solution;
            gs.lines->fitness_1d[idx(center, i, gs.lines->FoodNumber)] = FitnessSol;
        }
        else
        { // if the solution i can not be improved, increase its trial counter
            gs.lines->trial_1d[idx(center, i, gs.lines->FoodNumber)] = gs.lines->trial_1d[idx(center, i, gs.lines->FoodNumber)] + 1;
        }
        sort_large_by_fitness_1d(gs.lines->fitness_1d, gs.lines->foods_1d, gs.lines->FoodNumber, center);
    }
    
}
// end of employed bee phase
template <typename T>
__global__ void SendEmployedBees_cu_one(GlobalState &gs, int iter)
{

    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= gs.cols -2)
        return; 
    if (p.y >= gs.rows -2)
        return; 
    const int center = p.y * gs.cols + p.x;

    // Employed Bee Phase
    float r=0.1f; // random number
    // neighbour corresponds to k in equation v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij})*/
    int neighbour = 0;
    float param2change = 0.0; // param2change choose one param to change.
    
    curandState localState = gs.cs[center];
    for (int i = 0; i < gs.lines->FoodNumber; i++)
    {
        // The parameter to be changed is determined randomly
        // a random number in the range [0,1)
        r = curand_between(&localState, 0.0f, 2.0f); // generate random number between 0 and 2.
        gs.cs[center] = localState;

        param2change = r; // Demension = 4

        // A randomly chosen solution is used in producing a mutant solution of the solution i
        // r = curand_between(&localState, 0.0f, 1.0f); // generate random number between 0 and 1.
        // gs.cs[center] = localState;

        neighbour = 0;//(int)(r * gs.lines->FoodNumber); // gs.lines->FoodNumber

        

        // Randomly selected solution must be different from the solution i*/
        // while (neighbour == i)
        // {
        //     r = curand_between(&localState, 0.0f, 1.0f); // generate random number between 0 and 1.
        //     gs.cs[center] = localState;

        //     neighbour = (int)(r * gs.lines->FoodNumber);
        // }

        
        float4 solution = gs.lines->foods_1d[idx(center, i, gs.lines->FoodNumber)];
        float D_lower_bound_old = getD_cu(solution, p, gs.depth_lower_bound, gs.cameras->cameras[gs.current_image]); // Transform depth to the distance between center and the plane
        float D_upper_bound_old = getD_cu(solution, p, gs.depth_upper_bound, gs.cameras->cameras[gs.current_image]); // Transform depth to the distance between center and the plane

        // v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij})
        r = curand_between(&localState, 0.0f, 1.0f); // generate random number between 0 and 1.
        float var = 0;
        if (param2change < 0.3333333f)
        {
            var = 1 - fabs(solution.x);
            solution.x = gs.lines->foods_1d[idx(center, i, gs.lines->FoodNumber)].x + (var) * (r - 0.5);
        }
        else if (param2change < 2 * 0.3333333f)
        {
            var = 1 - fabs(solution.y);
            solution.y = gs.lines->foods_1d[idx(center, i, gs.lines->FoodNumber)].y + (var) * (r - 0.5);
        }
        else if (param2change < 3 * 0.3333333f)
        {
            var = 1 - fabs(solution.z);
            solution.z = gs.lines->foods_1d[idx(center, i, gs.lines->FoodNumber)].z + (var) * (r - 0.5);
        }
        else
        {
            if (solution.w - D_lower_bound_old > D_upper_bound_old - solution.w)   // closer to upper bound
            {
                var = D_upper_bound_old - solution.w;
            }
            else
            {
                var = solution.w - D_lower_bound_old;
            }

            solution.w = gs.lines->foods_1d[idx(center, i, gs.lines->FoodNumber)].w + (var) * (r - 0.5);
        }

        float4 viewVector;
        getViewVector_cu(&viewVector, gs.cameras->cameras[gs.current_image], p);
        NormalizeVec3(&solution);
        vecOnHemisphere_cu(&solution, viewVector);
        
        float D_lower_bound = getD_cu(solution, p, gs.depth_lower_bound, gs.cameras->cameras[gs.current_image]); // Transform depth to the distance between center and the plane
        float D_upper_bound = getD_cu(solution, p, gs.depth_upper_bound, gs.cameras->cameras[gs.current_image]); // Transform depth to the distance between center and the plane


        if (solution.w < D_lower_bound)
            solution.w = D_lower_bound;
        if (solution.w > D_upper_bound)
            solution.w = D_upper_bound;


        float ObjValSol = functionABC_singlepoint<T>(p.x, p.y, solution, gs, false);
        float FitnessSol = ObjValSol >= 0 ? (1 / (ObjValSol + 1)) : (1 - ObjValSol);

        // a greedy selection is applied between the current solution i and its mutant
        if (FitnessSol >  gs.lines->fitness_1d[idx(center, i, gs.lines->FoodNumber)])
        {
            // If the mutant solution is better than the current solution i,
            // replace the solution with the mutant and reset the trial counter of solution i.
            gs.lines->trial_1d[idx(center, i, gs.lines->FoodNumber)] = 0;
            gs.lines->foods_1d[idx(center, i, gs.lines->FoodNumber)] = solution;
            gs.lines->fitness_1d[idx(center, i, gs.lines->FoodNumber)] = FitnessSol;
        }
        else
        { // if the solution i can not be improved, increase its trial counter
            gs.lines->trial_1d[idx(center, i, gs.lines->FoodNumber)] = gs.lines->trial_1d[idx(center, i, gs.lines->FoodNumber)] + 1;
        }
        sort_large_by_fitness_1d(gs.lines->fitness_1d, gs.lines->foods_1d, gs.lines->FoodNumber, center);
    }
    
}


__global__ void ambc_compute_disp(GlobalState &gs)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= gs.cols)
        return;
    if (p.y >= gs.rows)
        return;

    const int center = p.y * gs.cols + p.x;
    float4 norm =  gs.lines->foods_1d[idx(center, 0, gs.lines->FoodNumber)];

    float4 norm_transformed;
    // Transform back normal to world coordinate
    float4 viewVector; 
    getViewVector_cu(&viewVector, gs.cameras->cameras[gs.current_image], p);
    vecOnHemisphere_cu ( &norm, viewVector );
    

    matvecmul4(gs.cameras->cameras[gs.current_image].R_orig_inv, norm, (&norm_transformed));
    
    if ( gs.lines->fitness_1d[idx(center, 0, gs.lines->FoodNumber)] != MAXCOST) 
    {
        norm_transformed.w = getDepthFromPlane3_cu(gs.cameras->cameras[gs.current_image], norm, norm.w, p);
    }else
        norm_transformed.w = 0;
    // temp1 = -norm.w*gs.cameras->cameras[REFERENCE].fx/(norm.x*(p.x-gs.cameras->cameras[REFERENCE].K[2]) + (norm.y*(p.y-gs.cameras->cameras[0].K[5]))*gs.cameras->cameras[0].alpha + norm.z*gs.cameras->cameras[0].fx);
    gs.lines->norm4[center] = norm_transformed;

    return;
}

template <typename T>
void ambc(GlobalState &gs)
{
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1); 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // int SHARED_SIZE_W_host;
// #ifndef SHARED_HARDCODED                        
    int blocksize_w = gs.params->box_hsize + 1; 
    int blocksize_h = gs.params->box_vsize + 1; 
    WIN_RADIUS_W = (blocksize_w) / (2);         
    WIN_RADIUS_H = (blocksize_h) / (2);        

    int BLOCK_W = 32;
    int BLOCK_H = (BLOCK_W / 2);                    
    // TILE_W = BLOCK_W;                                
    // TILE_H = BLOCK_H * 2;                            
    // SHARED_SIZE_W_m = (TILE_W + WIN_RADIUS_W * 2);   
    // SHARED_SIZE_H = (TILE_H + WIN_RADIUS_H * 2);     
    // SHARED_SIZE = (SHARED_SIZE_W_m * SHARED_SIZE_H); 
    // cudaMemcpyToSymbol(SHARED_SIZE_W, &SHARED_SIZE_W_m, sizeof(SHARED_SIZE_W_m));
    // SHARED_SIZE_W_host = SHARED_SIZE_W_m;
// #else
                                                     // SHARED_SIZE_W_host = SHARED_SIZE;
// #endif
    // int shared_size_host = SHARED_SIZE; // 1936

    dim3 grid_size;
    grid_size.x = (gs.cols + BLOCK_W - 1) / BLOCK_W; 
    grid_size.y = ((gs.rows / 2) + BLOCK_H - 1) / BLOCK_H; 
    dim3 block_size;
    block_size.x = BLOCK_W; 
    block_size.y = BLOCK_H;

    dim3 grid_size_initrand;
    grid_size_initrand.x = (gs.cols + 16 - 1) / 16; 
    grid_size_initrand.y = (gs.rows + 16 - 1) / 16;
    dim3 block_size_initrand;
    block_size_initrand.x = 32;
    block_size_initrand.y = 16;

    size_t avail;
    size_t total;
    cudaMemGetInfo(&avail, &total);
    size_t used = total - avail;

    printf("Device memory used: %fMB\n", used / 1000000.0f);
    printf("Blocksize is %dx%d\n", gs.params->box_hsize, gs.params->box_vsize); // 11x11

    printf("grid_size_initrand.x is %d\n", grid_size_initrand.x);   // 40
    printf("block_size_initrand.x is %d\n", block_size_initrand.x); // 16

    double time1=0.0f, time2=0.0f, time3=0.0f, time4=0.0f, time5=0.0f, time6 = 0.0f;
    float global_mean = 0.0;

    
 
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess)
    //     printf("Error1787: %s\n", cudaGetErrorString(err));
    // else
    //     printf("No Error in cuda 1787\n");
  
    
    int iters = 0;
    if (gs.cycles == 0)
    {
        iters = gs.params->warm_up_iters;
        ambc_init_cu<T><<<grid_size_initrand, block_size_initrand>>>(gs); // Initialization
    }
    else
    {
        ambc_calc_cost<T><<<grid_size_initrand, block_size_initrand>>>(gs);
        iters = gs.params->iterations;
    }
    cudaDeviceSynchronize();

    

    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err1));
    else
        printf("No Error in cuda\n");

   
    for (int it = 0; it < iters; it++)
    {
        cudaDeviceSynchronize();
        gs.lines->iters = it;
        time1 = get_time();
        if (gs.lines->FoodNumber == 1)
        {
            SendEmployedBees_cu_one<T><<<grid_size_initrand, block_size_initrand>>>(gs, it); // SendEmployedBees (Local solution disturbing)
        } 
        else
        {
            SendEmployedBees_cu<T><<<grid_size_initrand, block_size_initrand>>>(gs, it); // SendEmployedBees (Local solution disturbing)
        }
        cudaDeviceSynchronize();
        

        time2 = get_time();
        SendOnlookerBees_cu_black_best<T><<<grid_size, block_size>>>(gs);   // SendOnlookerBees (Spatial propagtaion) half of the image (checkerboard scheme)
        cudaDeviceSynchronize();
       

        time3 = get_time();
        SendOnlookerBees_cu_red_best<T><<<grid_size, block_size>>>(gs);     // SendOnlookerBees (Spatial propagtaion) half of the image (checkerboard scheme)
        cudaDeviceSynchronize();
 
        time4 = get_time();
        if (gs.lines->FoodNumber > 1)
        {
            SendScoutBees_cu<T><<<grid_size_initrand, block_size_initrand>>>(gs, it); // SendScoutBees
        }
        cudaDeviceSynchronize();

        time5 = get_time(); 

        


        global_mean = 0.0f;
        for (int i = 0; i < gs.cols * gs.rows; i++)
        {
            global_mean +=  gs.lines->fitness_1d[idx(i, 0, gs.lines->FoodNumber)];
        }
        global_mean = global_mean / (gs.cols * gs.rows);
        time6 = get_time(); 
        cudaDeviceSynchronize();  
        printf("cycle%d,img%d, iterations:%d, time12: %f, time23: %f, time34: %f, time45: %f, time56: %f, total time/iter: %f, global_mean: %f\n", 
                                            gs.cycles, 
                                            gs.current_image, 
                                            it + 1, 
                                            time2 - time1, 
                                            time3 - time2, 
                                            time4 - time3, 
                                            time5 - time4, 
                                            time5 - time1, 
                                            time6 - time5,
                                            global_mean);
        cudaDeviceSynchronize();                                   
    }

    // SaveNumBests<T><<<grid_size_initrand, block_size_initrand>>>(gs); 
    cudaDeviceSynchronize();
    ambc_compute_disp<<<grid_size_initrand, block_size_initrand>>>(gs);     // Transform solution to depth and normal at world coordinate system.
    cudaDeviceSynchronize();


    // cudaDeviceReset();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\tTotal time needed for computation: %f seconds\n", milliseconds / 1000.f);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    else
        printf("No Error in cuda\n");

    // cudaFree(&gs.cs);
    printf("ambc ends\n");
}

int runcuda(GlobalState &gs)
{
    printf("runcuda-after\n");
    if (gs.params->color_processing)
        ambc<float4>(gs);
    else {
        printf("runcuda-after2\n");
        ambc<float>(gs);
    }
    return 0;
}