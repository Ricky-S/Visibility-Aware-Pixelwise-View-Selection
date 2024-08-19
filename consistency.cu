/* vim: ft=cpp
 * */

//#include <helper_math.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include "globalstate.h"
#include "algorithmparameters.h"
#include "cameraparameters.h"
#include "linestate.h"
#include "config.h"

#include <vector_types.h> // float4
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "vector_operations.h"
#include "point_cloud_list.h"
// #include "config.h"

/* compute depth value from disparity or disparity value from depth
 * Input:  f         - focal length in pixel
 *         baseline  - baseline between cameras (in meters)
 *         d - either disparity or depth value
 * Output: either depth or disparity value
 */
__device__ float disparityDepthConversion_cu2(const float &f, const Camera_cu &cam_ref, const Camera_cu &cam, const float &d)
{
    float baseline = l2_float4(cam_ref.C4 - cam.C4);
    return f * baseline / d;
}



__device__ void get3Dpoint_cu(float4 *__restrict__ ptX, const Camera_cu &cam, const int2 &p, const float &depth)
{
    // in case camera matrix is not normalized: see page 162, then depth might not be the real depth but w and depth needs to be computed from that first
    const float4 pt = make_float4(
        depth * (float)p.x - cam.P_col34.x,
        depth * (float)p.y - cam.P_col34.y,
        depth - cam.P_col34.z,
        0);

    matvecmul4(cam.M_inv, pt, ptX);
}

/* get angle between two vectors in 3D
 * Input: v1,v2 - vectors
 * Output: angle in radian
 */
__device__ float getAngle_cu(const float4 &v1, const float4 &v2)
{
    float angle = acosf(dot4(v1, v2));
    // if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
    if (angle != angle)
        return 0.0f;
    // if ( acosf ( v1.dot ( v2 ) ) != acosf ( v1.dot ( v2 ) ) )
    // cout << acosf ( v1.dot ( v2 ) ) << " / " << v1.dot ( v2 )<< " / " << v1<< " / " << v2 << endl;
    return angle;
}

__device__ void project_on_camera(const float4 &X, const Camera_cu &cam, float2 *pt, float *depth)
{
    float4 tmp = make_float4(0, 0, 0, 0);
    matvecmul4P(cam.P, X, (&tmp));
    pt->x = tmp.x / tmp.z;
    pt->y = tmp.y / tmp.z;
    *depth = tmp.z;
}

__device__ void NormalizeVec31 (float4 *vec)
{
    const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
    const float inverse_sqrt = rsqrtf (normSquared);
    vec->x *= inverse_sqrt;
    vec->y *= inverse_sqrt;
    vec->z *= inverse_sqrt;
}


/**
 * @brief AMBC Weak Consistency Check. For each point of the reference camera compute the 3d point corresponding to the depth.
 *        Then reproject the 3d point to all the selectedViews for corresponding depth.
 *        Each selected view is considered consistent if:
 *        - Projected depths & angle of normal of selectedViews does not differ more than preset threshold.
 *        Each selected view is considered occluded if:
 *        - Projected depths of selectedViews satisfies the strong consistency check and less than corresponding depth of 3d point.
 *        Create a point if number_consistent & number_occluded is larger than preset threshold.
 *
 * @param gs
 * @param ref_camera
 * @return __global__
 */
__global__ void ambc_weak(GlobalState &gs, int ref_camera)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    const int cols = gs.cols;
    const int rows = gs.rows;

    if (p.x >= cols)
        return;
    if (p.y >= rows)
        return;

    const int center = p.y * cols + p.x;

    const CameraParameters_cu &camParams = *(gs.cameras);

    // if (gs.lines_consistency[ref_camera].used_pixels[center] == 1)
    //     return;

    const float4 normal = tex2D<float4>(gs.normals_depths[ref_camera], p.x, p.y );

    float X_depth = normal.w;

    float4 X;
    get3Dpoint_cu(&X, camParams.cameras[ref_camera], p, X_depth);
    int2 used_list[MAX_IMAGES];
    float4 consistent_X = X;
    float4 consistent_normal = normal;
    float consistent_texture = tex2D<float>(gs.imgs[ref_camera], p.x + 0.5f, p.y + 0.5f);
    int number_consistent = 0;
    int number_occluded = 0; // 只能被符合strong consistency的pixel遮挡
    int number_overad = 0;


    if (true)
    {
        float4 v_ref; // view vector between the 3D point and the reference camera
        float4 v_source;  // view vector between the 3D point and the souce camera
        // vec_ref.x = cpc.cameras[current_image].C4.x - forward_point.x;
        // vec_ref.y = cpc.cameras[current_image].C4.y - forward_point.y;
        // vec_ref.z = cpc.cameras[current_image].C4.z - forward_point.z;
        v_ref.x = normal.x;
        v_ref.y = normal.y;
        v_ref.z = normal.z;
        NormalizeVec31(&v_ref);
        v_source.x = camParams.cameras[ref_camera].C4.x - X.x;
        v_source.y = camParams.cameras[ref_camera].C4.y - X.y;
        v_source.z = camParams.cameras[ref_camera].C4.z - X.z;
        NormalizeVec31(&v_source);
        float rad1 = acosf ( v_ref.x * v_source.x + v_ref.y * v_source.y + v_ref.z * v_source.z);
        float maximum_angle_radians = 70 * M_PI / 180.0f;

        
        
        if (rad1 > maximum_angle_radians )
        {
            return;
        }    
    }



    for (int i = 0; i < camParams.selectedViewsNum[ref_camera]; i++)
    {

        int idxCurr = camParams.selectedViews[ref_camera][i];
        used_list[idxCurr].x = -1;
        used_list[idxCurr].y = -1;
        if (idxCurr == ref_camera)
            continue;

        // Project 3d point X on camera idxCurr
        float2 source_pixel;
        project_on_camera(X, camParams.cameras[idxCurr], &source_pixel, &X_depth);

        // Boundary check
        if (source_pixel.x >= 5 &&
            source_pixel.x < cols -5 &&
            source_pixel.y >= 5 &&
            source_pixel.y < rows -5)
        {

            // Compute interpolated depth and normal for source_pixel w.r.t. camera ref_camera
            float4 source; // first 3 components normal, fourth depth
            source = tex2D<float4>(gs.normals_depths[idxCurr], static_cast<int>(source_pixel.x + 0.5f), static_cast<int>(source_pixel.y + 0.5f));

            const float depth_disp = disparityDepthConversion_cu2  ( camParams.cameras[ref_camera].f, camParams.cameras[ref_camera], camParams.cameras[idxCurr], X_depth );
            const float source_disp = disparityDepthConversion_cu2 ( camParams.cameras[ref_camera].f, camParams.cameras[ref_camera], camParams.cameras[idxCurr], source.w );
            // First consistency check on depth

            if (fabsf(depth_disp - source_disp) < gs.params->depthThresh)
            {

                float angle = getAngle_cu(source, normal); // extract normal
                if (angle < gs.params->normalThresh)
                {
                    number_consistent++;
                    int2 tmp_p = make_int2((int)(source_pixel.x + 0.5), (int)(source_pixel.y + 0.5));
                    float4 tmp_X; // 3d point of consistent point on other view
                    get3Dpoint_cu(&tmp_X, camParams.cameras[idxCurr], tmp_p, source.w);

                    consistent_X = consistent_X + tmp_X;
                    consistent_normal = consistent_normal + source;
                    if (gs.params->saveTexture)
                        consistent_texture = consistent_texture + tex2D<float>(gs.imgs[idxCurr], source_pixel.x + 0.5f, source_pixel.y + 0.5f);

                    // Save the point for later check
                    used_list[idxCurr].x = (int)tmp_p.x;
                    used_list[idxCurr].y = (int)tmp_p.y;
                    
                    continue;
                }
            }
            
            if (X_depth - source.w > gs.params->depthThresh) // the pixel is occluded
            {
                if (gs.lines_consistency[idxCurr].used_pixels[((int)(source_pixel.y + 0.5f)) * cols + (int)(source_pixel.x + 0.5f)] >= 1) // occluded pixel satisfies strong consistency check
                    number_occluded++;
                    continue;
            }

            float4 vec_ref; // view vector between the 3D point and the reference camera
            float4 vec_source;  // view vector between the 3D point and the souce camera
            // vec_ref.x = cpc.cameras[current_image].C4.x - forward_point.x;
            // vec_ref.y = cpc.cameras[current_image].C4.y - forward_point.y;
            // vec_ref.z = cpc.cameras[current_image].C4.z - forward_point.z;
            vec_ref.x = normal.x;
            vec_ref.y = normal.y;
            vec_ref.z = normal.z;
            NormalizeVec31(&vec_ref);
            vec_source.x = camParams.cameras[idxCurr].C4.x - X.x;
            vec_source.y = camParams.cameras[idxCurr].C4.y - X.y;
            vec_source.z = camParams.cameras[idxCurr].C4.z - X.z;
            NormalizeVec31(&vec_source);
            float rad = acosf ( vec_ref.x * vec_source.x + vec_ref.y * vec_source.y + vec_ref.z * vec_source.z);
            float maximum_angle_radians = 70 * M_PI / 180.0f;

            
            
            if (rad > maximum_angle_radians )
            {
                number_overad++;
            }    
        }
        else
            continue;
    }

    // Average normals and points
    consistent_X = consistent_X / ((float)number_consistent + 1.0f);
    consistent_normal = consistent_normal / ((float)number_consistent + 1.0f);
    consistent_texture = consistent_texture / ((float)number_consistent + 1.0f);


    // (optional) save texture
    int num_c = camParams.selectedViewsNum[ref_camera] * 0.5 + 0.5;
    // if (number_consistent >= gs.params->number_consistent_input && number_consistent + number_overad + number_occluded >= max(camParams.selectedViewsNum[ref_camera] - gs.params->numConsistentThresh, 2))
    if (number_consistent >= gs.params->number_consistent_input && number_consistent + number_overad + number_occluded >= num_c)
    {

        
        if (gs.lines_consistency[ref_camera].used_pixels[center] < 1) // hardcoded for middlebury TODO FIX
        {
            gs.pc->points[center].coord = consistent_X;
            gs.pc->points[center].normal = consistent_normal;

            if (gs.params->saveTexture)
                gs.pc->points[center].texture = consistent_texture;
        }
        gs.lines_consistency[ref_camera].used_pixels[center] = 2;
        for (int i = 0; i < camParams.selectedViewsNum[ref_camera]; i++)
        {
            int idxCurr = camParams.selectedViews[ref_camera][i];
            // printf("%d\n", number_consistent);
            if (used_list[idxCurr].x == -1) 
            {

                // printf("%f\n", gs.lines_consistency[idxCurr].suggestions[c].w);
            }
            else
            {
                gs.lines_consistency[idxCurr].used_pixels[used_list[idxCurr].x + used_list[idxCurr].y * cols] = 2;
            }
        }
    }

    return;
}




__global__ void ambc_fusibile(GlobalState &gs, int ref_camera)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    const int cols = gs.cols;
    const int rows = gs.rows;

    if (p.x >= cols)
        return;
    if (p.y >= rows)
        return;

    const int center = p.y * cols + p.x;

    const CameraParameters_cu &camParams = *(gs.cameras);

    if (gs.lines_consistency[ref_camera].used_pixels[center] == 1)
        return;

    const float4 normal = tex2D<float4>(gs.normals_depths[ref_camera], p.x , p.y );
    // printf("Normal is %f %f %f\nDepth is %f\n", normal.x, normal.y, normal.z, normal.w);
    /*
     * For each point of the reference camera compute the 3d position corresponding to the corresponding depth.
     * Create a point only if the following conditions are fulfilled:
     * - Projected depths of other cameras does not differ more than gs.params.depthThresh
     * - Angle of normal does not differ more than gs.params.normalThresh
     */
    float depth = normal.w;

    float4 X;
    get3Dpoint_cu(&X, camParams.cameras[ref_camera], p, depth);

    float4 consistent_X = X;
    float4 consistent_normal = normal;
    float consistent_texture = tex2D<float>(gs.imgs[ref_camera], p.x + 0.5f, p.y + 0.5f);
    int number_consistent = 0;
    // int2 used_list[camParams.viewSelectionSubsetNumber];
    int2 used_list[MAX_IMAGES];


    if (true)
    {
        float4 v_ref; // view vector between the 3D point and the reference camera
        float4 v_source;  // view vector between the 3D point and the souce camera
        v_ref.x = normal.x;
        v_ref.y = normal.y;
        v_ref.z = normal.z;
        NormalizeVec31(&v_ref);
        v_source.x = camParams.cameras[ref_camera].C4.x - X.x;
        v_source.y = camParams.cameras[ref_camera].C4.y - X.y;
        v_source.z = camParams.cameras[ref_camera].C4.z - X.z;
        NormalizeVec31(&v_source);
        float rad1 = acosf ( v_ref.x * v_source.x + v_ref.y * v_source.y + v_ref.z * v_source.z);
        float maximum_angle_radians = 70 * M_PI / 180.0f;

        
        
        if (rad1 > maximum_angle_radians )
        {
            return;
        }    
    }


    for (int i = 0; i < gs.num_images; i++)
    {

        int idxCurr = i;
        used_list[idxCurr].x = -1;
        used_list[idxCurr].y = -1;
        if (idxCurr == ref_camera)
            continue;

        // Project 3d point X on camera idxCurr
        float2 source_pixel;
        project_on_camera(X, camParams.cameras[idxCurr], &source_pixel, &depth);

        // Boundary check
        if (source_pixel.x >= 5 &&
            source_pixel.x < cols-5 &&
            source_pixel.y >= 5 &&
            source_pixel.y < rows-5)
        {
            // printf("Boundary check passed\n");

            // Compute interpolated depth and normal for source_pixel w.r.t. camera ref_camera
            float4 source; // first 3 components normal, fourth depth
            // source = tex2D<float4>(gs.normals_depths[idxCurr], (int)(source_pixel.x + 0.5f) +0.5f , (int)(source_pixel.y + 0.5f) + 0.5f);
            source = tex2D<float4>(gs.normals_depths[idxCurr], static_cast<int>(source_pixel.x + 0.5f), static_cast<int>(source_pixel.y + 0.5f) );

            const float depth_disp = disparityDepthConversion_cu2                ( camParams.cameras[ref_camera].f, camParams.cameras[ref_camera], camParams.cameras[idxCurr], depth );
            const float source_disp = disparityDepthConversion_cu2 ( camParams.cameras[ref_camera].f, camParams.cameras[ref_camera], camParams.cameras[idxCurr], source.w );
            // First consistency check on depth

            if (fabsf(depth_disp - source_disp) < gs.params->depthThresh)
            {

                float angle = getAngle_cu(source, normal); // extract normal
                if (angle < gs.params->normalThresh)
                {

                    /// All conditions met:
                    //  - average 3d points and normals
                    //  - save resulting point and normal
                    //  - (optional) average texture (not done yet)

                    int2 tmp_p = make_int2(static_cast<int>(source_pixel.x + 0.5f), static_cast<int>(source_pixel.y + 0.5f));
                    float4 tmp_X; // 3d point of consistent point on other view
                    get3Dpoint_cu(&tmp_X, camParams.cameras[idxCurr], tmp_p, source.w);

                    consistent_X = consistent_X + tmp_X;
                    consistent_normal = consistent_normal + source;
                    if (gs.params->saveTexture)
                        consistent_texture = consistent_texture + tex2D<float>(gs.imgs[idxCurr], source_pixel.x + 0.5f, source_pixel.y + 0.5f);

                    // Save the point for later check
                    used_list[idxCurr].x = static_cast<int>(source_pixel.x + 0.5f);
                    used_list[idxCurr].y = static_cast<int>(source_pixel.y + 0.5f);

                    number_consistent++;
                }
            }
        }
        else
            continue;
    }

    // Average normals and points
    consistent_X = consistent_X / ((float)number_consistent + 1.0f);
    consistent_normal = consistent_normal / ((float)number_consistent + 1.0f);
    consistent_texture = consistent_texture / ((float)number_consistent + 1.0f);

    // If at least numConsistentThresh point agree:
    // Create point
    // Save normal
    // (optional) save texture

    int num_c = gs.params->number_consistent_input;
    // if (number_consistent >= max(camParams.selectedViewsNum[ref_camera] - gs.params->numConsistentThresh - 2, gs.params->number_consistent_input))
   
    if (number_consistent >= num_c)
    {
        // printf("\tEnough consistent points!\nSaving point %f %f %f", consistent_X.x, consistent_X.y, consistent_X.z);

        //if (!gs.params->remove_black_background || consistent_texture > 15) // hardcoded for middlebury TODO FIX
        
       
        if (gs.lines_consistency[ref_camera].used_pixels[center] < 1)
        {
            gs.pc->points[center].coord = consistent_X;
            gs.pc->points[center].normal = consistent_normal;
            
            if (gs.params->saveTexture)
                gs.pc->points[center].texture = consistent_texture;
        }
            

        

        // Mark corresponding point on other views as "used"
        gs.lines_consistency[ref_camera].used_pixels[center] = 1;
        for (int i = 0; i < camParams.selectedViewsNum[ref_camera]; i++)
        {
            int idxCurr = camParams.selectedViews[ref_camera][i];
            // printf("%d\n", number_consistent);
            if (used_list[idxCurr].x == -1) 
            {

                // printf("%f\n", gs.lines_consistency[idxCurr].suggestions[c].w);
            }
            else
            {
                gs.lines_consistency[idxCurr].used_pixels[used_list[idxCurr].x + used_list[idxCurr].y * cols] = 1;
            }
        }
        
    }

    return;
}


__global__ void ambc_strong_new(GlobalState &gs, int ref_camera)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    const int cols = gs.cols;
    const int rows = gs.rows;

    if (p.x >= cols)
        return;
    if (p.y >= rows)
        return;

    const int center = p.y * cols + p.x;

    const CameraParameters_cu &camParams = *(gs.cameras);

    // if (gs.lines_consistency[ref_camera].used_pixels[center] == 1)
    //     return;

    const float4 normal = tex2D<float4>(gs.normals_depths[ref_camera], p.x , p.y );
    // printf("Normal is %f %f %f\nDepth is %f\n", normal.x, normal.y, normal.z, normal.w);
    /*
     * For each point of the reference camera compute the 3d position corresponding to the corresponding depth.
     * Create a point only if the following conditions are fulfilled:
     * - Projected depths of other cameras does not differ more than gs.params.depthThresh
     * - Angle of normal does not differ more than gs.params.normalThresh
     */
    float depth = normal.w;

    float4 X;
    get3Dpoint_cu(&X, camParams.cameras[ref_camera], p, depth);

    float4 consistent_X = X;
    float4 consistent_normal = normal;
    float consistent_texture = tex2D<float>(gs.imgs[ref_camera], p.x + 0.5f, p.y + 0.5f);
    int number_consistent = 0;
    // int2 used_list[camParams.viewSelectionSubsetNumber];
    int2 used_list[MAX_IMAGES];


    if (true)
    {
        float4 v_ref; // view vector between the 3D point and the reference camera
        float4 v_source;  // view vector between the 3D point and the souce camera
        v_ref.x = normal.x;
        v_ref.y = normal.y;
        v_ref.z = normal.z;
        NormalizeVec31(&v_ref);
        v_source.x = camParams.cameras[ref_camera].C4.x - X.x;
        v_source.y = camParams.cameras[ref_camera].C4.y - X.y;
        v_source.z = camParams.cameras[ref_camera].C4.z - X.z;
        NormalizeVec31(&v_source);
        float rad1 = acosf ( v_ref.x * v_source.x + v_ref.y * v_source.y + v_ref.z * v_source.z);
        float maximum_angle_radians = 70 * M_PI / 180.0f;

        
        
        if (rad1 > maximum_angle_radians )
        {
            return;
        }    
    }


    for (int i = 0; i < camParams.selectedViewsNum[ref_camera]; i++)
    {

        int idxCurr = camParams.selectedViews[ref_camera][i];
        used_list[idxCurr].x = -1;
        used_list[idxCurr].y = -1;
        if (idxCurr == ref_camera)
            continue;

        // Project 3d point X on camera idxCurr
        float2 source_pixel;
        project_on_camera(X, camParams.cameras[idxCurr], &source_pixel, &depth);

        // Boundary check
        if (source_pixel.x >= 5 &&
            source_pixel.x < cols-5 &&
            source_pixel.y >= 5 &&
            source_pixel.y < rows-5)
        {
            // printf("Boundary check passed\n");

            // Compute interpolated depth and normal for source_pixel w.r.t. camera ref_camera
            float4 source; // first 3 components normal, fourth depth
            // source = tex2D<float4>(gs.normals_depths[idxCurr], (int)(source_pixel.x + 0.5f) +0.5f , (int)(source_pixel.y + 0.5f) + 0.5f);
            source = tex2D<float4>(gs.normals_depths[idxCurr], static_cast<int>(source_pixel.x + 0.5f), static_cast<int>(source_pixel.y + 0.5f) );

            const float depth_disp = disparityDepthConversion_cu2                ( camParams.cameras[ref_camera].f, camParams.cameras[ref_camera], camParams.cameras[idxCurr], depth );
            const float source_disp = disparityDepthConversion_cu2 ( camParams.cameras[ref_camera].f, camParams.cameras[ref_camera], camParams.cameras[idxCurr], source.w );
            // First consistency check on depth

            if (fabsf(depth_disp - source_disp) < gs.params->depthThresh)
            {

                float angle = getAngle_cu(source, normal); // extract normal
                if (angle < gs.params->normalThresh)
                {

                    /// All conditions met:
                    //  - average 3d points and normals
                    //  - save resulting point and normal
                    //  - (optional) average texture (not done yet)

                    int2 tmp_p = make_int2(static_cast<int>(source_pixel.x + 0.5f), static_cast<int>(source_pixel.y + 0.5f));
                    float4 tmp_X; // 3d point of consistent point on other view
                    get3Dpoint_cu(&tmp_X, camParams.cameras[idxCurr], tmp_p, source.w);

                    consistent_X = consistent_X + tmp_X;
                    consistent_normal = consistent_normal + source;
                    if (gs.params->saveTexture)
                        consistent_texture = consistent_texture + tex2D<float>(gs.imgs[idxCurr], source_pixel.x + 0.5f, source_pixel.y + 0.5f);

                    // Save the point for later check
                    used_list[idxCurr].x = static_cast<int>(source_pixel.x + 0.5f);
                    used_list[idxCurr].y = static_cast<int>(source_pixel.y + 0.5f);

                    number_consistent++;
                }
            }
        }
        else
            continue;
    }

    // Average normals and points
    consistent_X = consistent_X / ((float)number_consistent + 1.0f);
    consistent_normal = consistent_normal / ((float)number_consistent + 1.0f);
    consistent_texture = consistent_texture / ((float)number_consistent + 1.0f);

    // If at least numConsistentThresh point agree:
    // Create point
    // Save normal
    // (optional) save texture

    int num_c = camParams.selectedViewsNum[ref_camera] * 0.5 + 0.5;
    // if (number_consistent >= max(camParams.selectedViewsNum[ref_camera] - gs.params->numConsistentThresh - 2, gs.params->number_consistent_input))
    if (number_consistent >= max(num_c - 2, gs.params->number_consistent_input))
    {

        for (int i = 0; i < camParams.selectedViewsNum[ref_camera]; i++)
        {
            int idxCurr = camParams.selectedViews[ref_camera][i];
            float2 source_pixel1;
            project_on_camera(consistent_X, camParams.cameras[idxCurr], &source_pixel1, &depth);
            // printf("%f\n", depth);
            if (source_pixel1.x <= 5 || source_pixel1.x >= cols - 5 || source_pixel1.y <= 5 || source_pixel1.y >= rows - 5)
                continue;
            // float4 target = gs.lines_consistency[idxCurr].suggestions[source_pixel.x + source_pixel.y * cols];
            int c = (int)(source_pixel1.x + 0.5f) + (int)(source_pixel1.y + 0.5f) * cols;
            if (gs.lines_consistency[idxCurr].suggestions[c].w == 0)
            {
                gs.lines_consistency[idxCurr].suggestions[c].x = consistent_normal.x;
                gs.lines_consistency[idxCurr].suggestions[c].y = consistent_normal.y;
                gs.lines_consistency[idxCurr].suggestions[c].z = consistent_normal.z;
                gs.lines_consistency[idxCurr].suggestions[c].w = depth;
                // gs.lines_consistency[gs.num_images].test_s[c]++;
            }
            // else
            // {
            //     gs.lines_consistency[idxCurr].suggestions_1[c].x = consistent_normal.x;
            //     gs.lines_consistency[idxCurr].suggestions_1[c].y = consistent_normal.y;
            //     gs.lines_consistency[idxCurr].suggestions_1[c].z = consistent_normal.z;
            //     gs.lines_consistency[idxCurr].suggestions_1[c].w = depth;
            //     // gs.lines_consistency[gs.num_images].test_s1[c]++;
            // }
        }
    }
 

    if (number_consistent >= num_c)
    {
        //if (!gs.params->remove_black_background || consistent_texture > 15) // hardcoded for middlebury TODO FIX
        
       
        if (gs.lines_consistency[ref_camera].used_pixels[center] < 1)
        {
            gs.pc->points[center].coord = consistent_X;
            gs.pc->points[center].normal = consistent_normal;
            
            if (gs.params->saveTexture)
                gs.pc->points[center].texture = consistent_texture;
        }
            

        

        // Mark corresponding point on other views as "used"
        gs.lines_consistency[ref_camera].used_pixels[center] = 1;
        for (int i = 0; i < camParams.selectedViewsNum[ref_camera]; i++)
        {
            int idxCurr = camParams.selectedViews[ref_camera][i];
            // printf("%d\n", number_consistent);
            if (used_list[idxCurr].x == -1) 
            {

                // printf("%f\n", gs.lines_consistency[idxCurr].suggestions[c].w);
            }
            else
            {
                gs.lines_consistency[idxCurr].used_pixels[used_list[idxCurr].x + used_list[idxCurr].y * cols] = 1;
            }
        }
        
    }

    return;
}

__global__ void ambc_strong(GlobalState &gs, int ref_camera)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    const int cols = gs.cols;
    const int rows = gs.rows;

    if (p.x >= cols)
        return;
    if (p.y >= rows)
        return;

    const int center = p.y * cols + p.x;

    const CameraParameters_cu &camParams = *(gs.cameras);

    if (gs.lines_consistency[ref_camera].used_pixels[center] >= 1)
        return;

    const float4 normal = tex2D<float4>(gs.normals_depths[ref_camera], p.x + 0.5f, p.y + 0.5f);
    // printf("Normal is %f %f %f\nDepth is %f\n", normal.x, normal.y, normal.z, normal.w);
    /*
     * For each point of the reference camera compute the 3d position corresponding to the corresponding depth.
     * Create a point only if the following conditions are fulfilled:
     * - Projected depths of other cameras does not differ more than gs.params.depthThresh
     * - Angle of normal does not differ more than gs.params.normalThresh
     */
    float depth = normal.w;

    float4 X;
    get3Dpoint_cu(&X, camParams.cameras[ref_camera], p, depth);

    float4 consistent_X = X;
    float4 consistent_normal = normal;
    float consistent_texture = tex2D<float>(gs.imgs[ref_camera], p.x + 0.5f, p.y + 0.5f);
    int number_consistent = 0;
    // int valid_count = 0;
    // int2 used_list[camParams.viewSelectionSubsetNumber];
    int2 used_list[MAX_IMAGES];
    for (int i = 0; i < camParams.selectedViewsNum[ref_camera]; i++)
    {

        int idxCurr = camParams.selectedViews[ref_camera][i];
        used_list[idxCurr].x = -1;
        used_list[idxCurr].y = -1;
        if (idxCurr == ref_camera)
            continue;

        // Project 3d point X on camera idxCurr
        float2 source_pixel;
        project_on_camera(X, camParams.cameras[idxCurr], &source_pixel, &depth);

        // Boundary check
        if (source_pixel.x >= 0 &&
            source_pixel.x < cols &&
            source_pixel.y >= 0 &&
            source_pixel.y < rows)
        {
            // printf("Boundary check passed\n");

            // Compute interpolated depth and normal for source_pixel w.r.t. camera ref_camera
            float4 source; // first 3 components normal, fourth depth
            source = tex2D<float4>(gs.normals_depths[idxCurr], source_pixel.x + 0.5f, source_pixel.y + 0.5f);

            // const float depth_disp = disparityDepthConversion_cu2                ( camParams.cameras[ref_camera].f, camParams.cameras[ref_camera], camParams.cameras[idxCurr], depth );
            // const float tmp_normal_and_depth_disp = disparityDepthConversion_cu2 ( camParams.cameras[ref_camera].f, camParams.cameras[ref_camera], camParams.cameras[idxCurr], tmp_normal_and_depth.w );
            // First consistency check on depth

            if (fabsf(depth - source.w) < gs.params->depthThresh)
            {

                float angle = getAngle_cu(source, normal); // extract normal
                if (angle < gs.params->normalThresh)
                {

                    /// All conditions met:
                    //  - average 3d points and normals
                    //  - save resulting point and normal
                    //  - (optional) average texture (not done yet)

                    int2 tmp_p = make_int2((int)source_pixel.x, (int)source_pixel.y);
                    float4 tmp_X; // 3d point of consistent point on other view
                    get3Dpoint_cu(&tmp_X, camParams.cameras[idxCurr], tmp_p, source.w);

                    consistent_X = consistent_X + tmp_X;
                    consistent_normal = consistent_normal + source;
                    if (gs.params->saveTexture)
                        consistent_texture = consistent_texture + tex2D<float>(gs.imgs[idxCurr], source_pixel.x + 0.5f, source_pixel.y + 0.5f);

                    // Save the point for later check
                    used_list[idxCurr].x = (int)source_pixel.x;
                    used_list[idxCurr].y = (int)source_pixel.y;

                    number_consistent++;
                }
            }
        }
        else
            continue;
    }

    // Average normals and points
    consistent_X = consistent_X / ((float)number_consistent + 1.0f);
    consistent_normal = consistent_normal / ((float)number_consistent + 1.0f);
    consistent_texture = consistent_texture / ((float)number_consistent + 1.0f);

    // If at least numConsistentThresh point agree:
    // Create point
    // Save normal
    // (optional) save texture

    if (number_consistent >= max(camParams.selectedViewsNum[ref_camera] - gs.params->numConsistentThresh, gs.params->number_consistent_input))
    {
        // printf("\tEnough consistent points!\nSaving point %f %f %f", consistent_X.x, consistent_X.y, consistent_X.z);
        if (!gs.params->remove_black_background || consistent_texture > 15) // hardcoded for middlebury TODO FIX
        {
            gs.pc->points[center].coord = consistent_X;
            gs.pc->points[center].normal = consistent_normal;

            if (gs.params->saveTexture)
                gs.pc->points[center].texture = consistent_texture;

            // Mark corresponding point on other views as "used"
            gs.lines_consistency[ref_camera].used_pixels[center] = 1;
            for (int i = 0; i < camParams.selectedViewsNum[ref_camera]; i++)
            {
                int idxCurr = camParams.selectedViews[ref_camera][i];
                // printf("%d\n", number_consistent);
                if (used_list[idxCurr].x == -1)
                {

                    // printf("%f\n", gs.lines_consistency[idxCurr].suggestions[c].w);
                }
                else
                {
                    // printf("Used list point on camera %d is %d %d\n", idxCurr, used_list[idxCurr].x, used_list[idxCurr].y);
                    gs.lines_consistency[idxCurr].used_pixels[used_list[idxCurr].x + used_list[idxCurr].y * cols] = 1;
                }
                float2 source_pixel;
                project_on_camera(consistent_X, camParams.cameras[idxCurr], &source_pixel, &depth);
                // printf("%f\n", depth);
                if (source_pixel.x <= 5 || source_pixel.x >= cols - 5 || source_pixel.y <= 5 || source_pixel.y >= rows - 5)
                    continue;
                // float4 target = gs.lines_consistency[idxCurr].suggestions[source_pixel.x + source_pixel.y * cols];
                int c = (int)(source_pixel.x + 0.5f) + (int)(source_pixel.y + 0.5f) * cols;
                if (gs.lines_consistency[idxCurr].suggestions[c].w == 0)
                {
                    gs.lines_consistency[idxCurr].suggestions[c].x = consistent_normal.x;
                    gs.lines_consistency[idxCurr].suggestions[c].y = consistent_normal.y;
                    gs.lines_consistency[idxCurr].suggestions[c].z = consistent_normal.z;
                    gs.lines_consistency[idxCurr].suggestions[c].w = depth;
                    // gs.lines_consistency[gs.num_images].test_s[c]++;
                }
                // else
                // {
                //     gs.lines_consistency[idxCurr].suggestions_1[c].x = consistent_normal.x;
                //     gs.lines_consistency[idxCurr].suggestions_1[c].y = consistent_normal.y;
                //     gs.lines_consistency[idxCurr].suggestions_1[c].z = consistent_normal.z;
                //     gs.lines_consistency[idxCurr].suggestions_1[c].w = depth;
                //     // gs.lines_consistency[gs.num_images].test_s1[c]++;
                // }
            }
        }
    }

    return;
}

/* Copy point cloud to global memory */
// template< typename T >
void copy_point_cloud_to_host(GlobalState &gs, int cam, PointCloudList &pc_list)
{
    printf("Processing camera %d\n", cam);
    unsigned int count = pc_list.size;
    for (int y = 0; y < gs.pc->rows; y++)
    {
        for (int x = 0; x < gs.pc->cols; x++)
        {
            Point_cu &p = gs.pc->points[x + y * gs.pc->cols];
            const float4 X = p.coord;
            const float4 normal = p.normal;
            float texture = 127.0f;

            if (gs.params->saveTexture)
                texture = p.texture;

            if (count == pc_list.maximum)
            {
                printf("Not enough space to save points :'(\n... allocating more! :)");
                pc_list.increase_size(pc_list.maximum * 2);
            }
            if (X.x != 0 && X.y != 0 && X.z != 0)
            {
                pc_list.points[count].coord = X;
                pc_list.points[count].normal = normal;

                pc_list.points[count].texture = texture;

                count++;
            }
            p.coord = make_float4(0, 0, 0, 0);
        }
    }
    printf("Found %.7f million points\n", count / 1000000.0f);
    pc_list.size = count;
}

template <typename T>
void fusibile_cu(GlobalState &gs, PointCloudList &pc_list, int num_views)
{
#ifdef SHARED
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
#endif

    cudaError_t err;
    err = cudaGetLastError();
    printf("cuda check: %s\n", cudaGetErrorString(err));

#ifndef SHARED_HARDCODED // used
    // int blocksize_w = gs.params->box_hsize + 1; // 12, +1 for the gradient computation, gs.box_hsize=11
    // int blocksize_h = gs.params->box_vsize + 1; // +1 for the gradient computation, gs.box_vsize=11
    // WIN_RADIUS_W = (blocksize_w) / (2); // 6
    // WIN_RADIUS_H = (blocksize_h) / (2); // 6

    int BLOCK_W = 16;
    int BLOCK_H = (BLOCK_W / 2); // 16
    // TILE_W = BLOCK_W;                                
    // TILE_H = BLOCK_H * 2;                            
    // SHARED_SIZE_W_m = (TILE_W + WIN_RADIUS_W * 2);   
    // SHARED_SIZE_H = (TILE_H + WIN_RADIUS_H * 2);     
    // SHARED_SIZE = (SHARED_SIZE_W_m * SHARED_SIZE_H); 
    // cudaMemcpyToSymbol(SHARED_SIZE_W, &SHARED_SIZE_W_m, sizeof(SHARED_SIZE_W_m));
// SHARED_SIZE_W_host = SHARED_SIZE_W_m;
#else
    // SHARED_SIZE_W_host = SHARED_SIZE;
#endif

    err = cudaGetLastError();
    printf("cuda check: %s\n", cudaGetErrorString(err));

    int rows = gs.rows;
    int cols = gs.cols;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("Run gipuma\n");
    /*curandState* devStates;*/
    // cudaMalloc ( &gs.cs, rows*cols*sizeof( curandState ) );
    err = cudaGetLastError();
    printf("cuda check: %s\n", cudaGetErrorString(err));
    int count = 0;
    int i = 0;

    cudaGetDeviceCount(&count);
    if (count == 0)
    {
        fprintf(stderr, "There is no device.\n");
        return;
    }
    else
    {
        fprintf(stderr, "There is device.\n");
        printf("count: %d\n", count);
    }
    err = cudaGetLastError();
    printf("cuda check: %s\n", cudaGetErrorString(err));
    for (i = 0; i < count; i++)
    {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
        {
            if (prop.major >= 1)
            {
                break;
            }
        }
    }
    if (i == count)
    {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        printf("There is no device supporting CUDA.\n");
        return;
    }
    // float mind = gs.params.min_disparity;
    // float maxd = gs.params.max_disparity;
    // srand(0);
    // for(int x = 0; x < gs.cameras.cols; x++) {
    // for(int y = 0; y < gs.cameras.rows; y++) {
    // gs.lines.disp[y*gs.cameras.cols+x] = (float)rand()/(float)RAND_MAX * (maxd-mind) + mind;
    //[>printf("%f\n", gs.lines.disp[y*256+x]);<]
    // }
    // }
    /*printf("MAX DISP is %f\n", gs.params.max_disparity);*/
    /*printf("MIN DISP is %f\n", gs.params.min_disparity);*/
    cudaSetDevice(i);
    // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*128);//0727
    dim3 grid_size;
    grid_size.x = (cols + BLOCK_W - 1) / BLOCK_W;
    grid_size.y = ((rows / 2) + BLOCK_H - 1) / BLOCK_H;
    dim3 block_size;
    block_size.x = BLOCK_W;
    block_size.y = BLOCK_H;
    err = cudaGetLastError();
    printf("cuda check: %s\n", cudaGetErrorString(err));
    // dim3 grid_size_initrand;
    // grid_size_initrand.x=(cols+32-1)/32;
    // grid_size_initrand.y=(rows+32-1)/32;
    // dim3 block_size_initrand;
    // block_size_initrand.x=32;
    // block_size_initrand.y=32;

    dim3 grid_size_initrand;
    int block_size_temp = 32;
    grid_size_initrand.x = (cols + block_size_temp - 1) / block_size_temp;
    grid_size_initrand.y = (rows + block_size_temp - 1) / block_size_temp;
    dim3 block_size_initrand;
    block_size_initrand.x = block_size_temp;
    block_size_initrand.y = block_size_temp;

    /*     printf("Launching kernel with grid of size %d %d and block of size %d %d and shared size %d %d\nBlock %d %d and radius %d %d and tile %d %d\n",
               grid_size.x,
               grid_size.y,
               block_size.x,
               block_size.y,
               SHARED_SIZE_W,
               SHARED_SIZE_H,
               BLOCK_W,
               BLOCK_H,
               WIN_RADIUS_W,
               WIN_RADIUS_H,
               TILE_W,
               TILE_H
              );
     */
    printf("Grid size initrand is grid: %d-%d block: %d-%d\n", grid_size_initrand.x, grid_size_initrand.y, block_size_initrand.x, block_size_initrand.y);

    size_t avail;
    size_t total;
    cudaMemGetInfo(&avail, &total);
    size_t used = total - avail;
    printf("Device memory used: %fMB\n", used / 1000000.0f);
    printf("Number of iterations is %d\n", gs.params->iterations);
    printf("Blocksize is %dx%d\n", gs.params->box_hsize, gs.params->box_vsize);
    printf("Disparity threshold is \t%f\n", gs.params->depthThresh);
    printf("Normal threshold is \t%f\n", gs.params->normalThresh);
    printf("Number of consistent points is \t%d\n", gs.params->numConsistentThresh);
    printf("gs.params->number_consistent_input: \t%d\n", gs.params->number_consistent_input);
    printf("Cam scale is \t%f\n", gs.params->cam_scale);

    // int shared_memory_size = sizeof(float)  * SHARED_SIZE ;
    printf("Fusing points start\n");
    cudaEventRecord(start);

    err = cudaGetLastError();
    printf("cuda check: %s\n", cudaGetErrorString(err));

    // printf("Computing final disparity\n");
    // for (int cam=0; cam<10; cam++) {
    printf("num_views: %d\n", num_views);

    // int sum1 = 0, sum2 = 0;
    // for (int i=0; i<gs.rows; i++) {
    //     for (int j=0; j< gs.cols; j++) {
    //         sum1 += gs.lines_consistency[gs.num_images].test_s[i*gs.cols+j];
    //         sum2 += gs.lines_consistency[gs.num_images].test_s1[i*gs.cols+j];
    //     }
    // }
    // printf("sum1: %d, sum2:%d",sum1, sum2);
    for (int cam = 0; cam < num_views; cam++)
    {

        
        if(gs.cycles == gs.params->cycles - 1)
        {
            ambc_fusibile<<<grid_size_initrand, block_size_initrand>>>(gs, cam);
            cudaDeviceSynchronize();
        }
        else
        {
            ambc_strong_new<<<grid_size_initrand, block_size_initrand>>>(gs, cam);
            cudaDeviceSynchronize();
            ambc_weak<<<grid_size_initrand, block_size_initrand>>>(gs, cam);
            // ambc_fusibile<<<grid_size_initrand, block_size_initrand>>>(gs, cam);
            cudaDeviceSynchronize();
        }
        // ambc_rad<<<grid_size_initrand, block_size_initrand>>>(gs, cam);
        // cudaDeviceSynchronize();
        // fusibile<<< grid_size_initrand, block_size_initrand, cam>>>(gs, cam);
        // cudaDeviceSynchronize();
        copy_point_cloud_to_host(gs, cam, pc_list); // slower but saves memory
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        printf("cuda check: %s\n", cudaGetErrorString(err));
        // sum1 = 0, sum2 = 0;
        // for (int i=0; i<gs.rows; i++) {
        //     for (int j=0; j< gs.cols; j++) {
        //         sum1 += gs.lines_consistency[gs.num_images].test_s[i*gs.cols+j];
        //         sum2 += gs.lines_consistency[gs.num_images].test_s1[i*gs.cols+j];
        //     }
        // }
        // printf("sum1: %d, sum2:%d",sum1, sum2);
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\t\tELAPSED %f seconds\n", milliseconds / 1000.f);

    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    printf("run cuda finished\n");

    // print results to file
}

int runconsistency(GlobalState &gs, PointCloudList &pc_list, int num_views)
{
    printf("runconsistency\n");
    /*GlobalState *gs = new GlobalState;*/
    if (gs.params->color_processing)
    {
        printf("color processing\n");
        fusibile_cu<float4>(gs, pc_list, num_views);
    }
    else
    {
        printf("gray processing\n");
        fusibile_cu<float>(gs, pc_list, num_views);
    }
    // cudaDeviceReset();
    printf("runconsistency finished\n");
    return 0;
}
