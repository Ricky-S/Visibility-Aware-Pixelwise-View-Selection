
#ifndef MATHUTILS_H
#define MATHUTILS_H
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"



using namespace cv;

inline void copyOpencvMatToFloatArray ( Mat_<float> &m, float **a)
{
    for (int pj=0; pj<m.rows ; pj++)
    {
        for (int pi=0; pi<m.cols ; pi++)
        {
            (*a)[pi+pj*m.cols] = m(pj,pi);
        }
    } 
    // cout<<"finishconvert"<<endl;
}


inline void copyOpencvVecToFloat4 ( Vec3f &v, float4 *a)
{
    a->x = v(0);
    a->y = v(1);
    a->z = v(2);
}

#ifndef M_PI
#define M_PI    3.14159265358979323846f
#endif
#define M_PI_float    3.14159265358979323846f


/* get angle between two vectors in 3D
 * Input: v1,v2 - vectors
 * Output: angle in radian
 */
static float getAngle ( Vec3f v1, Vec3f v2 ) {
	float angle = acosf ( v1.dot ( v2 ) );
	//if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
	if ( angle != angle )
		return 0.0f;
	//if ( acosf ( v1.dot ( v2 ) ) != acosf ( v1.dot ( v2 ) ) )
		//cout << acosf ( v1.dot ( v2 ) ) << " / " << v1.dot ( v2 )<< " / " << v1<< " / " << v2 << endl;
	return angle;
}


#endif

// inline void NormalizeVec3 (float4 *vec)
// {
//     const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
//     const float inverse_sqrt = rsqrtf (normSquared);
//     vec->x *= inverse_sqrt;
//     vec->y *= inverse_sqrt;
//     vec->z *= inverse_sqrt;
// }

/*
 * some math helper functions
 */

// #pragma once

// #ifndef M_PI
// #define M_PI    3.14159265358979323846f
// #endif
// #define M_PI_float    3.14159265358979323846f



