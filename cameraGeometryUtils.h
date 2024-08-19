/*
 *  cameraGeometryUtils.h
 *
 * utility functions for camera geometry related stuff
 * most of them from: "Multiple View Geometry in computer vision" by Hartley and Zisserman
 */

#pragma once
#include <string>
#include "mathUtils.h"
#include <limits>
#include <signal.h>


inline Vec3f get3Dpoint ( Camera &cam, float x, float y, float depth ) {
    // in case camera matrix is not normalized: see page 162, then depth might not be the real depth but w and depth needs to be computed from that first

    Mat_<float> pt = Mat::ones ( 3,1,CV_32F );
    pt ( 0,0 ) = x;
    pt ( 1,0 ) = y;

    //formula taken from page 162 (alternative expression)
    Mat_<float> ptX = cam.M_inv * ( depth*pt - cam.P.col ( 3 ) );
    return Vec3f ( ptX ( 0 ),ptX ( 1 ),ptX ( 2 ) );
}

inline Vec3f get3Dpoint ( Camera &cam, int x, int y, float depth ){
    return get3Dpoint(cam,(float)x,(float)y,depth);
}

// get the viewing ray for a pixel position of the camera
static inline Vec3f getViewVector ( Camera &cam, int x, int y) {

    //get some point on the line (the other point on the line is the camera center)
    Vec3f ptX = get3Dpoint ( cam,x,y,1.0f );

    //get vector between camera center and other point on the line
    Vec3f v = ptX - cam.C;
    // cout <<"getViewVectorcam.C: " << cam.C(1) << cam.C(2) << cam.C(3) << endl;
    // cout <<"getViewVector: " << v(1) << v(2) << v(3) << endl;
    // cout <<"getViewVector: " << v(1) << v(2) << v(3) << endl;
    return normalize ( v );
}

// get the viewing ray for a pixel position of the camera
static inline Vec3f getViewVector_rev_angle ( Camera &cam, int x, int y) {

    //get some point on the line (the other point on the line is the camera center)
    Vec3f ptX = get3Dpoint ( cam,x,y,1.0f );

    //get vector between camera center and other point on the line
    Vec3f v = cam.C - ptX; // to make angles between camera and lines in 0-90.
    // cout <<"getViewVectorcam.C: " << cam.C(1) << cam.C(2) << cam.C(3) << endl;
    // cout <<"getViewVector: " << v(1) << v(2) << v(3) << endl;
    // cout <<"getViewVector: " << v(1) << v(2) << v(3) << endl;
    return normalize ( v );
}

/* get depth from 3D point
 * page 162: w = P3T*X (P3T ... third (=last) row of projection matrix P)
 */
float getDepth ( Vec3f &X, Mat_<float> &P ) {
    //assuming homogenous component of X being 1
    float w =  P ( 2,0 )*X ( 0 ) + P ( 2,1 ) * X ( 1 ) + P ( 2,2 ) * X ( 2 ) + P ( 2,3 );

    return w;
}

// Mat_<float> getTransformationMatrix ( Mat_<float> R, Mat_<float> t ) {
//     Mat_<float> transMat = Mat::eye ( 4,4, CV_32F );
//     //Mat_<float> Rt = - R * t;
//     R.copyTo ( transMat ( Range ( 0,3 ),Range ( 0,3 ) ) );
//     t.copyTo ( transMat ( Range ( 0,3 ),Range ( 3,4 ) ) );

//     return transMat;
// }

/* compute depth value from disparity or disparity value from depth
 * Input:  f         - focal length in pixel
 *         baseline  - baseline between cameras (in meters)
 *         d - either disparity or depth value
 * Output: either depth or disparity value
 */
float disparityDepthConversion ( float f, float baseline, float d ) {
    /*if ( d == 0 )*/
        /*return FLT_MAX;*/
    return f * baseline / d;
}



Mat_<float> scaleK ( Mat_<float> K, float scaleFactor ) {

    Mat_<float> K_scaled = K.clone();
    //scale focal length
    K_scaled ( 0,0 ) = K ( 0,0 ) / scaleFactor;
    K_scaled ( 1,1 ) = K ( 1,1 ) / scaleFactor;
    //scale center point
    K_scaled ( 0,2 ) = K ( 0,2 ) / scaleFactor;
    K_scaled ( 1,2 ) = K ( 1,2 ) / scaleFactor;

    return K_scaled;
}
// void copyOpencvVecToFloat4 ( Vec3f &v, float4 *a)
// {
//     a->x = v(0);
//     a->y = v(1);
//     a->z = v(2);
// }
void copyOpencvVecToFloatArray ( Vec3f &v, float *a)
{
    a[0] = v(0);
    a[1] = v(1);
    a[2] = v(2);
}
// void copyOpencvMatToFloatArray ( Mat_<float> &m, float **a)
// {
//     for (int pj=0; pj<m.rows ; pj++)
//         for (int pi=0; pi<m.cols ; pi++)
//         {
//             (*a)[pi+pj*m.cols] = m(pj,pi);
//         }
// }

/* get camera parameters (e.g. projection matrices) from file
 * Input:  inputFiles  - pathes to calibration files
 *         scaleFactor - if image was rescaled we need to adapt calibration matrix K accordingly
 * Output: camera parameters
 */
CameraParameters getCameraParameters( CameraParameters_cu &cpc,
                                       InputFiles inputFiles,
                                       float scaleFactor = 1.0f,
                                       bool transformP = true ,
                                       int refId=0)
{

    CameraParameters params;
    size_t numCameras = 2;
    params.cameras.resize ( numCameras );
    //get projection matrices

    //load projection matrix from file (e.g. for Kitti)
    // cout <<"inputFiles.calib_filename:"<<inputFiles.calib_filename <<endl;
    if ( !inputFiles.calib_filename.empty () ) {
        //two view case
        readCalibFileKitti ( inputFiles.calib_filename,params.cameras[0].P,params.cameras[1].P );
        params.rectified = false; // for Kitti data is actually rectified, set this to true for computation in disparity space

    }
    Mat_<float> KMaros = Mat::eye ( 3, 3, CV_32F );
    KMaros(0,0) = 8066.0;
    KMaros(1,1) = 8066.0;
    KMaros(0,2) = 2807.5;
    KMaros(1,2) = 1871.5;
    //load projection matrix from file (e.g. for Strecha)
    // cout << "test217" << endl;
    // Load pmvs files
    // cout <<"inputFiles.pmvs_folder:"<<inputFiles.pmvs_folder <<endl;
    // cout <<"inputFiles.p_folder:"<<inputFiles.p_folder <<endl;
    // cout <<"inputFiles.krt_file:"<<inputFiles.krt_file <<endl;
    // cout <<"inputFiles.in_ex_folder:"<<inputFiles.in_ex_folder <<endl;
    if ( !inputFiles.pmvs_folder.empty () ) {
        numCameras = inputFiles.num_images;
        params.cameras.resize ( numCameras );
        // cout << "numCameras" <<numCameras << endl;
        for ( size_t i = 0; i < numCameras; i++ ) {
            size_t lastindex = inputFiles.img_filenames[i].find_last_of(".");
            string filename_without_extension = inputFiles.img_filenames[i].substr(0, lastindex);
            cout << filename_without_extension << endl;
            readPFileStrechaPmvs ( inputFiles.p_folder + filename_without_extension + ".txt",params.cameras[i].P );
            size_t found = inputFiles.img_filenames[i].find_last_of ( "." );
            cout <<found << endl;
            //params.cameras[i].id = atoi ( inputFiles.img_filenames[i].substr ( 0,found ).c_str () );
            params.cameras[i].id = inputFiles.img_filenames[i].substr ( 0,found ).c_str ();
            // params.cameras[i].P = KMaros * params.cameras[i].P;
            //cout << params.cameras[i].P << endl;

        }
    }
    // Load p files strecha style
    else if ( !inputFiles.p_folder.empty () ) {
        numCameras = inputFiles.num_images;
        // cout << "numCameras" <<numCameras << endl;
        params.cameras.resize ( numCameras );
        // cout << "numCameras" <<numCameras << endl;
        cout << inputFiles.p_folder << endl;
        cout << inputFiles.img_filenames[0] << endl;
        for ( size_t i = 0; i < numCameras; i++ ) {

            // readPFileStrechaPmvs ( inputFiles.p_folder + inputFiles.img_filenames[i] + ".P",params.cameras[i].P );

            // related to i
            if ( i < 9) {
                // cout << inputFiles.p_folder + "rect_00"+ to_string(i+1) + "_3_r5000.png.P" << endl;
                readPFileStrechaPmvs ( inputFiles.p_folder + "rect_00"+ to_string(i+1) + "_3_r5000.png.P",params.cameras[i].P );
            } else {
                // cout << inputFiles.p_folder + "rect_0"+ to_string(i+1) + "_3_r5000.png.P" << endl;
                readPFileStrechaPmvs ( inputFiles.p_folder + "rect_0"+ to_string(i+1) + "_3_r5000.png.P",params.cameras[i].P );
            } // related to i

            size_t found = inputFiles.img_filenames[i].find_last_of ( "." );
            //params.cameras[i].id = atoi ( inputFiles.img_filenames[i].substr ( 0,found ).c_str () );
            params.cameras[i].id = inputFiles.img_filenames[i].substr ( 0,found ).c_str ();
           // params.cameras[i].P = KMaros * params.cameras[i].P;

            //cout << params.cameras[i].P << endl;
        }

    }
    else if (!inputFiles.in_ex_folder.empty () ) {
        // cout << "inputFiles.num_images;" << inputFiles.num_images<<endl;
        numCameras = inputFiles.num_images;
        params.cameras.resize ( numCameras );
        // cout << "Num Cameras " << numCameras << endl;
        for (int i = 0; i < numCameras; i++ )
        {
            
            size_t lastindex = inputFiles.img_filenames[i].find_last_of(".");
            string filename_without_extension = inputFiles.img_filenames[i].substr(0, lastindex);
            // cout <<"read cam:" << filename_without_extension<<endl;
            // printf("read cam: %s", filename_without_extension);
            read_in_ex ( inputFiles.in_ex_folder + filename_without_extension + "_cam.txt", params.cameras[i], inputFiles);
        }
        
    }


    // Load P matrix for middlebury format
    if ( !inputFiles.krt_file.empty () ) {
        numCameras = inputFiles.num_images;
        params.cameras.resize ( numCameras );
        //cout << "Num Cameras " << numCameras << endl;
        
        readKRtFileMiddlebury_ori ( inputFiles.krt_file, params.cameras, inputFiles);
    }


    /*cout << "KMaros is" << endl;*/
    /*cout << KMaros << endl;*/


    // decompose projection matrices into K, R and t
    vector<Mat_<float> > K ( numCameras );
    vector<Mat_<float> > R ( numCameras );
    vector<Mat_<float> > T ( numCameras );

    vector<Mat_<float> > C ( numCameras );
    vector<Mat_<float> > t ( numCameras );
    // cout << "test274" << endl;
    // cout << "camera[i].P" <<params.cameras[0].P<<endl;
    for ( size_t i = 0; i < numCameras; i++ ) {
        decomposeProjectionMatrix ( params.cameras[i].P,K[i],R[i],T[i] );

        // cout << "K: " << K[i] << endl;
        // cout << "R: " << R[i] << endl;
        // cout << "T: " << T[i] << endl;

        // get 3-dimensional translation vectors and camera center (divide by augmented component)
        C[i] = T[i] ( Range ( 0,3 ),Range ( 0,1 ) ) / T[i] ( 3,0 );
        t[i] = -R[i] * C[i];

        //cout << "C: " << C[i] << endl;
        //cout << "t: " << t[i] << endl;
    }

    // transform projection matrices (R and t part) so that P1 = K [I | 0]
    //computeTranslatedProjectionMatrices(R1, R2, t1, t2, params);
    Mat_<float> transform = Mat::eye ( 4,4 ,CV_32F);
    // cout << "test293" << endl;
    if ( transformP )
        transform = getTransformationReferenceToOrigin ( R[refId],t[refId] ); 
    /*cout << "transform is " << transform << endl;*/
    params.cameras[0].reference = true;
    params.idRef = 0;
    //cout << "K before scale is" << endl;
    //cout << K[0] << endl;

    //assuming K is the same for all cameras
    params.K = scaleK ( K[refId],scaleFactor );
    params.K_inv = params.K.inv ();
    // get focal length from calibration matrix
    params.f = params.K ( 0,0 );
    // cout << "test307" << endl;

    for ( size_t i = 0; i < numCameras; i++ ) {
        params.cameras[i].K = scaleK(K[i],scaleFactor);
        params.cameras[i].K_inv = params.cameras[i].K.inv ( );
        //params.cameras[i].f = params.cameras[i].K(0,0);
        // cout << "test camera" << i <<endl;
        // printf("tbefore %f\n", params.cameras[i].t(0));
        if ( !inputFiles.bounding_folder.empty () ) {
            Vec3f ptBL, ptTR;
            readBoundingVolume ( inputFiles.bounding_folder + inputFiles.img_filenames[i] + ".bounding",ptBL,ptTR );

            // cout << "d1: " << getDepth ( ptBL,params.cameras[i].P ) <<endl;
            // cout << "d2: " << getDepth ( ptTR,params.cameras[i].P ) <<endl;
        }

        params.cameras[i].R_orig_inv = R[i].inv (DECOMP_SVD);
        params.cameras[i].R = R[i];
        copyOpencvMatToFloatArray ( params.cameras[i].R,     &cpc.cameras[i].R_orig);       
        cpc.cameras[i].t4_orig.x = t[i](0);
        cpc.cameras[i].t4_orig.y = t[i](1);
        cpc.cameras[i].t4_orig.z = t[i](2);     
        transformCamera ( R[i],t[i], transform,    params.cameras[i], params.K );

        params.cameras[i].P_inv = params.cameras[i].P.inv ( DECOMP_SVD );
        params.cameras[i].M = params.cameras[i].P.colRange ( 0,3 ); 
        params.cameras[i].M_inv = params.cameras[i].P.colRange ( 0,3 ).inv ();

        // set camera baseline (if unknown we need to guess something)
        //float b = (float)norm(t1,t2,NORM_L2);
        params.cameras[i].baseline = 0.54f; //0.54 = Kitti baseline

        // K
        Mat_<float> tmpK = params.K.t ();
        //copyOpencvMatToFloatArray ( params.K, &cpc.K);
        //copyOpencvMatToFloatArray ( params.K_inv, &cpc.K_inv);
        copyOpencvMatToFloatArray ( params.cameras[i].K, &cpc.cameras[i].K);
        copyOpencvMatToFloatArray ( params.cameras[i].K_inv, &cpc.cameras[i].K_inv);
        copyOpencvMatToFloatArray ( params.cameras[i].R_orig_inv, &cpc.cameras[i].R_orig_inv);
        cpc.cameras[i].fy = params.K(1,1);
        cpc.f = params.K(0,0);
        cpc.cameras[i].f = params.K(0,0);
        cpc.cameras[i].fx = params.K(0,0);
        cpc.cameras[i].fy = params.K(1,1);
        cpc.cameras[i].baseline = params.cameras[i].baseline;
        cpc.cameras[i].reference = params.cameras[i].reference;

        /*params.cameras[i].alpha = params.K ( 0,0 )/params.K(1,1);*/
        cpc.cameras[i].alpha = params.K ( 0,0 )/params.K(1,1);
        // Copy data to cuda structure
        copyOpencvMatToFloatArray ( params.cameras[i].P,     &cpc.cameras[i].P);
        copyOpencvMatToFloatArray ( params.cameras[i].P_inv, &cpc.cameras[i].P_inv);
        copyOpencvMatToFloatArray ( params.cameras[i].M,     &cpc.cameras[i].M);
        copyOpencvMatToFloatArray ( params.cameras[i].M_inv, &cpc.cameras[i].M_inv);
        //copyOpencvMatToFloatArray ( params.K,                &cpc.cameras[i].K);
        //copyOpencvMatToFloatArray ( params.K_inv,            &cpc.cameras[i].K_inv);
        copyOpencvMatToFloatArray ( params.cameras[i].K,                &cpc.cameras[i].K);
        copyOpencvMatToFloatArray ( params.cameras[i].K_inv,            &cpc.cameras[i].K_inv);
        copyOpencvMatToFloatArray ( params.cameras[i].R,     &cpc.cameras[i].R);
        /*copyOpencvMatToFloatArray ( params.cameras[i].t, &cpc.cameras[i].t);*/
        /*copyOpencvVecToFloatArray ( params.cameras[i].C, cpc.cameras[i].C);*/
        copyOpencvVecToFloat4 ( params.cameras[i].C,         &cpc.cameras[i].C4);
        cpc.cameras[i].t4.x = params.cameras[i].t(0);
        cpc.cameras[i].t4.y = params.cameras[i].t(1);
        cpc.cameras[i].t4.z = params.cameras[i].t(2);
        // printf("t %f\n", params.cameras[i].t(0));
        Mat_<float> tmp = params.cameras[i].P.col(3);
        /*cpc.cameras[i].P_col3[0] = tmp(0,0);*/
        /*cpc.cameras[i].P_col3[1] = tmp(1,0);*/
        /*cpc.cameras[i].P_col3[2] = tmp(2,0);*/
        cpc.cameras[i].P_col34.x = tmp(0,0);
        cpc.cameras[i].P_col34.y = tmp(1,0);
        cpc.cameras[i].P_col34.z = tmp(2,0);
        //cout << params.cameras[i].P << endl;
        //cout << endl;
        // cout << "1: " << params.cameras[i].P(0,0) << endl;
        Mat_<float> tmpKinv = params.K_inv.t ();
    }

    return params;
}

