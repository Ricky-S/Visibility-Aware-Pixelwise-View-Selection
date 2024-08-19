/*
 * utility functions for visualization of results
 */

#pragma once
#include <sstream>
#include <fstream>

#if (CV_MAJOR_VERSION ==2)
#include <opencv2/contrib/contrib.hpp> // needed for applyColorMap!
#endif


/* compute gamma correction (just for display purposes to see more details in farther away areas of disparity image)
 * Input: img   - image
 *        gamma - gamma value
 * Output: gamma corrected image
 */
Mat correctGamma( Mat& img, double gamma ) {
 double inverse_gamma = 1.0 / gamma;

 Mat lut_matrix(1, 256, CV_8UC1 );
 uchar * ptr = lut_matrix.ptr();
 for( int i = 0; i < 256; i++ )
   ptr[i] = (int)( pow( (double) i / 255.0, inverse_gamma ) * 255.0 );

 Mat result;
 LUT( img, lut_matrix, result );

 return result;
}


static void getDisparityForDisplay(const Mat_<float> &disp, Mat &dispGray, Mat &dispColor, float numDisparities, float minDisp = 0.0f){
	float gamma = 2.0f; // to get higher contrast for lower disparity range (just for color visualization)
	disp.convertTo(dispGray,CV_16U,65535.f/(numDisparities-minDisp),-minDisp*65535.f/(numDisparities-minDisp));
	Mat disp8;
	disp.convertTo(disp8,CV_8U,255.f/(numDisparities-minDisp),-minDisp*255.f/(numDisparities-minDisp));
	if(minDisp == 0.0f)
		disp8 = correctGamma(disp8,gamma);
	applyColorMap(disp8, dispColor, COLORMAP_JET);
	for(int y = 0; y < dispColor.rows; y++){
		for(int x = 0; x < dispColor.cols; x++){
			if(disp(y,x) <= 0.0f)
				dispColor.at<Vec3b>(y,x) = Vec3b(0,0,0);
		}
	}
}

// static void convertDisparityDepthImage(const Mat_<float> &dispL, Mat_<float> &d, float f, float baseline){
// 	d = Mat::zeros(dispL.rows, dispL.cols, CV_32F);
// 	for(int y = 0; y < dispL.rows; y++){
// 		for(int x = 0; x < dispL.cols; x++){
// 			d(y,x) = disparityDepthConversion(f,baseline,dispL(y,x));
// 		}
// 	}
// }

static string getColorString(uint8_t color){
	stringstream ss;
	ss << (int)color << " " << (int)color << " " << (int)color;
	return ss.str();
}


static string getColorString(Vec3b color){
	stringstream ss;
	ss << (int)color(2) << " " << (int)color(1) << " " << (int)color(0);
	return ss.str();
}

static string getColorString(Vec3i color){
	stringstream ss;
	ss << (int)((float)color(2)/256.f) << " " << (int)((float)color(1)/256.f) << " " << (int)((float)color(0)/256.f);
	return ss.str();
}


static void getNormalsForDisplay(const Mat &normals, Mat &normals_display, int rtype = CV_16U){
	if(rtype == CV_8U)
		normals.convertTo(normals_display,CV_8U,128,128);
	else
		normals.convertTo(normals_display,CV_16U,32767,32767);
	cvtColor(normals_display,normals_display,COLOR_RGB2BGR);
}



static void storePlyFileBinaryPointCloud (char* plyFilePath, PointCloudList &pc, Mat_<float> &distImg) {
    cout << "store 3D points to ply file" << endl;

    FILE *outputPly;
    outputPly=fopen(plyFilePath,"wb");

    /*write header*/
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n",pc.size);
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    fprintf(outputPly, "property float nx\n");
    fprintf(outputPly, "property float ny\n");
    fprintf(outputPly, "property float nz\n");
    fprintf(outputPly, "property uchar red\n");
    fprintf(outputPly, "property uchar green\n");
    fprintf(outputPly, "property uchar blue\n");
    fprintf(outputPly, "end_header\n");

    distImg = Mat::zeros(pc.rows,pc.cols,CV_32F);

    //write data
#pragma omp parallel for
    for(long int i = 0; i < pc.size; i++) {
        const Point_li &p = pc.points[i];
        const float4 normal = p.normal;
        float4 X = p.coord;
        const char color = (int)p.texture;
        /*const int color = 127.0f;*/
        /*printf("Writing point %f %f %f\n", X.x, X.y, X.z);*/

        // if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
        //     X.x = 0.0f;
        //     X.y = 0.0f;
        //     X.z = 0.0f;
        // }
        if(!(X.x < 1000 && X.x > -1000) || !(X.y < 1000 && X.y > -1000) || !(X.z < 1000 && X.z >= -1000)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
#pragma omp critical
        {
            /*myfile << X.x << " " << X.y << " " << X.z << " " << normal.x << " " << normal.y << " " << normal.z << " " << color << " " << color << " " << color << endl;*/
            fwrite(&X.x,      sizeof(X.x), 1, outputPly);
            fwrite(&X.y,      sizeof(X.y), 1, outputPly);
            fwrite(&X.z,      sizeof(X.z), 1, outputPly);
            fwrite(&normal.x, sizeof(normal.x), 1, outputPly);
            fwrite(&normal.y, sizeof(normal.y), 1, outputPly);
            fwrite(&normal.z, sizeof(normal.z), 1, outputPly);
            fwrite(&color,  sizeof(char), 1, outputPly);
            fwrite(&color,  sizeof(char), 1, outputPly);
            fwrite(&color,  sizeof(char), 1, outputPly);
        }

    }
    fclose(outputPly);
}


static void storePlyFileBinaryPointCloud(char* plyFilePath, PointCloud &pc, Mat_<float> &distImg) {
    cout << "store 3D points to ply file" << endl;

    FILE *outputPly;
    outputPly=fopen(plyFilePath,"wb");

    /*write header*/
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n",pc.size);
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    /*fprintf(outputPly, "property float nx\n");*/
    /*fprintf(outputPly, "property float ny\n");*/
    /*fprintf(outputPly, "property float nz\n");*/
    /*fprintf(outputPly, "property uchar red\n");*/
    /*fprintf(outputPly, "property uchar green\n");*/
    /*fprintf(outputPly, "property uchar blue\n");*/
    fprintf(outputPly, "end_header\n");


    distImg = Mat::zeros(pc.rows,pc.cols,CV_32F);

    //write data
/*#pragma omp parallel for*/
    for(int i = 0; i < pc.size; i++){
        const Point_cu &p = pc.points[i];
        //const float4 normal = p.normal;
        float4 X = p.coord;
        /*printf("Writing point %f %f %f\n", X.x, X.y, X.z);*/
        /*float color = p.texture;*/
        //const char color = 127.0f;

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
/*#pragma omp critical*/
        {
            /*myfile << X.x << " " << X.y << " " << X.z << " " << normal.x << " " << normal.y << " " << normal.z << " " << color << " " << color << " " << color << endl;*/
            fwrite(&(X.x),      sizeof(float), 1, outputPly);
            fwrite(&(X.y),      sizeof(float), 1, outputPly);
            fwrite(&(X.z),      sizeof(float), 1, outputPly);
            /*fwrite(&(normal.x), sizeof(float), 1, outputPly);*/
            /*fwrite(&(normal.y), sizeof(float), 1, outputPly);*/
            /*fwrite(&(normal.z), sizeof(float), 1, outputPly);*/
            /*fwrite(&color,  sizeof(char), 1, outputPly);*/
            /*fwrite(&color,  sizeof(char), 1, outputPly);*/
            /*fwrite(&color,  sizeof(char), 1, outputPly);*/
        }

        /*distImg(y,x) = sqrt(pow(X.x-cam.C(0),2)+pow(X.y-cam.C(1),2)+pow(X.z-cam.C(2),2));*/
    }

    /*myfile.close();*/
    fclose(outputPly);
}




void saveConsImg(GlobalState &gs, string filename, string parentFolder, int idxCurr)
{
    printf("start generating consistency image\n");
    Mat depth(cv::Size(gs.cols, gs.rows), CV_8UC1);

    for (int x = 0; x < gs.cols; x++)
    {
        for (int y = 0; y < gs.rows; y++)
        {
            const int center = y * gs.cols + x; // y*640 +x
            // float4 f4 = gs.lines->norm4[center]; // 0724
            // float4 f4 = gs.lines->foods[center][0];
            float pixel = gs.lines_consistency[idxCurr].used_pixels[center];
            uchar z = (pixel/2.0) * 255;
            // uchar z = (uchar)255 * (pixel / ((gs.depth_upper_bound - gs.depth_lower_bound) * 2));
            depth.at<uchar>(y, x) = z;
        }
    }
    time_t now = time(0);
    char tmp[32] = {0};
    strftime(tmp, sizeof(tmp), "%Y-%m-%d-%H-%M-%S", localtime(&now));
    string outputPath;
    outputPath = parentFolder + "consis-" + filename.substr(0, filename.length() - 4) + "-" + std::string(tmp) + ".png";
    imwrite(outputPath, depth);
    printf("finish generating consis image\n");
}

// void saveNumBestImg(GlobalState &gs, string filename, string parentFolder)
// {
//     printf("start generating numBest image\n");
//     Mat depth(cv::Size(gs.cols, gs.rows), CV_8UC1);

//     for (int x = 0; x < gs.cols; x++)
//     {
//         for (int y = 0; y < gs.rows; y++)
//         {
//             const int center = y * gs.cols + x; // y*640 +x

//             float pixel = gs.lines->numBests[center];
//             float full = gs.cameras->selectedViewsNum_ambc[gs.current_image];
//             uchar z = (uchar)255 * (pixel / ((full) * 1.0));
//             depth.at<uchar>(y, x) = z;
//         }
//     }
//     time_t now = time(0);
//     char tmp[32] = {0};
//     strftime(tmp, sizeof(tmp), "%Y-%m-%d-%H-%M-%S", localtime(&now));
//     string outputPath;
//     outputPath = parentFolder + std::to_string(gs.cycles) +"cycle_numBest_" + filename.substr(0, filename.length() - 4) + "-" + std::string(tmp) + ".png";
//     imwrite(outputPath, depth);
//     printf("finish generating numBest image\n");
// }

void saveDepthImg(GlobalState &gs, string filename, string parentFolder)
{
    printf("start generating depth image\n");
    Mat depth(cv::Size(gs.cols, gs.rows), CV_8UC1);

    for (int x = 0; x < gs.cols; x++)
    {
        for (int y = 0; y < gs.rows; y++)
        {
            const int center = y * gs.cols + x;  // y*640 +x
            float4 f4 = gs.lines->norm4[center]; // 0724
            // float4 f4 = gs.lines->foods[center][0];
            // float pixel = gs.lines_consistency[idxCurr].used_pixels[center];
            float pixel = f4.w;

            uchar z = (uchar)255 * (pixel / ((gs.depth_upper_bound - gs.depth_lower_bound) * 2));
            depth.at<uchar>(y, x) = z;
        }
    }
    time_t now = time(0);
    char tmp[32] = {0};
    strftime(tmp, sizeof(tmp), "%Y-%m-%d-%H-%M-%S", localtime(&now));
    string outputPath;
    outputPath = parentFolder + "depth-" + filename.substr(0, filename.length() - 4) + "-" + std::string(tmp) + ".png";
    imwrite(outputPath, depth);
    printf("finish generating depth image\n");
}



static void storePlyFileBinary(GlobalState &gs, char *plyFilePath, const Mat_<float> &depthImg, const Mat_<Vec3f> &normals, const Mat_<float> img, Camera cam, Mat_<float> &distImg)
{

    cout << "Saving output depthmap in " << plyFilePath << endl;

    FILE *outputPly;
    outputPly = fopen(plyFilePath, "wb");
    // write header
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n", depthImg.rows * depthImg.cols);
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    fprintf(outputPly, "property float nx\n");
    fprintf(outputPly, "property float ny\n");
    fprintf(outputPly, "property float nz\n");
    fprintf(outputPly, "property uchar red\n");
    fprintf(outputPly, "property uchar green\n");
    fprintf(outputPly, "property uchar blue\n");
    fprintf(outputPly, "end_header\n");

    distImg = Mat::zeros(depthImg.rows, depthImg.cols, CV_32F);
// printf("a\n");
// write data
#pragma omp parallel for
    for (int x = 0; x < depthImg.cols; x++)
    {
        for (int y = 0; y < depthImg.rows; y++)
        {
            /*
            float zValue = depthImg(x,y);
            float xValue = ((float)x-cx)*zValue/camParams.f;
            float yValue = ((float)y-cy)*zValue/camParams.f;
            myfile << xValue << " " << yValue << " " << zValue << endl;
            */

            // Mat_<float> pt = Mat::ones(3,1,CV_32F);
            // pt(0,0) = (float)x;
            // pt(1,0) = (float)y;

            Vec3f n = normals(y, x);
            uint8_t color = img(y, x);
            // if (y > 410 && y < 420 && x > 250 && x < 260)
            //     color = 0x00FF00;
            // Mat_<float> ptX = P1_inv * depthImg(y,x)*pt;

            // printf("315x=%d , y=%d\n", x, y);
            Vec3f ptX = get3Dpoint(cam, x, y, depthImg(y, x));
            // printf("317x=%f , y=%f, z=%f\n", ptX(0), ptX(1), ptX(2));
            // Vec3f ptX_v1 = get3dPointFromPlane(cam.P_inv,cam.C,n,planes.d(y,x),x,y);
            // cout << ptX_v1 << " / " << ptX << endl;
            // if(depthImg(y,x) <= 0.0001f || depthImg(y,x) >= 900.0f) {
            if (depthImg(y, x) <= gs.depth_lower_bound || depthImg(y, x) >= gs.depth_upper_bound)
            {

                ptX(0) = 0.0f;
                ptX(1) = 0.0f;
                ptX(2) = 0.0f;
            }
            if (!(ptX(0) < FLT_MAX && ptX(0) > -FLT_MAX) || !(ptX(1) < FLT_MAX && ptX(1) > -FLT_MAX) || !(ptX(2) < FLT_MAX && ptX(2) >= -FLT_MAX))
            {
                ptX(0) = 0.0f;
                ptX(1) = 0.0f;
                ptX(2) = 0.0f;
            }

            // printf("x: %f, y: %f, z: %f\n", ptX(0), ptX(1), ptX(2));
#pragma omp critical
            {
                // myfile << ptX(0) << " " << ptX(1) << " " << ptX(2) << " " << n(0) << " " << n(1) << " " << n(2) << " " << getColorString(color) << endl;
                fwrite(&(ptX(0)), sizeof(float), 3, outputPly);
                fwrite(&(n(0)), sizeof(float), 3, outputPly);
                fwrite(&color, sizeof(color), 1, outputPly);
                fwrite(&color, sizeof(color), 1, outputPly);
                fwrite(&color, sizeof(color), 1, outputPly);
            }
            // printf("335x=%d , y=%d\n", x, y);
            distImg(y, x) = sqrt(pow(ptX(0) - cam.C(0), 2) + pow(ptX(1) - cam.C(1), 2) + pow(ptX(2) - cam.C(2), 2));

            //}else{f
            //	cout << ptX(0) << " " << ptX(1) << " " << ptX(2) << endl;
            //	cout << depthImg(y,x) << endl;
            //}

            // P *
            // cout << xValue << " " << yValue << " " << zValue << " / " <<
        }
    }
    fclose(outputPly);
    cout << "finished output ply in " << plyFilePath << endl;
}



void Save_3D_Model(GlobalState &gs,
                   Mat_<float> img_grayscale,
                   Camera cam,
                   InputFiles input_files,
                   string parentFolder)
{
    time_t now = time(0);
    char tmp[32] = {0};
    strftime(tmp, sizeof(tmp), "%Y-%m-%d-%H-%M-%S", localtime(&now)); //"%Y-%m-%d %H:%M:%S"
    // string temp = std::string(tmp) + "-" + input_files.img_filenames[REFERENCE] + ".ply";
    // char *plyFile = &temp[0];

    // string outputPath= parentFolder + "point-" + input_files.img_filenames[REFERENCE].substr ( 0, input_files.img_filenames[REFERENCE].length() - 4 ) +"-" +std::string(tmp) + ".ply";
    string outputPath = parentFolder + "point-" + input_files.img_filenames[gs.current_image].substr(0, input_files.img_filenames[gs.current_image].length() - 4) + "-" + std::string(tmp) + ".ply";

    char *plyFile = &outputPath[0];
    Mat_<Vec3f> norm0 = Mat::zeros(gs.rows, gs.cols, CV_32FC3);
    Mat_<float> cudadisp = Mat::zeros(gs.rows, gs.cols, CV_32FC1);

    for (int i = 0; i < gs.cols; i++) // 640
    {
        for (int j = 0; j < gs.rows; j++) // 480
        {

            int center = i + gs.cols * j;
            float4 n1 = gs.lines->norm4[center]; // 0724
            norm0(j, i) = Vec3f(n1.x,
                                n1.y,
                                n1.z);
            cudadisp(j, i) = n1.w;
        }
    }
    Mat cost_display;
    normalize(cudadisp, cost_display, 0, 65535, NORM_MINMAX, CV_16U);
    Mat_<float> disp0 = cudadisp.clone();
    Mat_<float> distImg;
    // gs.image_number = readKRtFileMiddlebury(dir_par, input_files, false);
    // cout<<"gs.image_number"<<endl;
    storePlyFileBinary(gs, plyFile, disp0, norm0, img_grayscale, cam, distImg);
}




















