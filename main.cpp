#include "main.h"

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <vector_types.h>
#include <ctime>
#include <chrono>
// #include <time.h>
// CUDA helper functions
// #include "driver_types.h"
// #include "helper_cuda.h"         // helper functions for CUDA error check

#include "globalstate.h"

#include "fileIoUtils.h"
#include "cameraGeometryUtils.h"
#include "displayUtils.h"

#include <sys/stat.h>    // mkdir
#include <sys/types.h>   // mkdir
#include "sys/sysinfo.h" // memory output

#define idx(i,j,lda) ( (j) + ((i)*(lda)) )

#include <map> // multimap
#ifdef _MSC_VER
#include <io.h>
#define R_OK 04
#else
#include <unistd.h>
#endif

using namespace chrono;
using namespace std;
vector<Camera> cameras;

struct sysinfo memInfo;

struct InputData
{
    string path;
    // int id;
    string id;
    int camId;
    Camera cam;
    Mat_<float> depthMap;
    Mat_<Vec3b> inputImage;
    Mat_<Vec3f> normals;
};

vector<Mat_<float>> disp_in_mem;
vector<Mat_<float>> normal_in_mem;

void print_memory_usage(int line_num)
{
    sysinfo(&memInfo);
    long long physMemUsed = memInfo.totalram - memInfo.freeram; // Physical Memory currently used
    physMemUsed *= memInfo.mem_unit;                            // Multiply to avoid int overflow on right hand side
    long long virtualMemUsed = memInfo.totalram - memInfo.freeram;
    virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
    virtualMemUsed *= memInfo.mem_unit;
    cout << line_num << "physMemUsed(Gb): " << physMemUsed / 1024.0 / 1024 / 1024 << ", virtualMemUsed(Gb): " << virtualMemUsed / 1024.0 / 1024 / 1024 << endl;
}

static void get_directory_entries(
    const char *dirname,
    vector<string> &directory_entries)
{
    DIR *dir;
    struct dirent *ent;

    // Open directory stream
    dir = opendir(dirname);
    if (dir != NULL)
    {
        // cout << "Dirname is " << dirname << endl;
        // cout << "Dirname type is " << ent->d_type << endl;
        // cout << "Dirname type DT_DIR " << DT_DIR << endl;

        // Print all files and directories within the directory
        while ((ent = readdir(dir)) != NULL)
        {
            // cout << "INSIDE" << endl;
            // if(ent->d_type == DT_DIR)
            {
                char *name = ent->d_name;
                if (strcmp(name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
                    continue;
                // printf ("dir %s/\n", name);
                directory_entries.push_back(string(name));
            }
        }

        closedir(dir);
    }
    else
    {
        // Could not open directory
        printf("Cannot open directory %s\n", dirname);
        exit(EXIT_FAILURE);
    }
    sort(directory_entries.begin(), directory_entries.end());
}

void addImageToTextureFloatColor(vector<Mat> &imgs, cudaTextureObject_t texs[]) // 0724
// void addImageToTextureFloatColor (vector<Mat > &imgs, cudaTextureObject_t texs[])
{

    // int rows = imgs[0].rows;
    // int cols = imgs[0].cols;
    // // Create channel with floating point type
    // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

    // // Allocate array with correct size and number of channels
    // cudaArray *cuArray;
    // checkCudaErrors(cudaMallocArray(&cuArray,
    //                                 &channelDesc,
    //                                 cols,
    //                                 rows));
    for (size_t i = 0; i < imgs.size(); i++)
    {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

        // Allocate array with correct size and number of channels
        cudaArray *cuArray; // 0724
        checkCudaErrors(cudaMallocArray(&cuArray,
                                        &channelDesc,
                                        cols,
                                        rows));

        checkCudaErrors(cudaMemcpy2DToArray(cuArray,
                                            0,
                                            0,
                                            imgs[i].ptr<float>(),
                                            imgs[i].step[0],
                                            cols * sizeof(float) * 4,
                                            rows,
                                            cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        // resDesc.res.array.array = cuArray;//0724
        resDesc.res.array.array = cuArray;

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        // cudaTextureObject_t &texObj = texs[i];
        checkCudaErrors(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
        // checkCudaErrors(cudaFreeArray(cuArray));
    }

    return;
}

// Taken from Gipuma
void addImageToTextureFloatGray(vector<Mat> &imgs, cudaTextureObject_t texs[])
// void addImageToTextureFloatGray(vector<Mat> &imgs, cudaTextureObject_t texs[])
{

    //  int rows = imgs[0].rows;
    // int cols = imgs[0].cols;
    // // Create channel with floating point type
    // cudaChannelFormatDesc channelDesc =
    //     cudaCreateChannelDesc(32,
    //                             0,
    //                             0,
    //                             0,
    //                             cudaChannelFormatKindFloat);
    // // Allocate array with correct size and number of channels
    // cudaArray *cuArray;
    // checkCudaErrors(cudaMallocArray(&cuArray,
    //                                 &channelDesc,
    //                                 cols,
    //                                 rows));
    for (size_t i = 0; i < imgs.size(); i++)
    {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(32,
                                  0,
                                  0,
                                  0,
                                  cudaChannelFormatKindFloat);
        // Allocate array with correct size and number of channels
        cudaArray *cuArray;
        checkCudaErrors(cudaMallocArray(&cuArray,
                                        &channelDesc,
                                        cols,
                                        rows));

        checkCudaErrors(cudaMemcpy2DToArray(cuArray,
                                            0,
                                            0,
                                            imgs[i].ptr<float>(),
                                            imgs[i].step[0],
                                            cols * sizeof(float),
                                            rows,
                                            cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        // cudaTextureObject_t &texObj = texs[i];
        checkCudaErrors(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
        // checkCudaErrors(cudaFreeArray(cuArray)); 
        // texs[i] = texObj;
    }

    // checkCudaErrors(cudaFreeArray(cuArray)); 

    return;
}




static void selectViews(CameraParameters &cameraParams, int imgWidth, int imgHeight, AlgorithmParameters &algParams, int current_image)
{
    vector<Camera> &cameras = cameraParams.cameras;
    Camera ref = cameras[cameraParams.idRef];

    // The center of the image
    int x = imgWidth / 2;
    int y = imgHeight / 2;

    cout << "algParams.viewSelection:" << algParams.viewSelection << endl;

    cameraParams.viewSelectionSubset.clear();

    Vec3f viewVectorRef = getViewVector(ref, x, y);

    // TODO hardcoded value makes it a parameter
    float minimum_angle_degree = algParams.min_angle;
    float maximum_angle_degree = algParams.max_angle;

    unsigned int maximum_view = algParams.max_views;
    float minimum_angle_radians = minimum_angle_degree * M_PI / 180.0f;
    float maximum_angle_radians = maximum_angle_degree * M_PI / 180.0f;
    float min_depth = 9999;
    float max_depth = 0;
    if (algParams.viewSelection)
        printf("Accepting intersection angle of central rays from %f to %f degrees, use --min_angle=<angle> and --max_angle=<angle> to modify them\n", minimum_angle_degree, maximum_angle_degree);

    for (size_t i = 0; i < cameras.size(); i++)
    {
        // if ( !algParams.viewSelection ) { //select all views, dont perform selection
        // cameraParams.viewSelectionSubset.push_back ( i );
        // continue;
        //}

        Vec3f vec = getViewVector(cameras[i], x, y);

        float baseline = static_cast<float>(norm(cameras[current_image].C, cameras[i].C)); // 0719?
        float angle = getAngle(viewVectorRef, vec);
        if (angle > minimum_angle_radians &&
            angle < maximum_angle_radians) // 0.6 select if angle between 5.7 and 34.8 (0.6) degrees (10 and 30 degrees suggested by some paper)
        {
            if (algParams.viewSelection)
            {

                cameraParams.viewSelectionSubset.push_back(static_cast<int>(i));

                // printf("\taccepting camera %ld with angle\t %f degree (%f radians) and baseline %f\n", i, angle*180.0f/M_PI, angle, baseline);
            }
            float min_range = (baseline / 2.0f) / sin(maximum_angle_radians / 2.0f);
            float max_range = (baseline / 2.0f) / sin(minimum_angle_radians / 2.0f);
            min_depth = std::min(min_range, min_depth);
            max_depth = std::max(max_range, max_depth);
            // printf("Min max ranges are %f %f\n", min_range, max_range);
            // printf("Min max depth are %f %f\n", min_depth, max_depth);
        }
        // else
        // printf("Discarding camera %ld with angle\t %f degree (%f radians) and baseline, %f\n", i, angle*180.0f/M_PI, angle, baseline);
    }


    if (algParams.depthMin == -1)
        algParams.depthMin = min_depth;
    if (algParams.depthMax == -1)
        algParams.depthMax = max_depth;

    cout << "cameras.size ():" << cameras.size() << endl;
    if (!algParams.viewSelection)
    {
        cameraParams.viewSelectionSubset.clear();
        for (size_t i = 1; i < cameras.size(); i++)
            cameraParams.viewSelectionSubset.push_back(static_cast<int>(i));
        return;
    }
    cout << "algParams.viewSelection" << algParams.viewSelection << endl;
    cout << "cameraParams.viewSelectionSubset.size():" << cameraParams.viewSelectionSubset.size() << endl;
    cout << "cameraParams.viewSelectionSubset.size():" << cameraParams.viewSelectionSubset.size() << endl;
}



static void consistency_selectViews(CameraParameters &cameraParams, int imgWidth, int imgHeight, int refID, bool viewSel, int min_angle, int max_angle)
{
    vector<Camera> cameras = cameraParams.cameras;
    Camera ref = cameras[refID];

    int x = imgWidth / 2;
    int y = imgHeight / 2;

    cameraParams.viewSelectionSubset.clear();
    vector<int> subset;

    Vec3f viewVectorRef = getViewVector(ref, x, y);

    float minimum_angle_degree = min_angle * 1.0;
    float maximum_angle_degree = max_angle * 1.0;

    float minimum_angle_radians = minimum_angle_degree * M_PI / 180.0f;
    float maximum_angle_radians = maximum_angle_degree * M_PI / 180.0f;
    // printf("Accepted intersection angle of central rays is %f to %f degrees\n", minimum_angle_degree, maximum_angle_degree);
    for (size_t i = 0; i < cameras.size(); i++)
    {

        if (!viewSel)
        { // select all views, dont perform selection
            subset.push_back(i);
            continue;
        }

        Vec3f vec = getViewVector(cameras[i], x, y);

        float angle = getAngle(viewVectorRef, vec);
        if (angle > minimum_angle_radians && angle < maximum_angle_radians) // 0.6 select if angle between 5.7 and 34.8 (0.6) degrees (10 and 30 degrees suggested by some paper)
        {
            subset.push_back(i);
            // printf("Accepting camera %ld with angle\t %f degree (%f radians)\n", i, angle*180.0f/M_PI, angle);
        }
       
    }

    cameraParams.selectedViews.push_back(subset);
}


//
static void consistency_addImageToTextureFloatColor(vector<Mat> &imgs, cudaTextureObject_t texs[], cudaArray *cuArray[])
{

    // int rows = imgs[0].rows;
    // int cols = imgs[0].cols;
    // // Create channel with floating point type
    // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

    // // Allocate array with correct size and number of channels
    // cudaArray *cuArray;
    // checkCudaErrors(cudaMallocArray(&cuArray,
    //                                 &channelDesc,
    //                                 cols,
    //                                 rows));
    for (size_t i = 0; i < imgs.size(); i++)
    {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

        // Allocate array with correct size and number of channels
        // cudaArray *cuArray;

        checkCudaErrors(cudaMemcpy2DToArray(cuArray[i],
                                            0,
                                            0,
                                            imgs[i].ptr<float>(),
                                            imgs[i].step[0],
                                            cols * sizeof(float) * 4,
                                            rows,
                                            cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        // texDesc.filterMode = cudaFilterModeLinear;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        checkCudaErrors(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
        // checkCudaErrors(cudaFreeArray(cuArray));
    }

    return;
}

//
void cu_Free_cuArray(int num_images, cudaArray *cuArray[])
{
    for (int i = 0; i < num_images; i++)
    {
        // cudaDestroyTextureObject(texs[i]);
        cudaFreeArray(cuArray[i]);
    }
}

void free_data(int num_images, GlobalState *gs)
{
    for (int i = 0; i < num_images; i++)
    {
        free(gs->dataArray[i]);
        free(gs->dataArray2[i]);
    }
}

int getParametersFromCommandLine_all(int argc,
                                     char **argv,
                                     InputFiles &inputFiles,
                                     OutputFiles &outputFiles,
                                     AlgorithmParameters &algParameters)
{
    cout << endl
         << endl
         << "---------------------------Reading params from command line arguments-------------------------------" << endl;

    // InputFiles inputFiles;
    string ext = ".png";

    string results_folder = "results/";

    const char *algorithm_opt = "--algorithm=";
    const char *blocksize_opt = "--blocksize=";
    const char *cost_choice_opt = "--cost_choice=";
    const char *colorProc_opt = "--color_processing=";
    const char *num_iterations_opt = "--iterations=";
    const char *cam_scale_opt = "--cam_scale=";
    const char *cost_gamma_opt = "--cost_gamma=";
    const char *n_best_opt = "--n_best=";
    const char *cost_comb_opt = "--cost_comb=";
    const char *depth_min_opt = "--depth_min=";
    const char *depth_max_opt = "--depth_max=";
    const char *min_angle_opt = "--min_angle=";
    const char *max_angle_opt = "--max_angle=";
    const char *viewSelection_opt = "-view_selection";
    const char *max_views_opt = "--max_views=";
    const char *cycles_opt = "--cycles=";

    const char *outputPath_opt = "-output_folder";

    const char *results_folder_opt = "-input_folder";
    const char *p_input_folder_opt = "-p_folder";
    const char *in_ex_folder_opt = "-in_ex_folder";
    const char *krt_file_opt = "-krt_file";
    const char *images_input_folder_opt = "-images_folder";
    const char *gt_opt = "-gt";
    const char *gt_nocc_opt = "-gt_nocc";
    const char *pmvs_folder_opt = "--pmvs_folder";
    const char *remove_black_background_opt = "-remove_black_background";
    const char *min_angle_degree_opt = "--min_angle_degree=";
    const char *max_angle_degree_opt = "--max_angle_degree=";
    const char *warm_up_iters_opt = "--warm_up_iters=";

    const char *disp_thresh_opt = "--disp_thresh=";
    const char *normal_thresh_opt = "--normal_thresh=";
    const char *num_consistent_opt = "--num_consistent=";
    const char *num_best_opt = "--num_best=";

    const char *number_consistent_input_opt = "--number_consistent_input=";

    // read in arguments
    int c = 0;
    for (int i = 1; i < argc; i++)
    {
        
        if (argv[i][0] != '-')
        {
            // Note: In case of not able to process the input image
            if (c % 2 == 1)
            // if (c % 5 == 0)
            // if (c % 3 < 2)
            // if (c % 2 == 0)
            // if (c % 4 < 3)
            // if (c % 5 == 0 || c % 5 == 2 || c % 5 == 4 )
            // if (c % 5 < 4)
            {
                inputFiles.img_filenames.push_back(argv[i]);
                printf("argv[i] : %s\n", argv[i]);
            }
            c++;
        }
        else if (strncmp(argv[i], algorithm_opt, strlen(algorithm_opt)) == 0)
        {
            char *_alg = argv[i] + strlen(algorithm_opt);
            // algParameters.algorithm = strcmp ( _alg, "pm" ) == 0 ? PM_COST :
            // strcmp ( _alg, "ct" ) == 0 ? CENSUS_TRANSFORM :
            // strcmp ( _alg, "sct" ) == 0 ? SPARSE_CENSUS :
            // strcmp ( _alg, "ct_ss" ) == 0 ? CENSUS_SELFSIMILARITY :
            // strcmp ( _alg, "adct" ) == 0 ? ADCENSUS :
            // strcmp ( _alg, "adct_ss" ) == 0 ? ADCENSUS_SELFSIMILARITY :
            // strcmp ( _alg, "pm_ss" ) == 0 ? PM_SELFSIMILARITY : -1;
            algParameters.algorithm = strcmp(_alg, "pm") == 0 ? 0 : strcmp(_alg, "ct") == 0    ? 1
                                                                : strcmp(_alg, "sct") == 0     ? 2
                                                                : strcmp(_alg, "ct_ss") == 0   ? 3
                                                                : strcmp(_alg, "adct") == 0    ? 5
                                                                : strcmp(_alg, "adct_ss") == 0 ? 6
                                                                : strcmp(_alg, "pm_ss") == 0   ? 4
                                                                                               : -1;
            if (algParameters.algorithm < 0)
            {
                printf("Command-line parameter error: Unknown stereo algorithm\n\n");
                return -1;
            }
        }
        else if (strncmp(argv[i], cost_comb_opt, strlen(cost_comb_opt)) == 0)
        {
            char *_alg = argv[i] + strlen(cost_comb_opt);
            // algParameters.cost_comb = strcmp ( _alg, "all" ) == 0 ? COMB_ALL :
            // strcmp ( _alg, "best_n" ) == 0 ? COMB_BEST_N :
            // strcmp ( _alg, "angle" ) == 0 ? COMB_ANGLE :
            // strcmp ( _alg, "good" ) == 0 ? COMB_GOOD : -1;
            algParameters.cost_comb = strcmp(_alg, "all") == 0 ? 0 : strcmp(_alg, "best_n") == 0 ? 1
                                                                 : strcmp(_alg, "angle") == 0    ? 2
                                                                 : strcmp(_alg, "good") == 0     ? 3
                                                                                                 : -1;
            cout << "algParameters.cost_comb(COMB_ALL = 0, COMB_BEST_N = 1, COMB_ANGLE = 2, COMB_GOOD = 3): " << algParameters.cost_comb << endl;
            if (algParameters.cost_comb < 0)
            {
                printf("Command-line parameter error: Unknown cost combination method\n\n");
                return -1;
            }
        }
        else if (strncmp(argv[i], cost_choice_opt, strlen(cost_choice_opt)) == 0)
        {
            char *_alg = argv[i] + strlen(cost_choice_opt);
            algParameters.cost_choice =
                strcmp(_alg, "weighted_difference") == 0 ? 1 : strcmp(_alg, "bncc") == 0 ? 2
                                                           : strcmp(_alg, "angle") == 0  ? 3
                                                           : strcmp(_alg, "good") == 0   ? 4
                                                                                         : -1;
            cout << "algParameters.cost_choice(1:weighted_difference, 2:bncc): " << algParameters.cost_choice << endl;
            if (algParameters.cost_choice < 0)
            {
                printf("Command-line parameter error: Unknown cost_choice method\n\n");
                return -1;
            }
        }
        else if (strncmp(argv[i], blocksize_opt, strlen(blocksize_opt)) == 0)
        {
            int k_size;
            if (sscanf(argv[i] + strlen(blocksize_opt), "%d", &k_size) != 1 ||
                k_size < 1 || k_size % 2 != 1)
            {
                printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
                return -1;
            }
            algParameters.box_hsize = k_size;
            algParameters.box_vsize = k_size;
        }
        else if (strncmp(argv[i], num_iterations_opt, strlen(num_iterations_opt)) == 0)
        {
            sscanf(argv[i] + strlen(num_iterations_opt), "%d", &algParameters.iterations);
            cout << "algParameters.iterations: " << algParameters.iterations << endl;
        }
        else if (strncmp(argv[i], cam_scale_opt, strlen(cam_scale_opt)) == 0)
        {
            sscanf(argv[i] + strlen(cam_scale_opt), "%f", &algParameters.cam_scale);
            cout << "algParameters.cam_scale: " << algParameters.cam_scale << endl;
        }
        else if (strncmp(argv[i], depth_min_opt, strlen(depth_min_opt)) == 0)
        {
            sscanf(argv[i] + strlen(depth_min_opt), "%f", &algParameters.depthMin);
            cout << "algParameters.depthMin: " << algParameters.depthMin << endl;
        }
        else if (strncmp(argv[i], depth_max_opt, strlen(depth_max_opt)) == 0)
        {
            sscanf(argv[i] + strlen(depth_max_opt), "%f", &algParameters.depthMax);
            cout << "algParameters.depthMax: " << algParameters.depthMax << endl;
        }
        else if (strncmp(argv[i], min_angle_opt, strlen(min_angle_opt)) == 0)
        {
            sscanf(argv[i] + strlen(min_angle_opt), "%f", &algParameters.min_angle);
            cout << "algParameters.min_angle: " << algParameters.min_angle << endl;
        }
        else if (strncmp(argv[i], max_angle_opt, strlen(max_angle_opt)) == 0)
        {
            sscanf(argv[i] + strlen(max_angle_opt), "%f", &algParameters.max_angle);
            cout << "algParameters.max_angle: " << algParameters.max_angle << endl;
        }
        else if (strncmp(argv[i], max_views_opt, strlen(max_views_opt)) == 0)
        {
            sscanf(argv[i] + strlen(max_views_opt), "%u", &algParameters.max_views);
            cout << "algParameters.max_views: " << algParameters.max_views << endl;
        }
        else if (strncmp(argv[i], cycles_opt, strlen(cycles_opt)) == 0)
        {
            sscanf(argv[i] + strlen(cycles_opt), "%d", &algParameters.cycles);
            cout << "algParameters.cycles: " << algParameters.cycles << endl;
        }

        else if (strcmp(argv[i], viewSelection_opt) == 0)
        {
            algParameters.viewSelection = true;
            cout << "algParameters.viewSelection (1:true, 0:false):" << algParameters.viewSelection << endl;
        }
        else if (strncmp(argv[i], colorProc_opt, strlen(colorProc_opt)) == 0)
        {
            // algParameters.color_processing = true;
            int temp;
            sscanf(argv[i] + strlen(colorProc_opt), "%d", &temp);
            if (temp == 1)
                algParameters.color_processing = true;
            else
                algParameters.color_processing = false;
            cout << "algParams.color_processing(1:COLOR, 0:GRAY): " << algParameters.color_processing << endl;
        }
        else if (strcmp(argv[i], outputPath_opt) == 0)
        {
            outputFiles.parentFolder = argv[++i];
            cout << "outputFiles.parentFolder: " << outputFiles.parentFolder << endl;
        }
        else if (strncmp(argv[i], cost_gamma_opt, strlen(cost_gamma_opt)) == 0)
        {
            sscanf(argv[i] + strlen(cost_gamma_opt), "%f", &algParameters.gamma);
            cout << "algParameters.gamma: " << algParameters.gamma << endl;
        }
        else if (strncmp(argv[i], n_best_opt, strlen(n_best_opt)) == 0)
        {
            sscanf(argv[i] + strlen(n_best_opt), "%d", &algParameters.n_best);
            cout << "algParameters.n_best: " << algParameters.n_best << endl;
        }

        else if (strcmp(argv[i], results_folder_opt) == 0)
        {
            inputFiles.results_folder = argv[++i];
            cout << "inputFiles.results_folder: " << inputFiles.results_folder << endl;
        }
        else if (strcmp(argv[i], p_input_folder_opt) == 0)
        {
            inputFiles.p_folder = argv[++i];
            cout << "inputFiles.p_folder: " << inputFiles.p_folder << endl;
        }
        else if (strcmp(argv[i], in_ex_folder_opt) == 0)
        {
            inputFiles.in_ex_folder = argv[++i];
            cout << "in_ex_folder: " << inputFiles.in_ex_folder << endl;
        }
        else if (strcmp(argv[i], krt_file_opt) == 0)
        {
            inputFiles.krt_file = argv[++i];
            cout << "inputFiles.krt_file: " << inputFiles.krt_file << endl;
        }
        else if (strcmp(argv[i], images_input_folder_opt) == 0)
        {
            // int j=i;
            inputFiles.images_folder = argv[++i];

            // cout <<"1" << argv[j] << endl;
            // cout <<"2" << argv[j+1] << endl;
            // cout <<"3" << argv[j+2] << endl;
            cout << "inputFiles.images_folder: " << inputFiles.images_folder << endl;
        }
        else if (strcmp(argv[i], gt_opt) == 0)
        {
            inputFiles.gt_filename = argv[++i];
            cout << "inputFiles.gt_filename: " << inputFiles.gt_filename << endl;
        }
        else if (strcmp(argv[i], gt_nocc_opt) == 0)
        {
            inputFiles.gt_nocc_filename = argv[++i];
            cout << "inputFiles.gt_nocc_filename: " << inputFiles.gt_nocc_filename << endl;
        }
        else if (strncmp(argv[i], pmvs_folder_opt, strlen(pmvs_folder_opt)) == 0)
        {
            inputFiles.pmvs_folder = argv[++i];
            cout << "inputFiles.pmvs_folder: " << inputFiles.pmvs_folder << endl;
        }
        else if (strcmp(argv[i], remove_black_background_opt) == 0)
        {
            algParameters.remove_black_background = true;
            cout << "remove_black_background(1: remove): " << algParameters.remove_black_background << endl;
        }
        else if (strncmp(argv[i], warm_up_iters_opt, strlen(warm_up_iters_opt)) == 0)
        {
            sscanf(argv[i] + strlen(warm_up_iters_opt), "%d", &algParameters.warm_up_iters);
            cout << "algParameters.warm_up_iters: " << algParameters.warm_up_iters << endl;
        }
        else if (strncmp(argv[i], min_angle_degree_opt, strlen(min_angle_degree_opt)) == 0)
        {
            sscanf(argv[i] + strlen(min_angle_degree_opt), "%d", &algParameters.min_angle_degree);
            cout << "algParameters.min_angle_degree: " << algParameters.min_angle_degree << endl;
        }
        else if (strncmp(argv[i], max_angle_degree_opt, strlen(max_angle_degree_opt)) == 0)
        {
            sscanf(argv[i] + strlen(max_angle_degree_opt), "%d", &algParameters.max_angle_degree);
            cout << "algParameters.max_angle_degree: " << algParameters.max_angle_degree << endl;
        }
        else if (strncmp(argv[i], disp_thresh_opt, strlen(disp_thresh_opt)) == 0)
        {
            sscanf(argv[i] + strlen(disp_thresh_opt), "%f", &algParameters.depthThresh);
            cout << "algParameters.depthThresh:" << algParameters.depthThresh << endl;
        }
        else if (strncmp(argv[i], normal_thresh_opt, strlen(normal_thresh_opt)) == 0)
        {
            float angle_degree;
            sscanf(argv[i] + strlen(normal_thresh_opt), "%f", &angle_degree);
            algParameters.normalThresh = angle_degree * M_PI / 180.0f;
            cout << "algParameters.normalThresh:" << algParameters.normalThresh * 180 / M_PI << endl;
        }
        else if (strncmp(argv[i], num_consistent_opt, strlen(num_consistent_opt)) == 0)
        {
            sscanf(argv[i] + strlen(num_consistent_opt), "%d", &algParameters.numConsistentThresh);
            cout << "algParameters.numConsistentThresh:" << algParameters.numConsistentThresh << endl;
        }
        // else if (strncmp(argv[i], min_angle_degree_opt, strlen(min_angle_degree_opt)) == 0)
        // {
        //     sscanf(argv[i] + strlen(min_angle_degree_opt), "%d", &algParameters.min_angle_degree);
        //     cout << "min_angle_degree_opt: " << algParameters.min_angle_degree << endl;
        // }
        // else if (strncmp(argv[i], max_angle_degree_opt, strlen(max_angle_degree_opt)) == 0)
        // {
        //     sscanf(argv[i] + strlen(max_angle_degree_opt), "%d", &algParameters.max_angle_degree);
        //     cout << "max_angle_degree_opt: " << algParameters.max_angle_degree << endl;
        // }
        else if (strncmp(argv[i], number_consistent_input_opt, strlen(number_consistent_input_opt)) == 0)
        {
            sscanf(argv[i] + strlen(number_consistent_input_opt), "%d", &algParameters.number_consistent_input);
            cout << "algParameters.number_consistent_input: " << algParameters.number_consistent_input << endl;
        }
        else if (strncmp(argv[i], num_best_opt, strlen(num_best_opt)) == 0)
        {
            sscanf(argv[i] + strlen(num_best_opt), "%d", &algParameters.num_best);
            cout << "algParameters.num_best: " << algParameters.num_best << endl;
        }
        else
        {
            printf("Command-line parameter error: unknown option %s\n", argv[i]);
            // return -1;
        }
    }
    cout << "inputFiles.img_filenames.size(): " << inputFiles.img_filenames.size() << endl;

    cout << endl
         << "---------------------------Finished Reading params from command line arguments-------------------------------" << endl
         << endl;
    return 0;
}




/* Copy point cloud to global memory */
// template< typename T >
void copy_point_cloud_to_host_cpu(GlobalState &gs, int cam, PointCloudList &pc_list)
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
    // printf("Found %.7f million points\n", count / 1000000.0f);
    pc_list.size = count;
}


void get3Dpoint_cpu(float4 *ptX, const Camera_cu &cam, const int2 &p, const float &depth)
{
    // in case camera matrix is not normalized: see page 162, then depth might not be the real depth but w and depth needs to be computed from that first
    const float4 pt = make_float4(
        depth * (float)p.x - cam.P_col34.x,
        depth * (float)p.y - cam.P_col34.y,
        depth - cam.P_col34.z,
        0);

    matvecmul4(cam.M_inv, pt, ptX);
}


// void get3Dpoint_cpu(float4 *ptX, const Camera_cu &cam, const int2 &p)
// {
//     // in case camera matrix is not normalized: see page 162, then depth might not be the real depth but w and depth needs to be computed from that first
//     float4 pt;
//     pt.x = (float)p.x - cam.P_col34.x;
//     pt.y = (float)p.y - cam.P_col34.y;
//     pt.z = 1.0f - cam.P_col34.z;

//     matvecmul4(cam.M_inv, pt, ptX);
// }

float4 operator+(float4 a, float4 b) {
    return make_float4(a.x+b.x,
                       a.y+b.y,
                       a.z+b.z,
                       0);
}

float4 operator-(float4 a) {
    return make_float4(-a.x,
                       -a.y,
                       -a.z,
                       0);
}

float4 operator-(float4 a, float4 b) {
    return make_float4(a.x-b.x,
                       a.y-b.y,
                       a.z-b.z,
                       0);
}

float4 operator/(float4 a, float k) {
    return make_float4(a.x/k,
                       a.y/k,
                       a.z/k,
                       0);
}

float l2_float4 (float4 a) {
    return sqrtf( pow2 (a.x) +
             pow2 (a.y) +
             pow2 (a.z));

}

void project_on_camera_cpu(const float4 &X, const Camera_cu &cam, float2 *pt, float *depth)
{
    float4 tmp = make_float4(0, 0, 0, 0);
    matvecmul4P(cam.P, X, (&tmp));
    pt->x = tmp.x / tmp.z;
    pt->y = tmp.y / tmp.z;
    *depth = tmp.z;
}



void NormalizeVec3_cpu (float4 *vec)
{
    const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
    const float inverse_sqrt = pow (normSquared, -0.5); // return pow(x, -0.5); rsqrtf
    vec->x *= inverse_sqrt;
    vec->y *= inverse_sqrt;
    vec->z *= inverse_sqrt;
}


float disparityDepthConversion_cpu(const float &f, const Camera_cu &cam_ref, const Camera_cu &cam, const float &d)
{
    float baseline = l2_float4(cam_ref.C4 - cam.C4);
    return f * baseline / d;
}

float getAngle_cpu(const float4 &v1, const float4 &v2)
{
    float angle = acosf(dot4(v1, v2));
    // if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
    if (angle != angle)
        return 0.0f;

    return angle;
}


void ambc_fusibile_cpu(GlobalState &gs, int ref_camera, vector<Mat> normals_and_depth)
{
    // int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    // const int cols = gs.cols;
    // const int rows = gs.rows;

    // if (p.x >= cols)
    //     return;
    // if (p.y >= rows)
    //     return;

    // const int center = p.y * cols + p.x;
    const CameraParameters_cu &camParams = *(gs.cameras);
    
    for (int x = 3; x < gs.cols-3; x++)
    {
        for (int y = 3; y < gs.rows-3; y++)
        {
            int center = y * gs.cols + x;
            // printf("center: %d, x: %d, y: %d\n", center, x, y);
            if (gs.lines_consistency[ref_camera].used_pixels[center] == 1) 
                continue;

            
            const float4 normal = normals_and_depth[ref_camera].at<float4>(y,x); //tex2D<float4>(gs.normals_depths[ref_camera], p.x , p.y );
            float depth = normal.w;

            float4 X;
            int2 p = make_int2(x, y);
            get3Dpoint_cpu(&X, camParams.cameras[ref_camera], p, depth);

            float4 consistent_X = X;
            float4 consistent_normal = normal;

            float consistent_texture = 0;// tex2D<float>(gs.imgs[ref_camera], p.x + 0.5f, p.y + 0.5f); //TODO: Add color
            int number_consistent = 0;
            // int2 used_list[camParams.viewSelectionSubsetNumber];
            int2 used_list[MAX_IMAGES];

               
            /*
            * For each point of the reference camera compute the 3d position corresponding to the corresponding depth.
            * Create a point only if the following conditions are fulfilled:
            * - Projected depths of other cameras does not differ more than gs.params.depthThresh
            * - Angle of normal does not differ more than gs.params.normalThresh
            */
   
            if (false)
            {
                float4 v_ref; // view vector between the 3D point and the reference camera
                float4 v_source;  // view vector between the 3D point and the souce camera
                v_ref.x = normal.x;
                v_ref.y = normal.y;
                v_ref.z = normal.z;
                NormalizeVec3_cpu(&v_ref);
                v_source.x = camParams.cameras[ref_camera].C4.x - X.x;
                v_source.y = camParams.cameras[ref_camera].C4.y - X.y;
                v_source.z = camParams.cameras[ref_camera].C4.z - X.z;
                NormalizeVec3_cpu(&v_source);
                float rad1 = acosf ( v_ref.x * v_source.x + v_ref.y * v_source.y + v_ref.z * v_source.z);
                float maximum_angle_radians = 80 * M_PI / 180.0f;

                
                if (rad1 > maximum_angle_radians )
                {
                    continue;
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
                project_on_camera_cpu(X, camParams.cameras[idxCurr], &source_pixel, &depth);

                // Boundary check
                if (source_pixel.x >= 5 &&
                    source_pixel.x < gs.cols-5 &&
                    source_pixel.y >= 5 &&
                    source_pixel.y < gs.rows-5)
                {
                    // printf("Boundary check passed\n");

                    // Compute interpolated depth and normal for source_pixel w.r.t. camera ref_camera
                    float4 source; // first 3 components normal, fourth depth
                    source = normals_and_depth[idxCurr].at<float4>(static_cast<int>(source_pixel.y + 0.5f),static_cast<int>(source_pixel.x + 0.5f));
                    // source = tex2D<float4>(gs.normals_depths[idxCurr], static_cast<int>(source_pixel.x + 0.5f), static_cast<int>(source_pixel.y + 0.5f) );

                    const float depth_disp = disparityDepthConversion_cpu  ( camParams.cameras[ref_camera].f, camParams.cameras[ref_camera], camParams.cameras[idxCurr], depth );
                    const float source_disp = disparityDepthConversion_cpu ( camParams.cameras[ref_camera].f, camParams.cameras[ref_camera], camParams.cameras[idxCurr], source.w );
                    // First consistency check on depth

                    if (fabsf(depth_disp - source_disp) < gs.params->depthThresh)
                    {

                        float angle = getAngle_cpu(source, normal); // extract normal
                        if (angle < gs.params->normalThresh)
                        {

                            /// All conditions met:
                            //  - average 3d points and normals
                            //  - save resulting point and normal
                            //  - (optional) average texture (not done yet)

                            int2 tmp_p = make_int2(static_cast<int>(source_pixel.x + 0.5f), static_cast<int>(source_pixel.y + 0.5f));
                            float4 tmp_X; // 3d point of consistent point on other view
                            get3Dpoint_cpu(&tmp_X, camParams.cameras[idxCurr], tmp_p, source.w);

                            consistent_X = consistent_X + tmp_X;
                            consistent_normal = consistent_normal + source;
                            // if (gs.params->saveTexture) //TODO: save color
                            //     consistent_texture = consistent_texture + tex2D<float>(gs.imgs[idxCurr], source_pixel.x + 0.5f, source_pixel.y + 0.5f);

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

            int num_c = 5;
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
                        gs.pc->points[center].texture = 0;//consistent_texture; TODO: save color texture
                }
                    

                

                //// Mark corresponding point on other views as "used"
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
                        gs.lines_consistency[idxCurr].used_pixels[used_list[idxCurr].x + used_list[idxCurr].y * gs.cols] = 1;
                    }
                }
                
            }


        }
    }

   
    return;
}






template <typename T>
void fusibile_cpu(GlobalState &gs, PointCloudList &pc_list, int num_views, vector<Mat> normals_and_depth)
{

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


    // printf("Computing final disparity\n");
    // for (int cam=0; cam<10; cam++) {
    printf("num_views: %d\n", num_views);

    for (int cam = 0; cam < num_views; cam++)
    {

        
        if(gs.cycles == gs.params->cycles - 1)
        {
            ambc_fusibile_cpu(gs, cam, normals_and_depth);
        }
        else
        {
            // ambc_strong_new<<<grid_size_initrand, block_size_initrand>>>(gs, cam);
            // cudaDeviceSynchronize();
            // ambc_weak<<<grid_size_initrand, block_size_initrand>>>(gs, cam);
            ambc_fusibile_cpu(gs, cam, normals_and_depth);
        }
  
        copy_point_cloud_to_host_cpu(gs, cam, pc_list); // slower but saves memory

    }

    printf("run fusibile cpu finished\n");

}



int runconsistency_cpu(GlobalState &gs, PointCloudList &pc_list, int num_views, vector<Mat> normals_and_depth)
{
    printf("runconsistency\n");
    /*GlobalState *gs = new GlobalState;*/
    // if (gs.params->color_processing)
    // {
    //     printf("color processing\n");
    //     fusibile_cpu<float4>(gs, pc_list, num_views, normals_and_depth);
    // }
    // else
    {
        printf("gray processing\n");
        fusibile_cpu<float>(gs, pc_list, num_views, normals_and_depth);
    }
    // cudaDeviceReset();
    printf("runconsistency finished\n");
    return 0;
}




//
int consistency_main(InputFiles inputFiles,
                     AlgorithmParameters &algParameters,
                     GlobalState *gs)
{
    cout << endl
         << "------------Run consistency_main----------------------------------------" << endl
         << endl;
 


    if (inputFiles.pmvs_folder.size() > 0)
    {
        inputFiles.images_folder = inputFiles.pmvs_folder + "/visualize/";
        inputFiles.p_folder = inputFiles.pmvs_folder + "/txt/";
    }
    cout << "image folder is " << inputFiles.images_folder << endl;
    cout << "p folder is " << inputFiles.p_folder << endl;
    cout << "pmvs folder is " << inputFiles.pmvs_folder << endl;
    cout << "krt_file folder is " << inputFiles.krt_file << endl;
    cout << "min_angle_degree_opt: " << algParameters.min_angle_degree << endl;
    cout << "max_angle_degree_opt: " << algParameters.max_angle_degree << endl;

  
    char output_folder[256];
    sprintf(output_folder, "%sconsistencyCheck/", inputFiles.results_folder.c_str());
    // sprintf(output_folder, "%s/consistencyCheck-%04d%02d%02d-%02d%02d%02d/",results_folder.c_str(), pTime->tm_year+1900, pTime->tm_mon+1,pTime->tm_mday,pTime->tm_hour, pTime->tm_min, pTime->tm_sec);
    // cout << "output_folder:" << output_folder << endl;

#if defined(_WIN32)
    _mkdir(output_folder);
#else
    mkdir(output_folder, 0777);
#endif

    cout << "output_folder: " << output_folder << endl;
    vector<string> subfolders;

    get_subfolders(inputFiles.results_folder.c_str(), subfolders);
    std::sort(subfolders.begin(), subfolders.end());
    // rearrange the folder and choose first num folders.
    for (auto i = subfolders.begin(); i != subfolders.end(); ++i)
        std::cout << "subfolder_name:" << *i << endl;


    // int numImages = gs->num_images;
    // cout << "numImages is " << gs->num_images << endl;
    // algParameters.num_img_processed = min((int)gs->num_images, algParameters.num_img_processed);

    // inputFiles.num_images = gs->num_images;
    // cout << "inputFiles.num_images " << inputFiles.num_images << endl;
    // cout << "algParameters.num_img_processed: " << algParameters.num_img_processed << endl; // images as ref images. normally 1.

    size_t avail;
    size_t total;
    cudaMemGetInfo(&avail, &total);
    size_t used = total - avail;
    printf("Device memory used: %fMB\n", used / 1000000.0f);

    uint32_t rows = gs->rows;
    uint32_t cols = gs->cols;

    CameraParameters camParams = getCameraParameters(*(gs->cameras),
                                                         inputFiles,
                                                         algParameters.cam_scale,
                                                         false,
                                                         gs->current_image);

    for (int i = 0; i < algParameters.num_img_processed; i++)
    {
        algParameters.min_disparity = disparityDepthConversion(camParams.f, camParams.cameras[i].baseline, camParams.cameras[i].depthMax);
        algParameters.max_disparity = disparityDepthConversion(camParams.f, camParams.cameras[i].baseline, camParams.cameras[i].depthMin);
    }
    cout << "algParameters.min_disparity:" << algParameters.min_disparity << endl;
    cout << "algParameters.max_disparity:" << algParameters.max_disparity << endl;

    // int numSelViews;
    for (int j = 0; j < gs->num_images; j++)
    {

        consistency_selectViews(camParams, cols, rows, j, true, algParameters.min_angle_degree, algParameters.max_angle_degree); // 对所有视角进行能见度筛选
        // int numSelViews = camParams.viewSelectionSubset.size ();
        int numSelViews = camParams.selectedViews[j].size();
        cout << "for image: " << j << ", Selected views: " << numSelViews << ". Details: ";
        // gs->cameras->viewSelectionSubsetNumber = numSelViews;
        gs->cameras->selectedViewsNum[j] = numSelViews;

        for (int i = 0; i < numSelViews; i++)
        {
            // cout << camParams.viewSelectionSubset[i] << ", ";
            cout << camParams.selectedViews[j][i] << ",";
            // gs->cameras->viewSelectionSubset[i] = camParams.viewSelectionSubset[i];
            gs->cameras->selectedViews[j][i] = camParams.selectedViews[j][i];
        }
        cout << endl;
    }

    vector<InputData> inputData;

    for (int camIdx = 0; camIdx < gs->num_images; camIdx++)
    {
        InputData dat;
        // dat.id = id;
        dat.camId = camIdx;
        dat.cam = camParams.cameras[camIdx];
    
        dat.normals = normal_in_mem[camIdx];
        dat.depthMap = disp_in_mem[camIdx];
        inputData.push_back(dat);
    }


    cout << endl
         << "inputData.size(): " << inputData.size() << endl;
    // Init parameters
    gs->params = &algParameters;

    PointCloudList pc_list;
    pc_list.resize(gs->rows * gs->cols);
    pc_list.size = 0;
    pc_list.rows = gs->rows;
    pc_list.cols = gs->cols;
    gs->pc->rows = gs->rows;
    gs->pc->cols = gs->cols;

    cudaMemGetInfo(&avail, &total);
    used = total - avail;

    // vector<Mat> img_grayscale_float(img_grayscale.size());
    // vector<Mat> img_color_float(img_grayscale.size());
    // vector<Mat> img_color_float_alpha(img_grayscale.size());
    vector<Mat> normals_and_depth(gs->num_images);

    for (int i = 0; i < gs->num_images; i++)
    {
        /* Create vector of normals and disparities */
        vector<Mat_<float>> normal(3);
        normals_and_depth[i] = Mat::zeros(gs->rows, gs->cols, CV_32FC4);
        split(inputData[i].normals, normal);
        normal.push_back(inputData[i].depthMap);
        merge(normal, normals_and_depth[i]);
    }

    // gs->lines->free();

    cudaMemGetInfo(&avail, &total);
    used = total - avail;

    consistency_addImageToTextureFloatColor(normals_and_depth, gs->normals_depths, gs->cuArray); // Pass normal and depth to CUDA memory

    cout << "----------Run consistency------------------------------" << endl;

    // cudaMemGetInfo(&avail, &total);
    // used = total - avail;

    runconsistency(*gs, pc_list, gs->num_images);
    // cout << "----------Run consistency------------------------------" << endl;
    // runconsistency_cpu(*gs, pc_list, gs->num_images, normals_and_depth);

    // Mat_<Vec3f> norm0 = Mat::zeros ( img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC3 );
    Mat_<float> distImg;
    char plyFile[512];
    time_t now = time(0);
    char time_now[20] = {0};

    strftime(time_now, sizeof(time_now), "%Y%m%d-%H%M%S", localtime(&now)); //"%Y-%m-%d %H:%M:%S"

    // sprintf(plyFile, "%s_%s_cycle_%d_fuse_depT_%.4f_norT_%.2f_numC_%d.ply", output_folder,, gs->cycles, algParameters.depthThresh, algParameters.normalThresh * 180 / M_PI, algParameters.numConsistentThresh);
    
    // cout<< "plyFile name:"<< plyFile << endl;

    // sprintf(plyFile, "%s_0129_0.15_npvs_cycle_%d_fuse_depT_%.4f_norT_%.2f_numC_%d.ply", output_folder, gs->cycles, algParameters.depthThresh, algParameters.normalThresh * 180 / M_PI, algParameters.numConsistentThresh);

    // printf("Writing ply file %s\n", plyFile);
    // if ((gs->cycles == 0) || (gs->cycles == gs->params->cycles - 1))
    {
        // storePlyFileBinaryPointCloud(plyFile, pc_list, distImg);
    }
    
    // printf("finish ply file %s\n", plyFile);
    // for (int i = 0; i < gs->num_images; i++)
    // {
    //     saveConsImg(*gs, inputFiles.img_filenames[i], output_folder, i);
    // }

    return 0;
}





int ambc_main(InputFiles inputFiles,
              OutputFiles outputFiles,
              AlgorithmParameters &algParameters,
              vector<Mat> img_grayscale,
              GlobalState *gs)
{
    cout << endl
         << "---------Run ambc_main----------------------------------------" << endl;

    gs->depth_lower_bound = algParameters.depthMin;
    gs->depth_upper_bound = algParameters.depthMax;
    gs->params = &algParameters;
    cout << "gs->depth_lower_bound: " << gs->depth_lower_bound << endl;
    cout << "gs->depth_upper_bound: " << gs->depth_upper_bound << endl;
 
    // algParameters.num_img_processed = min(gs->num_images, algParameters.num_img_processed);
    // cout << "algParameters->num_img_processed: " << algParameters.num_img_processed << endl;

    CameraParameters cameraParams = getCameraParameters(*(gs->cameras), inputFiles, algParameters.cam_scale, true, gs->current_image);
    CameraParameters cameraParams_orig = getCameraParameters(*(gs->cameras_orig), inputFiles, algParameters.cam_scale, false, gs->current_image);
    cameraParams.idRef = gs->current_image; // change index
 
    cout << "cameraParams.idRef: " << cameraParams.idRef << "; gs->current_image: " << gs->current_image << endl;
    // writeParametersToFile ( resultsFile, inputFiles, algParameters, gtParameters, numPixels );

    if (gs->cycles == 0)
    {
        selectViews(cameraParams, gs->cols, gs->rows, algParameters, gs->current_image);

        int numSelViews = cameraParams.viewSelectionSubset.size();
        cout << "Total number of images used: " << numSelViews << endl;
        cout << "Selected views: ";
        for (int i = 0; i < numSelViews; i++)
        {
            // myfile << cameraParams.viewSelectionSubset[i] << ", ";
            cout << cameraParams.viewSelectionSubset[i] << ", ";
            // gs->cameras->viewSelectionSubset[i] = cameraParams.viewSelectionSubset[i];
            gs->cameras->selectedViews_ambc[gs->current_image][i] = cameraParams.viewSelectionSubset[i];
        }
        cout << endl;

        gs->params = &algParameters;

        gs->cameras->selectedViewsNum_ambc[gs->current_image] = static_cast<int>(numSelViews);
    }
    cout << "gs->cameras->selectedViewsNum_ambc[gs->current_image]:" << gs->cameras->selectedViewsNum_ambc[gs->current_image] << endl;

    runcuda(*gs);   // Core CUDA function
    

    char outputFolder[256];
    sprintf(outputFolder, "%s/depth_points/", outputFiles.parentFolder);
    mkdir(outputFolder, 0777);
    printf("%s\n", outputFolder);

    // change ref to gs.current_image
    // save depth_points
    // saveDepthImg(*gs, inputFiles.img_filenames[gs->current_image], std::string(outputFolder));
    // saveNumBestImg(*gs, inputFiles.img_filenames[gs->current_image], std::string(outputFolder));
    CameraParameters camParamsNotTransformed = getCameraParameters(*(gs->cameras), inputFiles, algParameters.cam_scale, false, gs->current_image);

   
    // Save_3D_Model(*gs, img_grayscale[gs->current_image], camParamsNotTransformed.cameras[gs->current_image], inputFiles, std::string(outputFolder));
    string ref_name = inputFiles.img_filenames[gs->current_image].substr(0, inputFiles.img_filenames[gs->current_image].length() - 4);
    time_t timeObj;
    time(&timeObj);
    tm *pTime = localtime(&timeObj);
    sprintf(outputFolder, "%s/%04d%02d%02d_%02d%02d%02d_%s", outputFiles.parentFolder, pTime->tm_year + 1900, pTime->tm_mon + 1, pTime->tm_mday, pTime->tm_hour, pTime->tm_min, pTime->tm_sec, ref_name.c_str());
    // mkdir(outputFolder, 0777);
    cout << "outputFolder: " << outputFolder << endl;

    Mat_<Vec3f> norm0 = Mat::zeros(gs->rows, gs->cols, CV_32FC3);
    Mat_<float> cudadisp = Mat::zeros(gs->rows, gs->cols, CV_32FC1);

    for (int i = 0; i < gs->cols; i++)
    {
        for (int j = 0; j < gs->rows; j++)
        {
            int center = i + gs->cols * j;
            float4 n = gs->lines->norm4[center];
            norm0(j, i) = Vec3f(n.x,
                                n.y,
                                n.z);
            cudadisp(j, i) = n.w;
        }
    }

    Mat_<Vec3f> norm0disp = norm0.clone();
    Mat planes_display, planescalib_display, planescalib_display2;
    getNormalsForDisplay(norm0disp, planes_display);
    Mat testImg_display;
    int sizeN = gs->cols / 8;
    float halfSize = (float)sizeN / 2.0f;
    Mat_<Vec3f> normalTestImg = Mat::zeros(sizeN, sizeN, CV_32FC3);
    for (int i = 0; i < sizeN; i++)
    {
        for (int j = 0; j < sizeN; j++)
        {
            float y = (float)i / halfSize - 1.0f;
            float x = (float)j / halfSize - 1.0f;
            float xy = pow(x, 2) + pow(y, 2);
            if (xy <= 1.0f)
            {
                float z = sqrt(1.0f - xy);
                normalTestImg(sizeN - 1 - i, sizeN - 1 - j) = Vec3f(-x, -y, -z);
            }
        }
    }

    normalTestImg.convertTo(testImg_display, CV_16U, 32767, 32767);
    cvtColor(testImg_display, testImg_display, COLOR_RGB2BGR);

    testImg_display.copyTo(planes_display(Rect(gs->cols - testImg_display.cols, 0, testImg_display.cols, testImg_display.rows)));
    // writeImageToFile ( "./", "normals", planes_display );

    char outputFolder_normal[256];
    sprintf(outputFolder_normal, "%s/normal/", outputFiles.parentFolder);
    mkdir(outputFolder_normal, 0777);
    sprintf(outputFolder_normal, "%s/normal/cycle%d_%04d%02d%02d_%02d%02d%02d_%s", outputFiles.parentFolder, gs->cycles, pTime->tm_year + 1900, pTime->tm_mon + 1, pTime->tm_mday, pTime->tm_hour, pTime->tm_min, pTime->tm_sec, ref_name.c_str());
    if ((gs->cycles == 0) || (gs->cycles == gs->params->cycles - 1))
    {
        //write normal image to file
        // writeImageToFile(outputFolder_normal, "normals", planes_display);
        // Mat_<Vec3f> plane0 = Mat::zeros(gs->rows, gs->cols, CV_32FC3);
        Mat_<Vec3f> plane0 = norm0disp.clone();
        for (int x = 0; x < gs->cols; x++)
        {   
            for (int y = 0; y < gs->rows; y++)
            {
                const int center = y * gs->cols + x;
                if (gs->lines_consistency[gs->current_image].used_pixels[center] == 0)
                {
                    plane0(y, x) = Vec3f(0,0,0);
                }
               
            }
            
        }
        getNormalsForDisplay(plane0, planescalib_display);
        // writeImageToFile(outputFolder_normal, "normals_with_cons", planescalib_display);
    }



    planes_display.release();

    Mat_<float> disp0 = cudadisp.clone();

    disp_in_mem.push_back(disp0);
    normal_in_mem.push_back(norm0);


    if ((gs->cycles == gs->params->cycles - 1))
    {
        mkdir(outputFolder, 0777);
        char outputPath[512];
        sprintf(outputPath, "%s/disp.dmb", outputFolder);
        cout << "outputPath" << outputPath << endl;
        writeDmb(outputPath, disp0);
        sprintf(outputPath, "%s/normals.dmb", outputFolder);
        cout << "outputPath" << outputPath << endl;
        writeDmbNormal(outputPath, norm0);
    }

    cout << endl
         << "---------finished Run ambc_main----------------------------------------" << endl;

    return 0;
}

int copy_foods_from_cuda(GlobalState *gs)
{

    size_t count_foods = gs->rows * gs->cols * gs->lines->FoodNumber * sizeof(float4);
    size_t count_fitness = gs->rows * gs->cols * gs->lines->FoodNumber * sizeof(float);
    cudaMemcpy(gs->foods_all[gs->current_image], gs->lines->foods_1d, count_foods, cudaMemcpyDefault);
    cudaMemcpy(gs->fitness_all[gs->current_image], gs->lines->fitness_1d, count_fitness, cudaMemcpyDefault);

    return 0;
}



int copy_foods_to_cuda(GlobalState *gs)
{

    size_t count_foods = gs->rows * gs->cols * gs->lines->FoodNumber * sizeof(float4);
    size_t count_fitness = gs->rows * gs->cols * gs->lines->FoodNumber * sizeof(float);
    cudaMemcpy(gs->lines->foods_1d, gs->foods_all[gs->current_image], count_foods, cudaMemcpyDefault);
    cudaMemcpy(gs->lines->fitness_1d, gs->fitness_all[gs->current_image], count_fitness, cudaMemcpyDefault);

    return 0;
}

void free_foods_all(GlobalState *gs)
{
    for (int i = 0; i < gs->num_images; i++)
    {
        free(gs->foods_all[i]);
        free(gs->fitness_all[i]);
    }
    free(gs->foods_all);
    free(gs->fitness_all);
}

int read_images(InputFiles inputFiles, vector<Mat_<Vec3b>> *img_color, vector<Mat_<uint8_t>> *img_grayscale, AlgorithmParameters *algParameters)
{
    int num_images = (*img_color).size();
    cout << "num_images:" << num_images << endl;
    for (int i = 0; i < num_images; i++)
    {
        // printf("path: %s\n", inputFiles.images_folder.c_str());
        cout << inputFiles.img_filenames[i] << endl;

        // Mat originalImage = imread((inputFiles.images_folder + inputFiles.img_filenames[i]), IMREAD_GRAYSCALE);
        // if (originalImage.rows == 0)
        // {
        //     printf("Image seems to be invalid\n");
        //     return -1;
        // }

        // Mat resizedImage;
        // resize(originalImage, resizedImage, Size(originalImage.cols / 2, originalImage.rows / 2));
        // (*img_grayscale)[i] = resizedImage;

        // if (algParameters->color_processing) // default: false
        // {
        //     Mat originalColorImage = imread((inputFiles.images_folder + inputFiles.img_filenames[i]), IMREAD_COLOR);
        //     Mat resizedColorImage;
        //     resize(originalColorImage, resizedColorImage, Size(originalColorImage.cols / 2, originalColorImage.rows / 2));
        //     (*img_color)[i] = resizedColorImage;
        // }

        (*img_grayscale)[i] = imread((inputFiles.images_folder + inputFiles.img_filenames[i]), IMREAD_GRAYSCALE);
        if (algParameters->color_processing) // default: false
        {
            (*img_color)[i] = imread((inputFiles.images_folder + inputFiles.img_filenames[i]), IMREAD_COLOR);
        }
        if ((*img_grayscale)[i].rows == 0)
        {
            printf("Image seems to be invalid\n");
            return -1;
        }
    }

    return 0;
}

int initialize_foods_all(GlobalState *gs)
{
    gs->foods_all = (float4 **)malloc(sizeof(float4 *) * gs->num_images);
    gs->fitness_all = (float **)malloc(sizeof(float *) * gs->num_images);
    for (int i = 0; i < gs->num_images; i++)
    {
        gs->foods_all[i] = (float4 *)malloc(sizeof(float4) * gs->rows * gs->cols * gs->lines->FoodNumber);
        gs->fitness_all[i] = (float *)malloc(sizeof(float) * gs->rows * gs->cols * gs->lines->FoodNumber);
        // for (int j = 0; j < gs->rows * gs->cols; j++)
        // {
        //     // gs->foods_all[i][j] = (float4 *)malloc(sizeof(float4) * gs->lines->FoodNumber);
        //     gs->fitness_all[i][j] = (float *)malloc(sizeof(float) * gs->lines->FoodNumber);
        // }
    }

    return 0;
}

int initialize_resize(GlobalState *gs)
{
    gs->lines->resize(gs->rows * gs->cols); // Resize lines  1200 * 1600   480 * 640
    gs->cameras = new CameraParameters_cu;
    gs->pc = new PointCloud;
    gs->resize(gs->num_images);
    gs->pc->resize(gs->rows * gs->cols);

    for (int i = 0; i < gs->num_images; i++)
    {
        gs->lines_consistency[i].n = gs->rows * gs->cols;
        gs->lines_consistency[i].resize(gs->lines_consistency[i].n);
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
        checkCudaErrors(cudaMallocArray(&gs->cuArray[i],
                                        &channelDesc,
                                        gs->cols,
                                        gs->rows));
    }

    checkCudaErrors(cudaMalloc(&gs->cs, gs->rows * gs->cols * sizeof(curandState)));

    return 0;
}

// Add all images to GPU memory
int initialize_imgs(GlobalState *gs,
                    vector<Mat_<Vec3b>> *img_color,
                    vector<Mat_<uint8_t>> *img_grayscale,
                    vector<Mat> *img_grayscale_float,
                    vector<Mat> *img_color_float,
                    vector<Mat> *img_color_float_alpha,
                    AlgorithmParameters *algParameters)
{
    for (int i = 0; i < gs->num_images; i++)
    {
        (*img_grayscale)[i].convertTo((*img_grayscale_float)[i], CV_32FC1); // or CV_32F works (too)
        if (algParameters->color_processing)
        {
            vector<Mat_<float>> rgbChannels(3);
            (*img_color_float_alpha)[i] = Mat::zeros(gs->rows, gs->cols, CV_32FC4);
            (*img_color)[i].convertTo((*img_color_float)[i], CV_32FC3); // or CV_32F works (too)
            Mat alpha(gs->rows, gs->cols, CV_32FC1);
            split((*img_color_float)[i], rgbChannels);
            rgbChannels.push_back(alpha);
            merge(rgbChannels, (*img_color_float_alpha)[i]);
        }
    }

    if (algParameters->color_processing)
    {
        addImageToTextureFloatColor(*img_color_float_alpha, gs->imgs);
    }
    else
    {
        addImageToTextureFloatGray(*img_grayscale_float, gs->imgs);
    }

    return 0;
}



int main(int argc, char **argv)
{
    clock_t start, end, start_all, end_all;
    auto start_chrono = std::chrono::system_clock::now();
    
    start_all = clock();
    InputFiles inputFiles;
    OutputFiles outputFiles;

    AlgorithmParameters *algParameters = new AlgorithmParameters;

    int ret;
    ret = getParametersFromCommandLine_all(argc, argv, inputFiles, outputFiles, *algParameters);
    if (ret != 0)
    {
        throw std::runtime_error("Error reading params!");
        return ret;
    }
    // vector<string> img_filenames_ori = inputFiles.img_filenames;

    if (inputFiles.img_filenames.empty())
        throw std::runtime_error("There was a problem finding the input files!");

    int num_images = inputFiles.img_filenames.size();
    inputFiles.num_images = num_images;
    cout << "-------initialize GlobalState -------------------------------" << endl;
    GlobalState *gs = new GlobalState;

    vector<Mat_<Vec3b>> img_color(num_images);
    vector<Mat_<uint8_t>> img_grayscale(num_images);

    read_images(inputFiles, &img_color, &img_grayscale, algParameters);

    cout << "num_images:" << num_images << endl;

    gs->rows = 466;//img_grayscale[0].rows; // 480
    gs->cols = 732;//img_grayscale[0].cols; // 640
    gs->num_images = num_images;
    printf("gs->rows %d\n", gs->rows); // 480
    printf("gs->cols %d\n", gs->cols); // 640

    cout << "-------initialize foods_all -------------------------------" << endl;
    cout << "gs->lines->FoodNumber:" << gs->lines->FoodNumber << endl;
    initialize_foods_all(gs);

    cout << "-------initialize resize -------------------------------" << endl;
    initialize_resize(gs);

    vector<Mat> img_grayscale_float(num_images);
    vector<Mat> img_color_float(num_images);
    vector<Mat> img_color_float_alpha(num_images);

    cout << "algParameters->color_processing: " << algParameters->color_processing << endl;
    initialize_imgs(gs,
                    &img_color,
                    &img_grayscale,
                    &img_grayscale_float,
                    &img_color_float,
                    &img_color_float_alpha,
                    algParameters);

    InputFiles consistency_inputFiles = inputFiles;
    

    for (int i = 0; i < gs->num_images; i++)
    {
        // gs->lines_consistency[i].n = gs->rows * gs->cols;
        gs->lines_consistency[i].reset(gs->lines_consistency[i].n);
    }

    for (int cycle = 0; cycle < algParameters->cycles; cycle++)
    {
        gs->cycles = cycle;
        cout << endl
             << "---------start round " << cycle << "-------------------------------" << endl
             << endl;

        for (int i = 0; i < num_images; i++)
        {

            gs->current_image = i;

            start = clock();
            copy_foods_to_cuda(gs);
            end = clock();
            cout << "copy_foods_to_cuda time  (s): " << (double)(end - start) / CLOCKS_PER_SEC << endl;

            cout << endl
                 << "---------for image: " << i << "-------------------------" << endl;

            ambc_main(inputFiles, outputFiles, *algParameters, img_grayscale_float, gs);

            start = clock();
            copy_foods_from_cuda(gs);
            end = clock();
            cout << "copy_foods_from_cuda time  (s): " << (double)(end - start) / CLOCKS_PER_SEC << endl;
        }

        // consistency_inputFiles.img_filenames.clear();
        for (int i = 0; i < gs->num_images; i++)
        {
            // gs->lines_consistency[i].n = gs->rows * gs->cols;
            gs->lines_consistency[i].reset(gs->lines_consistency[i].n);
        }

        // cu_Free_cuArray(gs->num_images, gs->cuArray);
        // gs->lines->free();
        consistency_main(consistency_inputFiles, *algParameters, gs);
        // cu_Free_cuArray(gs->num_images, gs->cuArray);
        // gs->lines->resize(gs->rows * gs->cols);


        // free_data(gs->num_images, gs);

        disp_in_mem.clear();
        normal_in_mem.clear();
    }

    free_foods_all(gs);
    end_all = clock();
    auto end_chrono = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_chrono - start_chrono);
    cout << "Total Runtime  (s): " << (double)(end_all - start_all) / CLOCKS_PER_SEC << endl;
    cout << "Total Runtime Chrono  (s): " << ((double) duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << endl;
    
    return 0;
}
