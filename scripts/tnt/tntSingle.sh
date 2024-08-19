#!
# prog="../fuse/build/MVS"
prog="./build/MVS"
warping="./fusibileOurs/build/fusibile"
inputdir="./data/tnt_new/${2}/images/"
batch_name="tnt"
# output_dir_basename="results/$batch_name/$(date +"%Y%m%d-%H%M%S")"
output_dir_basename="results/$batch_name/${1}"
in_ex_folder="./data/tnt_new/${2}/cams_1/"

scale=1
blocksize=15
warm_up_iters=20 # 20
iter=2 # 5
cost_gamma=10
cost_comb="best_n"
n_best=2 
depth_max=5 # 800
depth_min=0.01 #0.01 # 300
image_list_array=`( cd "$inputdir" && ls) `
output_dir=${output_dir_basename}/$i/
echo $output_dir

mkdir -p $output_dir  
cost_choice="bncc"    # weighted_difference/bncc
max_views=9
color_processing=0 # 1:color, 0: gray
num_best=5
cycles=3 # 6

min_angle=15
max_angle=30
min_angle_degree=10 # used for algParameters.min_angle_degree
max_angle_degree=30


number_consistent_input=4 #2

for ij in $image_list_array
do
    image_list+=( $ij )
done



cmd2="-in_ex_folder $in_ex_folder
-output_folder $output_dir 
-no_display 
--cam_scale=$scale 
--iterations=$iter 
--blocksize=$blocksize 
--cost_gamma=$cost_gamma 
--cost_comb=best_n 
--n_best=$n_best 
--depth_max=$depth_max 
--depth_min=$depth_min 
-view_selection 
--min_angle=$min_angle 
--max_angle=$max_angle 
--max_views=$max_views 
--color_processing=$color_processing 
--cost_choice=$cost_choice 
-input_folder $output_dir 
--disp_thresh=$disp_thresh 
--normal_thresh=$normal_thresh 
--num_consistent=$num_consistent
--min_angle_degree=$min_angle_degree 
--max_angle_degree=$max_angle_degree
-remove_black_background
--number_consistent_input=$number_consistent_input
--warm_up_iters=$warm_up_iters
--num_best=$num_best
--cycles=$cycles"

# $prog ${image_list[@]} -images_folder "$inputdir" -p_folder $p_folder -output_folder $output_dir -no_display --cam_scale=$scale --iterations=$iter --blocksize=$blocksize --cost_gamma=$cost_gamma --cost_comb=best_n --n_best=$n_best --depth_max=$depth_max --depth_min=$depth_min -view_selection --min_angle=$min_angle --max_angle=$max_angle --max_views=$max_views --color_processing=$color_processing  --cost_choice=$cost_choice -input_folder $output_dir --disp_thresh=$disp_thresh --normal_thresh=$normal_thresh --num_consistent=$num_consistent --min_angle_degree=$min_angle_degree --max_angle_degree=$max_angle_degree -remove_black_background --number_consistent_input=$number_consistent_input --warm_up_iters=$warm_up_iters --num_best=$num_best
echo $prog ${image_list[@]} -images_folder "$inputdir" $cmd2
$prog ${image_list[@]} -images_folder "$inputdir" $cmd2


# fuse options
disp_thresh=0.5 #0.5 larger means more vertics
normal_thresh=30 # 30
num_consistent=3 # 5
min_angle=5
max_angle=70

echo $warping -input_folder $output_dir -in_ex_folder $in_ex_folder -krt_file $p_folder -images_folder $inputdir --cam_scale=$scale --depth_min=$depth_min --depth_max=$depth_max --disp_thresh=$disp_thresh --normal_thresh=$normal_thresh --num_consistent=$num_consistent
# $warping -input_folder $output_dir -in_ex_folder $in_ex_folder -krt_file $p_folder -images_folder $inputdir --cam_scale=$scale --depth_min=$depth_min --depth_max=$depth_max --disp_thresh=$disp_thresh --normal_thresh=$normal_thresh --num_consistent=$num_consistent -remove_black_background
# without p folder
$warping -input_folder $output_dir -in_ex_folder $in_ex_folder -images_folder $inputdir --cam_scale=$scale --depth_min=$depth_min --depth_max=$depth_max --disp_thresh=$disp_thresh --normal_thresh=$normal_thresh --num_consistent=$num_consistent -remove_black_background


