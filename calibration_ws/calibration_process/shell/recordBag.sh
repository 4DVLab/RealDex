## !/bin/bash

duration=$1

output=$2

echo $duration 
echo $output 

rosbag record  /cam0/depth/camera_info /cam0/depth/image_raw /cam0/joint_states /cam0/rgb/camera_info /cam0/rgb/image_raw  /cam0/depth_to_rgb/camera_info /cam0/depth_to_rgb/image_raw /cam1/depth/camera_info /cam1/depth/image_raw  /cam1/joint_states /cam1/rgb/camera_info /cam1/rgb/image_raw /cam1/depth_to_rgb/camera_info /cam1/depth_to_rgb/image_raw  /cam2/depth/camera_info /cam2/depth/image_raw /cam2/joint_states /cam2/rgb/camera_info /cam2/rgb/image_raw /cam2/depth_to_rgb/camera_info /cam2/depth_to_rgb/image_raw  /cam3/depth/camera_info /cam3/depth/image_raw /cam3/joint_states /cam3/rgb/camera_info /cam3/rgb/image_raw /cam3/depth_to_rgb/camera_info /cam3/depth_to_rgb/image_raw /cam0/points2 /cam1/points2 /cam2/points2 /cam3/points2 -O $output --duration $duration  -b 3072 &

ssh user@server  "rosbag record /tf /tf_static /tf_vive -O $output --duration $duration  -b 3072"

ssh user@server "ls -l $output"

