## !/bin/zsh

./record4mkv.sh
 sleep 20

 cd ../../k4a-calibration/build
 make -j$(nproc)
 ./calib_k4a ../input/output-1.mkv ../input/output-2.mkv ../input/output-3.mkv ../input/output-4.mkv
cd -
 sleep 20

 cd ../build
./k4acalibration_process
 sleep 2

# cd ../..
# source devel/setup.zsh
# roslaunch multicam drive1cam_withnodlet.launch