## !/bin/zsh

k4arecorder --list

echo "please check whether the index of cameras correct.\n"

k4arecorder -d WFOV_UNBINNED  --device 3 --external-sync subordinate --imu OFF -c 1080p -r 5 -l 2 output-4.mkv&

k4arecorder -d WFOV_UNBINNED  --device 0 --external-sync subordinate --imu OFF -c 1080p -r 5 -l 2 output-3.mkv&

k4arecorder -d WFOV_UNBINNED  --device 2 --external-sync subordinate --imu OFF -c 1080p -r 5 -l 2 output-2.mkv&

k4arecorder -d WFOV_UNBINNED  --device 1 --external-sync master --imu OFF -c 1080p -r 5 -l 2 output-1.mkv

cd /home/lab4dv/IntelligentHand/calibration_ws/calibration_process/shell
mv ./output* ../../k4a-calibration/input/