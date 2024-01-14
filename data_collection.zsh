#!/bin/zsh
cd /home/lab4dv/IntelligentHand/calibration_ws
source /home/lab4dv/.zshrc
source /home/lab4dv/IntelligentHand/calibration_ws/devel/setup.zsh
rosnode kill /server_record
roslaunch multicam drivecam_withnodlet.launch