## !/bin/bash

workdir_local="~/data/bags/"
workdir_ssd="~/data/ssd/"
worddir_sda="~/data/sda/"

# compress bagfile and remove the original bag

cd $workdir_local

traverse_dir()
{
    local filepath=$1 
    for file in `ls -a $filepath`
    do
        # echo ${filepath}/$file
        if [ -d ${filepath}/$file ]
        then
            if [[ $file != "." && $file != ".." ]]
            then
                #递归
                traverse_dir ${filepath}/$file
            fi
        else
            #调用查找指定后缀文件
            check_suffix ${filepath}/$file 
                
        fi
    done
}

#!/bin/bash

gnome-terminal -t "roscore" -x bash -c "roscore;exec bash;"
sleep 1s
gnome-terminal -t "ros_server" -x bash -c "roslaunch rosbridge_server rosbridge_websocket.launch;exec bash;"
sleep 1s
gnome-terminal -t "tf2" -x bash -c "rosrun tf2_web_republisher tf2_web_republisher;exec bash;"