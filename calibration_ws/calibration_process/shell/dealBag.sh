#!/bin/bash



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
 
 

check_suffix()
{
    local file=$1
    # echo "${file#*.}"
    if [ "${file#*.}"x = "bag.active"x ] ;then
        file_pre="${file%%.*}"
        echo $file_pre
        rosbag reindex $file_pre.bag.active
        rosbag fix --force $file_pre.bag.active $file_pre.bag
        rm $file_pre.bag.*
    fi    
}
 

 traverse_dir ~/data




