

you can use `ctrl` +`Alt`+`T` to active a new terminator and split windows in the new terminator

# in case 
1. memory cache issue  

```
sudo sh -c 'echo 1 >  /proc/sys/vm/drop_caches'
```

2. cameras issue
   ```
   k4aviewer
   ```
   to check all cameras are listed ,if some of them miss, re-plug it

3. network issue

check whether WiFi is off and wired network is on

4. any other issue
call yaxun by +86 18307331878

# Before 
1. step into calibration_ws and `source devel/setup.zsh`
2. export
   ```
  export ROS_IP=127.0.0.1 
  export ROS_MASTER_URI=http://localhost:11311/

   ```
3. drive 4 cameras
   ```
   roslaunch multicam drive4cam.launch
   ```
4. exit and `soruce ~/.zshrc` then `source devel/setup.zsh`
5. wait for showhand ready
6. recheck by rviz the comment the rviz line in `/home/lab4dv/IntelligentHand/calibration_ws/src/multicam/launch/driveCamera/drive4cam.launch`
7. reroslaunch by ` roslaunch multicam drive4cam.launch`
8. begin data collection

# Data collection
1. cd into `~/data`
2. mount ssd and sda
```
sudo mount /dev/nvme1n1 ssd
sudo mount /dev/sda sda 
```
3. make sure only collect data into `ssd` or `bags`, maybe fistly `ssd` and then `sda`
4. collect different categories of data into different directory and record the number of sequences
```
mkdir example
cd example
vim example.md
```
5. calculate all number and write into `data_collection.md`


# Process
1. compress all data and make sure all cores except 4 are in use
`rosbag compress example_0_20230930.bag`,  and while compressing pleas make there memory for using
2. after making sure the compression is done(the compressed bag is around 13 G), delete `example.orig.bag`
2.  ssh to remote-server to make sure remote-server is reachable , and do any other operator which may not be done in sftp terminator
```
 ssh yangyaxun@10.15.89.174 -p 22112
``` 
password: `4DVlab123!`
3. sftp to remote server, the password is same as the above
```
sftp -P 22112 yangyaxun@10.15.89.174 
``` 
password: `4DVlab123!`
4. any command with `l` as began is used for local, and without `l` as begin is used for remote
5. use `cd /sharedata/home/shared/intelligent_hand/bags` to remote directory
6. us `put example.bag` to upload local bag to remote server
7. make sure remote bag is save correctly then delete local file. 
