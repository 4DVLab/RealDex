启动能运行ROS GUI的docker
```
sudo docker run --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" -it yumengliu/ros-noetic-ubuntu20 bash
```
