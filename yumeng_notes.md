启动能运行ROS GUI的docker
```
sudo docker run --net=host -e DISPLAY=host.docker.internal:0 --volume="$HOME/.Xauthority:/root/.Xauthority:rw" -it yumengliu/ros-noetic-ubuntu20 bash
```
