启动能运行ROS GUI的docker
```
#!/bin/bash

# allow access from localhost
xhost + 127.0.0.1

sudo docker run --net=host -e DISPLAY=host.docker.internal:0 --volume="$HOME/.Xauthority:/root/.Xauthority:rw" -it yumengliu/ros-noetic-ubuntu20 bash
```
