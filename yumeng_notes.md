启动能运行ROS GUI的docker  
参考 [here](https://gist.github.com/cschiewek/246a244ba23da8b9f0e7b11a68bf3285)
```
#!/bin/bash

# allow access from localhost
xhost + 127.0.0.1

sudo docker run --net=host -e DISPLAY=host.docker.internal:0 --volume="$HOME/.Xauthority:/root/.Xauthority:rw" -it yumengliu/ros-noetic-ubuntu20 bash
```

Meet libGL error:  failed to load driver: swrast
[Solution](https://psycomp.utsc.utoronto.ca/support/index.php/2021/10/14/libgl-error-when-using-ssh-x-forwarding-with-macos/)  
```
export LIBGL_ALWAYS_INDIRECT=1
```
