#!/bin/bash

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash" --
# Due to a bug in XmlRpcServer, setting these limits is necessary on arch to avoid ROS from allocating all RAM and
# running OOM (https://answers.ros.org/question/336963/rosout-high-memory-usage/)
ulimit -Sn 524288 && ulimit -Hn 524288
exec "$@"
