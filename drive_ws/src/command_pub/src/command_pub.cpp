#include <ros/ros.h>
#include <trajectory_msgs/JointTrajectory.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "command_pub");
    ros::NodeHandle n;

    ros::Publisher command_pub = n.advertise<trajectory_msgs::JointTrajectory>("/ra_trajectory_controller/command", 1000, true);
    
    ros::Rate loop_rate(100);

    // firstly publish only once
    while(ros::ok())
    {
        trajectory_msgs::JointTrajectory command;
      command.header.frame_id="";
      command.header.seq =1;
      command.header.stamp = ros::Time::now();
      command.joint_names = {"ra_shoulder_pan_joint", "ra_shoulder_lift_joint", "ra_elbow_joint", "ra_wrist_1_joint","ra_wrist_2_joint", "ra_wrist_3_joint" };
      std::vector<trajectory_msgs::JointTrajectoryPoint> points;
      trajectory_msgs::JointTrajectoryPoint point;
      point.positions ={
  -0.1728296583873927,
  -1.3142340456813264,
  1.7273566707683816,
  -0.4320490671130368,
  1.2689927138621389,
  -2.836351243023124

      };
      point.time_from_start.fromNSec(8000000);
      points.push_back(point);
      command.points = points;
      command_pub.publish(command);
      ROS_INFO("have published");
      ros::spinOnce();
      loop_rate.sleep();
    }
    
 

    return 0;

}