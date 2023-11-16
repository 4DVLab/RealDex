#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_broadcaster.h>

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include <Eigen/Eigen>
using json = nlohmann::json;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "static_publish_4");
    ros::NodeHandle nh;
    tf::Transform cam0tcam_cali[3];

    for(int i=1;i<4;i++)
    {
        std::string calibration_str = "/home/lab4dv/IntelligentHand/calibration_ws/calibration_process/data/cali0"+std::to_string(i)+".json";
        std::ifstream calibration_file(calibration_str);
        if (!calibration_file.is_open()) {
            std::cerr << "Error opening the file cali0"+std::to_string(i) << std::endl;
            return 1;
        }
        json calibration_json = json::parse(calibration_file);
        calibration_file.close();
        
        cam0tcam_cali[i-1].setOrigin(tf::Vector3(calibration_json["value0"]["translation"]["x"], calibration_json["value0"]["translation"]["y"], calibration_json["value0"]["translation"]["z"])); 
        cam0tcam_cali[i-1].setRotation(tf::Quaternion(calibration_json["value0"]["rotation"]["x"], calibration_json["value0"]["rotation"]["y"], calibration_json["value0"]["rotation"]["z"], calibration_json["value0"]["rotation"]["w"]));
    
    }
  
    tf::TransformListener listener;
    static tf::TransformBroadcaster br;
    tf::StampedTransform cam_transform[4];
    tf::Transform cam0tcam_transform[3];
    ros::Rate rate(10.0);

    while(nh.ok())
    {
        ros::spinOnce();

        try
        {
            for(int i=0;i<4;i++)
            {
                listener.lookupTransform("/cam"+std::to_string(i)+"_camera_base", "/cam"+std::to_string(i)+"_rgb_camera_link", ros::Time(0), cam_transform[i]);
                // ROS_INFO("%d,  %lf, %lf, %lf, %lf, %lf, %lf , %lf", i, cam_transform[i].getOrigin().getX() ,cam_transform[i].getOrigin().getY() , cam_transform[i].getOrigin().getZ(), cam_transform[i].getRotation().getW(), cam_transform[i].getRotation().getX(), cam_transform[i].getRotation().getY(), cam_transform[i].getRotation().getZ() );
            }     

        }   
        catch(tf::TransformException &ex)
        {
            ROS_ERROR("%s", ex.what());
        }

        for(int i=1; i<4; i++)
        {
            cam0tcam_transform[i-1] = cam_transform[0]*cam0tcam_cali[i-1]*cam_transform[i].inverse();
            br.sendTransform(tf::StampedTransform(cam0tcam_transform[i-1], ros::Time::now(), "/cam0_camera_base", "/cam"+std::to_string(i)+"_camera_base"));
            
            // ROS_INFO("cam0 to cam%d, %lf %lf %lf", i, cam0tcam_transform[i-1].getOrigin().getX(), cam0tcam_transform[i-1].getOrigin().getY(), cam0tcam_transform[i-1].getOrigin().getZ());

        }

        rate.sleep();
    }
    return 0;
}