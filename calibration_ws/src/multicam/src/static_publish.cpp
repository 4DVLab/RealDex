#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_broadcaster.h>

#include <jsoncpp/json/json.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include <Eigen/Eigen>
using json = nlohmann::json;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "static_publish");
    ros::NodeHandle nh;
    std::string calibration_1 = "/home/lab4dv/intelligentHand/IntelligentHand/calibration_ws/k4a-calibration/output/cali01.json";

     
    std::ifstream calibration_1_file(calibration_1);
    if (!calibration_1_file.is_open()) {
        std::cerr << "Error opening the file." << std::endl;
        return 1;
    }
    json calibration_1_json = json::parse(calibration_1_file);
    calibration_1_file.close();
    tf::Transform cam0tcam1_cali;
    cam0tcam1_cali.setOrigin(tf::Vector3(calibration_1_json["value0"]["translation"]["x"], calibration_1_json["value0"]["translation"]["y"], calibration_1_json["value0"]["translation"]["z"])); 
    cam0tcam1_cali.setRotation(tf::Quaternion(calibration_1_json["value0"]["rotation"]["x"], calibration_1_json["value0"]["rotation"]["y"], calibration_1_json["value0"]["rotation"]["z"], calibration_1_json["value0"]["rotation"]["w"]));
    
    tf::TransformListener listener;
    static tf::TransformBroadcaster br;
    tf::StampedTransform cam0_transform;
    tf::StampedTransform cam1_transform;
    tf::StampedTransform cam2_transform;
    tf::StampedTransform cam3_transform;
    tf::Transform cam0tcam1_transform;
    tf::Transform cam0tcam2_transform;
    tf::Transform cam0tcam3_transform;
    ros::Rate rate(10.0);

    while(nh.ok())
    {

        try
        {
            listener.lookupTransform("/cam0_camera_base", "/cam0_rgb_camera_link", ros::Time(0), cam0_transform);
           // listener.lookupTransform("/cam1_camera_link", "/cam1_camera_color_optical_frame", ros::Time(0), cam1_transform);
            listener.lookupTransform("/cam1_camera_base", "/cam1_rgb_camera_link", ros::Time(0), cam1_transform);
            //listener.lookupTransform("/cam3_camera_link", "/cam3_camera_color_optical_frame", ros::Time(0), cam3_transform);
        }
        catch(tf::TransformException &ex)
        {
            ROS_ERROR("%s", ex.what());
        }
        // std::cout<<"cam0_transform: "<<cam0_transform.getOrigin().x()<<" "<<cam0_transform.getOrigin().y()<<" "<<cam0_transform.getOrigin().z() << 
        // cam0_transform.getRotation().x()<< cam0_transform.getRotation().y()<< cam0_transform.getRotation().z() << cam0_transform.getRotation().w() <<std::endl;
        // cam0tcam1_transform = *cam0tcam1_cali* ;
        cam0tcam1_transform = cam0_transform*cam0tcam1_cali*cam1_transform.inverse();
        br.sendTransform(tf::StampedTransform(cam0tcam1_transform, ros::Time::now(), "/cam0_camera_base", "/cam1_camera_base"));


        rate.sleep();
    }
    return 0;
}