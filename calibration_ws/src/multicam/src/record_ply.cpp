#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_broadcaster.h>

#include <jsoncpp/json/json.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Eigen>


std::string output;

void call_back(const sensor_msgs::PointCloud2ConstPtr& input)
{
    
//Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg (*input, *cloud);//cloud is the output
   std::vector<int> mapping;
    pcl::removeNaNFromPointCloud(*cloud, *cloud,mapping);
    pcl::PLYWriter writer;
    if(! output.empty())
    writer.write(output, *cloud);
    
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "record_ply");
    ros::NodeHandle nh("~");
     std::string cloud;

    ROS_INFO("lallala");
    nh.getParam("cloud", cloud);
    nh.getParam("output", output);

    std::cout<< cloud <<" "<<output;

    ros::Subscriber sub = nh.subscribe(cloud, 10, call_back);

    ros::spin();

    return 0;
}
