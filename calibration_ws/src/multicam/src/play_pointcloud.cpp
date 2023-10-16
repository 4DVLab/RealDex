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

#include <Eigen/Eigen>
using json = nlohmann::json;


int main(int argc, char **argv)
{
     ros::init(argc, argv, "play_pointcloud");
    ros::NodeHandle nh("~");
    std::string dir;
    nh.getParam("dir", dir);

    ros::Publisher publisher = nh.advertise<sensor_msgs::PointCloud2>("m_cloud", 1000);

    boost::filesystem::path path(dir);
    if(! boost::filesystem::exists(path) || ! boost::filesystem::is_directory(path))
    {
        std::cout<<"input  "<<path.string()<<" not exist or not a directory."<< std::endl;
        return 1;
    }


    pcl::PLYReader reader;
    boost::filesystem::recursive_directory_iterator end_iter;
    ros::Rate loop_rate(5);
    for(boost::filesystem::recursive_directory_iterator iter(path); iter != end_iter; iter++)
    {
        if(boost::filesystem::is_regular_file(*iter) )
        {
            std::string curSuffix = iter->path().string().substr(iter->path().string().size() - std::string(".ply").size());
 
            if (".ply" == curSuffix)
            {
                std::cout<< iter->path().string()<<std::endl;

                pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud;
                reader.read(iter->path().string(), *pointcloud);

                sensor_msgs::PointCloud2::Ptr rosPointcloud;
                
                pcl::toROSMsg(*pointcloud, *rosPointcloud);

                
                publisher.publish(rosPointcloud);
                ros::spinOnce();

                loop_rate.sleep();
            }


        }
    }


    
    return 0;
}