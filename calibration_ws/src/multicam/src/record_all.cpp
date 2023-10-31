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
#include <sensor_msgs/Image.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Eigen>

#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>


class CameraDataProcess
{
    private:
        std::string outputDir;
    public:
        void setOutputDir(std::string outputDir);
        std::string getOutputDir();
        void depthImageCallback(const sensor_msgs::ImageConstPtr& input);
        void rgbImageCallback(const sensor_msgs::ImageConstPtr& input);
        void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& input);

};

void CameraDataProcess::setOutputDir(std::string outputDir)
{
    this->outputDir = outputDir;
}
std::string CameraDataProcess::getOutputDir()
{
    return this->outputDir;
}
void CameraDataProcess::depthImageCallback(const sensor_msgs::ImageConstPtr& input)
{
    double image_time = input->header.stamp.toNSec();
    std::string frame =  input->header.frame_id;
    if()
   ROS_INFO("send msg = %d", image_time);

    cv_bridge::CvImageConstPtr ptr;

    ptr = cv_bridge::toCvCopy(input, "bgr8");
    cv::Mat show_img = ptr->image;
    std::string image_name = std::to_string(image_time) + ".png";              
    cv::imwrite(this->getOutputDir() +  image_name, show_img);

}
void CameraDataProcess::rgbImageCallback(const sensor_msgs::ImageConstPtr& input);
void CameraDataProcess::pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& input);


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
    ros::init(argc, argv, "record_all");
    ros::NodeHandle nh("~");

    nh.getParam("output", output);

    std::cout<< "record all file into "<<output;

    ros::Subscriber depthToRGBSub[4], rgbSub[4], tf, pointcloudSub[4];
    for(int i=0; i<4; i++)
    {
        ros::Subscriber sub = nh.subscribe("/cam"+std::to_string(i)+"/de", 10, call_back);
    }
    

    ros::spin();

    return 0;
}
