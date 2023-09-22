#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
 
#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
 
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <sensor_msgs/CompressedImage.h>

int main(int argc, char** argv)
{
      // Initialize ROS
    ros::init (argc, argv, "pointcloud_merge");
    ros::NodeHandle nh;
 
    rosbag::Bag bag;
    bag.open("/home/lab4dv/data/bags/yangtao/yangtao_0_20230905.bag", rosbag::bagmode::Read);
    
    std::vector<std::string> topics; 
    topics.push_back(std::string(" /cam0/points2"));    
    topics.push_back(std::string(" /cam1/points2"));   
    topics.push_back(std::string(" /cam2/points2"));   
    topics.push_back(std::string(" /cam3/points2"));        
            
  
    rosbag::View view(bag, rosbag::TopicQuery(topics));
    pcl::visualization::CloudViewer viewer("test_view");

     int num;
    while(true){
        rosbag::View::iterator it = view.begin(); 
        std::cout<<"please input num:";
        std::cin>>num;
        std::cout<<"num:"<<num<<std::endl;

        std::advance(it,num);
        for(; it !=  view.end(); ++it)
        {  
            auto m = *it;
            std::string topic   = m.getTopic();
            std::cout<<"topic:"<<topic<<std::endl;
            std::string callerid       = m.getCallerId();
            std::cout<<"callerid:"<<callerid<<std::endl;
            ros::Time  time = m.getTime();
            std::cout<<"time:"<<time.sec<<":"<<time.nsec<<std::endl;


            sensor_msgs::PointCloud2::ConstPtr input = m.instantiate<sensor_msgs::PointCloud2>();
        //sensor_msgs::PointCloud2ConstPtr input = *s ;
        if (input != NULL)
        {
            // 创建一个输出的数据格式
            sensor_msgs::PointCloud2 input;  //ROS中点云的数据格式
            //对数据进行处理
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
 
            pcl::fromROSMsg(input,*cloud);
     
            //blocks until the cloud is actually rendered
            viewer.showCloud(cloud);
    
            //pub.publish (input);
        }

        }



    }

    return 0;
}