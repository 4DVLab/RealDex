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
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/crop_box.h>

#include<Eigen/Eigen>

#include <boost/thread/thread.hpp>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
int main(int argc, char** argv)
{

    std::string str="/home/lab4dv/data/bags/meal_spoon/meal_spoon_0_20230921.bag";
    std::string name="meal_spoon_0_20230921";
    std::string dir="/home/lab4dv/data/img_pcd/meal_spoon/";

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
    pcl::PCDWriter writer;

    rosbag::Bag bag;
    bag.open(str, rosbag::bagmode::Read);
    
    std::vector<std::string> pcd_topic[4], rgb_topic[4], depth_topic[4];
    rosbag::View view[4];
    rosbag::View::iterator it[4];
    for(int i=0;i<4;i++)
    {
        pcd_topic[i].push_back(std::string("/cam"+std::to_string(i)+"/points2")); 
        view[i].addQuery(bag, rosbag::TopicQuery(pcd_topic[i]));
        it[i]= view[i].begin(); 
        std::system(("mkdir -p "+dir+name+"/cam"+std::to_string(i)+"/pcd").c_str());
        std::system(("mkdir -p "+dir+name+"/cam"+std::to_string(i)+"/rgb").c_str());
        std::system(("mkdir -p "+dir+name+"/cam"+std::to_string(i)+"/depth").c_str());
        std::system(("mkdir -p "+dir+name+"/pcd/").c_str());
        
    } 

    
    //read tranformation
    Eigen::Affine3d transform[3];
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
        transform[i-1]= Eigen::Affine3d::Identity();
        transform[i-1].rotate(Eigen::Quaterniond(calibration_json["value0"]["rotation"]["w"],calibration_json["value0"]["rotation"]["x"], calibration_json["value0"]["rotation"]["y"], calibration_json["value0"]["rotation"]["z"]).toRotationMatrix());
        transform[i-1].translation()<<calibration_json["value0"]["translation"]["x"], calibration_json["value0"]["translation"]["y"], calibration_json["value0"]["translation"]["z"]; 
     
    }

    //read transformation of table

    Eigen::Affine3d t_transform[4];
    for(int i=0; i<4; i++)
    {
        std::string str = "/home/lab4dv/IntelligentHand/calibration_ws/k4a-calibration/input/pn0"+std::to_string(i)+".json";
        std::ifstream t_transform_file(str);
        if(!t_transform_file.is_open())
        {
            std::cerr<<"Error openning the file cn0"+std::to_string(i) <<std::endl;
            return 1;
        }
        json transform_json = json::parse(t_transform_file);
        t_transform_file.close();
        t_transform[i] = Eigen::Affine3d::Identity();
        t_transform[i].rotate(Eigen::Quaterniond(transform_json["value0"]["rotation"]["w"], transform_json["value0"]["rotation"]["x"], transform_json["value0"]["rotation"]["y"], transform_json["value0"]["rotation"]["z"]).toRotationMatrix());
        t_transform[i].translation()<<transform_json["value0"]["translation"]["x"], transform_json["value0"]["translation"]["y"], transform_json["value0"]["translation"]["z"];
    }

       
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::CropBox<pcl::PointXYZRGB> crop;
    crop.setMin(Eigen::Vector4f(-0.5, -2, -2, 1));
    crop.setMax(Eigen::Vector4f(0.5, 0.6, 0.05, 1));
    
    while(it[0]!=view[0].end()&& it[1]!=view[1].end() &&it[2]!=view[2].end() &&it[3]!=view[3].end())
    {  
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr first_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_all(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
        icp.setMaxCorrespondenceDistance(1);
        icp.setMaximumIterations(30);
        icp.setTransformationEpsilon(1e-8);
        icp.setEuclideanFitnessEpsilon(1);
        // pcl::NormalDistributionsTransform<pcl::PointXYZRGB, pcl::PointXYZRGB> ndt;
        // ndt.setResolution(0.01);
        ros::Time first_time;

        for(int i=0; i<4 ;i++)
        {
            auto m = *it[i];
            std::string topic   = m.getTopic();
            std::cout<<"topic:"<<topic<<std::endl;
            ros::Time  ttime = m.getTime();
            std::cout<<"time:"<<ttime.sec<<":"<<ttime.nsec<<std::endl;


            if(!i)
                first_time = ttime;
            sensor_msgs::PointCloud2::ConstPtr input = m.instantiate<sensor_msgs::PointCloud2>();
            if (input != NULL)
                {
                    pcl::fromROSMsg(*input,*cloud);
                    std::vector<int> mapping;
                    pcl::removeNaNFromPointCloud(*cloud, *cloud,mapping);
                    // std::cout<<mapping.size()<<endl;
                    pcl::transformPointCloud(*cloud, *cloud, t_transform[i].inverse());

                    

                    crop.setInputCloud(cloud);
                    crop.setKeepOrganized(true);
                    crop.setUserFilterValue(0.1f);
                    crop.setKeepOrganized(true);
                    // crop.setRotation
                    crop.filter(*cloud);

                    // pcl::transformPointCloud(*cloud, *cloud, t_transform[i]);

                    if(i)
                       { 
                        
                        //no need any more 
                        // pcl::transformPointCloud(*cloud, *cloud, transform[i-1]);

                        icp.setInputSource(cloud);
                        icp.setInputTarget(first_cloud);
                        icp.align(*cloud);

                         
                        // ndt.setInputSource(cloud);
                        // ndt.setInputTarget(first_cloud);
                        // ndt.align(*cloud);

                        // viewer->removeAllPointClouds();
                        // viewer->removeAllShapes();
                        // //  viewer->setBackgroundColor(1, 1, 1);
                        // viewer->addPointCloud(cloud, "cloud_tmp");
                        // viewer->addArrow(pcl::PointXYZ(0, 0, 1), pcl::PointXYZ(0, 0, -1), 1, 0, 0, 1,  "x axis");
                        // viewer->addArrow(pcl::PointXYZ(0, 1, 0), pcl::PointXYZ(0, -1, 0), 1, 1, 0, 1,  "y axis");
                        // viewer->addArrow(pcl::PointXYZ(1, 0, 0), pcl::PointXYZ(-1, 0, 0), 1, 0, 1, 1,  "z axis");
                        // viewer->spin();

                       }
                    else{
                       (*first_cloud) = (*cloud);
                    //    first_cloud = cloud;
                    }
                 //   writer.write(dir+name+"/cam"+std::to_string(i)+"/pcd/"+std::to_string(ttime.toSec())+".pcd", *cloud);
                   
                    (*cloud_all) = (*cloud_all) + (*cloud);

                    viewer->removeAllPointClouds();
                    viewer->removeAllShapes();
                    //  viewer->setBackgroundColor(1, 1, 1);
                     viewer->addPointCloud(cloud_all, "cloud_tmp");
                     viewer->addArrow(pcl::PointXYZ(0, 0, 1), pcl::PointXYZ(0, 0, -1), 1, 0, 0, 1,  "x axis");
                     viewer->addArrow(pcl::PointXYZ(0, 1, 0), pcl::PointXYZ(0, -1, 0), 1, 1, 0, 1,  "y axis");
                     viewer->addArrow(pcl::PointXYZ(1, 0, 0), pcl::PointXYZ(-1, 0, 0), 1, 0, 1, 1,  "z axis");
                     viewer->spin();
                 
                    
                }
            else 
            {
                cout<<"fail pointclouds";
                break;
            }
            it[i]++;

          
        }
            
            // viewer->removeAllPointClouds();
            // viewer->addPointCloud(cloud_all, "cloud_all");
            // viewer->spin();
            // writer.write(dir+name+"/pcd/"+std::to_string(first_time.toSec())+".pcd", *cloud_all);
            break;

    }
   


    return 0;
}