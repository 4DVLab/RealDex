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
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

#include<Eigen/Eigen>

#include <boost/thread/thread.hpp>
#include <fstream>
#include <iostream>
#include <string.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
int main(int argc, char** argv)
{
    if(argc != 2)
    {
        cerr<<"Error num of argument";
        return 1;
    }

    std::string str = argv[1]; 
    int p_begin = str.find_last_of("/") + 1;
    std::string name = str.substr(p_begin, str.find_first_of(".")-p_begin); 
    std::string dir = "/home/lab4dv/data/img_ply"+str.substr(0, p_begin-1).substr(str.substr(0, p_begin-1).find_last_of("/"))+"/";
 
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
    pcl::PLYWriter writer;

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
        std::system(("mkdir -p "+dir+name+"/cam"+std::to_string(i)).c_str());
        // std::system(("mkdir -p "+dir+name+"/cam"+std::to_string(i)+"/rgb").c_str());
        // std::system(("mkdir -p "+dir+name+"/cam"+std::to_string(i)+"/depth").c_str());
        std::system(("mkdir -p "+dir+name).c_str());
        
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

    Eigen::Affine3d t_transform;
    str = "/home/lab4dv/IntelligentHand/calibration_ws/k4a-calibration/input/pn00.json";
    std::ifstream t_transform_file(str);
    if(!t_transform_file.is_open())
    {
        std::cerr<<"Error opening the file pn00"<<std::endl;
        return 1;
    }
    json transform_json = json::parse(t_transform_file);
    t_transform_file.close();
    t_transform = Eigen::Affine3d::Identity();
    t_transform.rotate(Eigen::Quaterniond(transform_json["value0"]["rotation"]["w"], transform_json["value0"]["rotation"]["x"], transform_json["value0"]["rotation"]["y"], transform_json["value0"]["rotation"]["z"]).toRotationMatrix());
    t_transform.translation()<<transform_json["value0"]["translation"]["x"], transform_json["value0"]["translation"]["y"], transform_json["value0"]["translation"]["z"];


    pcl::CropBox<pcl::PointXYZRGB> crop;
    // crop.setMin(Eigen::Vector4f(-0.5, -2, -2, 1));
    // crop.setMax(Eigen::Vector4f(0.5, 0.6, 0.5, 1));
    crop.setMin(Eigen::Vector4f(-1, -1, -1, 1));
    crop.setMax(Eigen::Vector4f(1, 0.6, 0.5, 1));
    crop.setKeepOrganized(true);
    crop.setUserFilterValue(0.1f);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setMeanK (10);
    sor.setStddevMulThresh (1);

    // pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    // vg.setLeafSize(0.001f, 0.001f, 0.001f);

    pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
    outrem.setRadiusSearch(0.08);
    outrem.setMinNeighborsInRadius (10);
    outrem.setKeepOrganized(true);
    

    std::ofstream outfile;
    outfile.open("/home/lab4dv/data/bags/test/timestamp.txt");

    // while(it[0]!=view[0].end()&& it[1]!=view[1].end() &&it[2]!=view[2].end() &&it[3]!=view[3].end())
    while(it[0]!=view[0].end())
    {  
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr first_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_all(new pcl::PointCloud<pcl::PointXYZRGB>);


        pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
        icp.setMaxCorrespondenceDistance(1);
        icp.setMaximumIterations(30);
        icp.setTransformationEpsilon(1e-8);
        icp.setEuclideanFitnessEpsilon(1);
        ros::Time first_time;


        std::cout<<std::endl;
        outfile<<std::endl;

        ros::Time ttime[4];
        for(int i=0;i<1;i++)
        {
            auto m = *it[i];
            ros::Time  ttime = m.getTime();

            
            if(!i)
            {
                first_time = ttime;
                std::cout<<"first_time "<<first_time.toNSec()<<std::endl;
                outfile<<"first_time "<<first_time.toNSec()<<std::endl;
            }
                
            else
            {
                std::cout<<"time:"<<ttime.toNSec() <<" "<< int(ttime.toNSec() - first_time.toNSec())<<std::endl;
                outfile<<"time:"<<ttime.toNSec() <<" "<<int(ttime.toNSec() - first_time.toNSec())<<std::endl;
            }

            //timestamp  align

            

        }



        for(int i=0; i<4 ;i++)
        {
            if (i==1)
                continue;
            auto m = *it[i];
            std::string topic = m.getTopic();
            std::cout<<"topic:"<<topic<<std::endl;
            outfile<<"topic:"<<topic<<std::endl;
            
            
            sensor_msgs::PointCloud2::ConstPtr input = m.instantiate<sensor_msgs::PointCloud2>();
            if (input != NULL)
                {
                    pcl::fromROSMsg(*input,*cloud);

                    std::vector<int> mapping;
                    pcl::removeNaNFromPointCloud(*cloud, *cloud,mapping);

                    // vg.setInputCloud(cloud);
                    // vg.filter(*cloud);

                    // std::cout<<mapping.size()<<endl;
                    if(!i)
                    pcl::transformPointCloud(*cloud, *cloud, t_transform.inverse()); 
                    if(i)
                    {
                        pcl::transformPointCloud(*cloud, *cloud, t_transform.inverse()*transform[i-1]);
                    }
                    

                   

                    crop.setInputCloud(cloud);
                    crop.filter(*cloud);
                    
                    sor.setInputCloud(cloud);
                    // sor.setNegative(true);
                    sor.filter(*cloud);

                    outrem.setInputCloud(cloud);
                    outrem.filter(*first_cloud);

                    // viewer->removeAllPointClouds();
                    // viewer->removeAllShapes();
                    // viewer->setBackgroundColor(1, 1, 1);
                    // viewer->addPointCloud(cloud, "cloud_tmp");
                    // viewer->addArrow(pcl::PointXYZ(0, 0, 1), pcl::PointXYZ(0, 0, -1), 1, 0, 0, 1,  "x axis");
                    // viewer->addArrow(pcl::PointXYZ(0, 1, 0), pcl::PointXYZ(0, -1, 0), 1, 1, 0, 1,  "y axis");
                    // viewer->addArrow(pcl::PointXYZ(1, 0, 0), pcl::PointXYZ(-1, 0, 0), 1, 0, 1, 1,  "z axis");
                    // viewer->spin();


                    if(i)
                       { 

                        icp.setInputSource(cloud);
                        icp.setInputTarget(first_cloud);
                        icp.align(*cloud);


          
                
                       }
                    else{
                       (*first_cloud) = (*cloud);
           
                    }
                   
                    
                    pcl::transformPointCloud(*cloud, *cloud, t_transform);
                    
                    writer.write(dir+name+"/cam"+std::to_string(i)+"/"+std::to_string(first_time.toSec())+".ply", *cloud);
                    
                    (*cloud_all) = (*cloud_all) + (*cloud);
           
                
                    


                }
            else 
            {
                cout<<"fail pointclouds";
                break;
            }
            it[i]++;

          
        }
            
            

            writer.write(dir+name+"/"+std::to_string(first_time.toNSec())+".ply", *cloud_all);
           
            

    }
   
   outfile.close();


    return 0;
}