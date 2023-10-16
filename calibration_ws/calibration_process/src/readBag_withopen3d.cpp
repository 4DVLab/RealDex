

// #include <open3d/Open3D.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/pipelines/registration/Registration.h>
#include <open3d/pipelines/registration/ColoredICP.h>
#include <open3d/utility/Console.h>
#include <open3d/io/PointCloudIO.h>
#include <open3d/visualization/visualizer/Visualizer.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
 
#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>



#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>

#include<Eigen/Eigen>

#include <boost/thread/thread.hpp>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;




int main(int argc, char** argv)
{

    std::string str="/home/lab4dv/data/bags/test/test_1_20231009.bag";
    std::string name="test_1_20231009";
    std::string dir="/home/lab4dv/data/img_pcd/test/";

    rosbag::Bag bag;
    bag.open(str, rosbag::bagmode::Read);
    
    std::vector<std::string> pc_topic[4], rgb_topic[4], depth_topic[4];
    rosbag::View view[4];
    rosbag::View::iterator it[4];
    for(int i=0;i<4;i++)
    {
        pc_topic[i].push_back(std::string("/cam"+std::to_string(i)+"/points2")); 
        view[i].addQuery(bag, rosbag::TopicQuery(pc_topic[i]));
        it[i]= view[i].begin(); 
        std::system(("mkdir -p "+dir+name+"/cam"+std::to_string(i)+"/pcd").c_str());
        std::system(("mkdir -p "+dir+name+"/cam"+std::to_string(i)+"/rgb").c_str());
        std::system(("mkdir -p "+dir+name+"/cam"+std::to_string(i)+"/depth").c_str());
        std::system(("mkdir -p "+dir+name+"/ply/").c_str());
        
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

    
    
    
    pcl::CropBox<pcl::PointXYZRGB> crop;
    crop.setMin(Eigen::Vector4f(-0.5, -2, -2, 1));
    crop.setMax(Eigen::Vector4f(0.5, 0.6, 0.05, 1));

    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setMeanK (10);
    sor.setStddevMulThresh (0.9);

    pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
    outrem.setRadiusSearch(0.08);
    outrem.setMinNeighborsInRadius (10);
    outrem.setKeepOrganized(true);

    open3d::visualization::Visualizer viewer;
    viewer.CreateVisualizerWindow("point_cloud", 1280, 960);
    
    while(it[0]!=view[0].end()&& it[1]!=view[1].end() &&it[2]!=view[2].end() &&it[3]!=view[3].end())
    {  
        std::shared_ptr<open3d::geometry::PointCloud> cloud(new open3d::geometry::PointCloud);
        std::shared_ptr<open3d::geometry::PointCloud> cloud_all(new open3d::geometry::PointCloud);
        std::shared_ptr<open3d::geometry::PointCloud> d_first_cloud(new open3d::geometry::PointCloud);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      
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
                    pcl::fromROSMsg(*input,*tmp_cloud);

                    std::vector<int> mapping;
                    pcl::removeNaNFromPointCloud(*tmp_cloud, *tmp_cloud,mapping);
                    // std::cout<<mapping.size()<<endl;
                    if(!i)
                    pcl::transformPointCloud(*tmp_cloud, *tmp_cloud, t_transform[0].inverse()); 
                    if(i)
                    {
                        pcl::transformPointCloud(*tmp_cloud, *tmp_cloud, t_transform[0].inverse()*transform[i-1]);
                    }
                    crop.setInputCloud(tmp_cloud);
                    crop.setKeepOrganized(true);
                    crop.setUserFilterValue(0.1f);
                    crop.setKeepOrganized(true);
                    crop.filter(*tmp_cloud);
                    sor.setInputCloud(tmp_cloud);
                    // sor.setNegative(true);
                    sor.filter(*tmp_cloud);

                    outrem.setInputCloud(tmp_cloud);
                    outrem.filter(*tmp_cloud);
                    
                    std::cout<< tmp_cloud->points.size()<<std::endl;
                    const int count = tmp_cloud->points.size();
                    
                    cloud->points_.reserve(count);
                    cloud->colors_.reserve(count);

                    for(int j=0; j<tmp_cloud->points.size(); j++)
                    {
                        if (!tmp_cloud->points[j].x && !tmp_cloud->points[j].y && !tmp_cloud->points[j].z)
                            continue;
                        Eigen::Vector3d point(tmp_cloud->points[j].x, tmp_cloud->points[j].y, tmp_cloud->points[j].z);
                        Eigen::Vector3d color((double)tmp_cloud->points[j].r /255, (double)tmp_cloud->points[j].g /255, (double)tmp_cloud->points[j].b /255);  
                                          
                        cloud->points_.push_back(point);
                        cloud->colors_.push_back(color);
                    }
                    // std::cout<<std::endl<< cloud->points_.size()<<" "<< cloud->colors_.size();
                    // open3d::io::WritePointCloudToPLY("tmp_cloud.ply", *cloud, true);
                     const double normal_radius = 0.02;
                     open3d::geometry::KDTreeSearchParamHybrid normals_params(normal_radius, 30);
                     const bool fast_normal_computation = true;
                     cloud->EstimateNormals(normals_params, fast_normal_computation);
                     // Incorporate the assumption that normals should be pointed towards the camera
                     cloud->OrientNormalsTowardsCameraLocation(Eigen::Vector3d(0, 0, 0));
                     
                    if(i)
                       { 

                        std::shared_ptr<open3d::geometry::PointCloud> d_cloud = cloud->VoxelDownSample(0.01);

                        open3d::pipelines::registration::TransformationEstimationForColoredICP transform_estimate(0.96);
                        open3d::pipelines::registration::ICPConvergenceCriteria criteria(1e-16, 1e-16, 500);
                        auto result = open3d::pipelines::registration::RegistrationColoredICP(
                            *d_cloud,
                            *d_first_cloud,
                            0.05,
                            Eigen::Matrix4d::Identity(),
                            transform_estimate,
                            criteria
                        );
                       auto transform4x4 = result.transformation_.cast<float>();
                       cloud->Transform(transform4x4.cast<double>());
                       }
                    else{
                      (*d_first_cloud) = (*cloud);
                      d_first_cloud = d_first_cloud->VoxelDownSample(0.01);
                   }
                 
                   
                   (*cloud_all) = (*cloud_all) + (*cloud);

                    viewer.ClearGeometries();
                    viewer.GetRenderOption().SetPointSize(1);
                    viewer.AddGeometry(cloud_all);
                    viewer.UpdateRender();
                    viewer.PollEvents();
                    
                    
                }
            else 
            {
                std::cout<<"fail pointclouds";
                break;
            }
            it[i]++;
            
            
        }
            
            viewer.Run();

    }
   


    return 0;
}