#include <ros/ros.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#define IF_TEST false

std::vector<double> ra_point_vector[100000];
std::vector<double> ra_timestamp_vector;

std::vector<double> rh_wr_point_vector[100000];
std::vector<double> rh_wr_timestamp_vector;

std::vector<double> rh_point_vector[100000];
std::vector<double> rh_timestamp_vector;

 int ra_current_points = 0, rh_wr_current_points = ra_current_points ;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "command_pub");
    ros::NodeHandle n;

    ros::Publisher ra_command_pub = n.advertise<trajectory_msgs::JointTrajectory>("/ra_trajectory_controller/command", 1000, true);
    ros::Publisher rh_wr_command_pub = n.advertise<trajectory_msgs::JointTrajectory>("/rh_wr_trajectory_controller/command", 1000, true);
    
    ros::Rate loop_rate(20);

    // road from text

    //ra process
    std:: string ra_path = "/home/user/test_ra_points.txt";
    std::string line;
    std::ifstream myfile;
    
    myfile.open(ra_path);
    
    int ra_points_num = 0;

    if (myfile.is_open()) {
        while (getline(myfile, line)) {
          std::stringstream line_stringstream(line);
          std::vector<double> one_points;
          std::string one_point;
          bool if_time = true;
          while (getline(line_stringstream, one_point, ' '))
          {
            if(if_time)
            {
              if_time = false;
              ra_timestamp_vector.push_back(stod(one_point));
            }
            else
            one_points.push_back(stod(one_point));
          }

            ra_point_vector[ra_points_num++] = one_points;

        }
        myfile.close();
    }
    //rh_wr process
    std::string rh_wr_path = "/home/user/test_rh_wr_points.txt";
    myfile.open(rh_wr_path);

    int rh_wr_points_num=0;

    if(myfile.is_open())
    {
      while(getline(myfile, line))
      {
          std::stringstream line_stringstream(line);
          std::vector<double> one_points;
          std::string one_point;
          bool if_time = true;
          while (getline(line_stringstream, one_point, ' '))
          {
            if(if_time)
            {
              if_time = false;
              rh_wr_timestamp_vector.push_back(stod(one_point));
            }
            else
            one_points.push_back(stod(one_point));
          }

            rh_wr_point_vector[rh_wr_points_num++] = one_points;
      }
      myfile.close();
      }

  if(!IF_TEST)
  {
    std::ifstream read_current_value_file("/home/user/IntelligentHand/drive_ws/src/command_pub/config/current_value.json");
    if (!read_current_value_file.is_open()) {
            std::cerr << "Error in open current value configuration file" << std::endl;
            return 1;
        }
    json read_current_value = json::parse(read_current_value_file);
    read_current_value_file.close();
    ra_current_points = read_current_value["ra_value"];
    rh_wr_current_points = read_current_value["rh_wr_value"]; 
  }

  
  // before move following command, use moveit! to initialize pose

  
  
   
    int ra_seq =1, rh_wr_seq = 1;
    while(ros::ok())
    {
      trajectory_msgs::JointTrajectory ra_command;
      ra_command.header.frame_id="";
      ra_command.header.seq =ra_seq++;
      ra_command.header.stamp = ros::Time::now();
      ra_command.joint_names = {"ra_shoulder_pan_joint", "ra_shoulder_lift_joint", "ra_elbow_joint", "ra_wrist_1_joint","ra_wrist_2_joint", "ra_wrist_3_joint" };
      std::vector<trajectory_msgs::JointTrajectoryPoint> ra_points;
      trajectory_msgs::JointTrajectoryPoint ra_point;
      
      ra_point.positions = ra_point_vector[ra_current_points];
      ra_current_points = (ra_current_points + 1) % ra_points_num;

      ra_point.time_from_start.fromNSec(8000000);
      ra_points.push_back(ra_point);
      ra_command.points = ra_points;

      trajectory_msgs::JointTrajectory rh_wr_command;
      rh_wr_command.header.frame_id="";
      rh_wr_command.header.seq =rh_wr_seq++;
      rh_wr_command.header.stamp = ros::Time::now();
      rh_wr_command.joint_names = {"rh_WRJ2", "rh_WRJ1" };
      std::vector<trajectory_msgs::JointTrajectoryPoint> rh_wr_points;
      trajectory_msgs::JointTrajectoryPoint rh_wr_point;
      
      rh_wr_point.positions = rh_wr_point_vector[rh_wr_current_points];
      rh_wr_current_points = (rh_wr_current_points + 1) % rh_wr_points_num;

      rh_wr_point.time_from_start.fromNSec(8000000);
      rh_wr_points.push_back(rh_wr_point);
      rh_wr_command.points = rh_wr_points;

      if(IF_TEST)
      {
        // for debug

      //   int old_ra_value, new_ra_value;
      //   new_ra_value = (ra_current_points -1 +ra_points_num) % ra_points_num;
      //   old_ra_value = (ra_current_points -2 +ra_points_num) % ra_points_num;
      //   for( int i=0; i<ra_point_vector[new_ra_value].size();i++)
      //   {
      //    double gap_value = ra_point_vector[new_ra_value][i] - ra_point_vector[old_ra_value][i];
      //    std::cout<<gap_value<<" ";
      //    if(ra_current_points>1&&gap_value > max_gap_value[i])
      //    {
      //      max_gap_index[i] = ra_current_points -1;
      //      max_gap_value[i] = gap_value;
      //    }
      // }

      //   std::cout << std::endl;     

        //fake ra_command_pub
        std::cout<<"ra_command_pub:"<<ra_command_pub.getTopic()<<std::endl
        <<"header.frame_id "<<ra_command.header.frame_id<<std::endl<<
        "header.seq "<<ra_command.header.seq<<std::endl
        <<"header.stamp "<<ra_command.header.stamp<<std::endl
        <<"joint_names ";
        for(int i=0; i<ra_command.joint_names.size(); i++)
        std::cout << ra_command.joint_names[i] <<" ";
        std::cout <<std::endl<< "positions ";
        for(int i=0; i<ra_command.points.size(); i++)
        {
          std::cout<< ra_command.points[i]<<" ";
        }
        std::cout<<std::endl;

        //fake rh_wr_command_pub
        std::cout<<"rh_wr_command_pub:"<<rh_wr_command_pub.getTopic()<<std::endl
        <<"header.frame_id "<<rh_wr_command.header.frame_id<<std::endl<<
        "header.seq "<<rh_wr_command.header.seq<<std::endl
        <<"header.stamp "<<rh_wr_command.header.stamp<<std::endl
        <<"joint_names ";
        for(int i=0; i<rh_wr_command.joint_names.size(); i++)
        std::cout << rh_wr_command.joint_names[i] <<" ";
        std::cout <<std::endl<< "positions ";
        for(int i=0; i<rh_wr_command.points.size(); i++)
        {
          std::cout<< rh_wr_command.points[i]<<" ";
        }
        std::cout<<std::endl;

      }
      else{
        ra_command_pub.publish(ra_command);
        rh_wr_command_pub.publish(rh_wr_command);

      }

      
      ROS_INFO("ra_current_num: %d, rh_wr_currenct_num: %d", ra_current_points, rh_wr_current_points);
      ros::spinOnce();
      loop_rate.sleep();
      
    }
    if(!IF_TEST)
    {
      std::ofstream current_value_record_file("/home/user/IntelligentHand/drive_ws/src/command_pub/config/current_value.json");
      json current_value_record = {
        {"ra_value", ra_current_points},
        {"rh_wr_value", rh_wr_current_points}
      };
      current_value_record_file<<std::setw(4)<<current_value_record<<std::endl;
      current_value_record_file.close();
 
    }
    //// for debug
    // for(int i=0; i<6; i++)
    // std::cout << max_gap_index[i] <<" "<<max_gap_value[i]<<std::endl;


    return 0;

}