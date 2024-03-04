#include <ros/ros.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>

#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>

// #include <moveit_visual_tools/moveit_visual_tools.h>
using json = nlohmann::json;

#define LOOP_RATE 50

#define IF_TEST false
#define IF_CONFIG false


 void back_to_initial(std::string PLANNING_GROUP, std::vector<double> goal_joint_positions)
{
  if(IF_TEST)
  return;
   // before move following command, use moveit! to initialize pose

  moveit::planning_interface::MoveGroupInterface move_group_interface(PLANNING_GROUP);

  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
  
  const moveit::core::JointModelGroup* joint_model_group = move_group_interface.getCurrentState()->getJointModelGroup(PLANNING_GROUP);
   moveit::core::RobotStatePtr current_state = move_group_interface.getCurrentState();
  std::vector<double> joint_group_positions;
  current_state->copyJointGroupPositions(joint_model_group, joint_group_positions);

  std::cout<<" joint_group_position:"<<std::endl;
  for(int i=0; i<joint_group_positions.size(); i++)
  {
    std::cout<<joint_group_positions[i]<<" ";
  }
  std::cout<<std::endl;

    move_group_interface.setMaxVelocityScalingFactor(0.1);
    move_group_interface.setMaxAccelerationScalingFactor(0.1);

    move_group_interface.setJointValueTarget(goal_joint_positions);

    moveit::planning_interface::MoveGroupInterface::Plan joint_plan;
    bool success = (move_group_interface.plan(joint_plan) == moveit::core::MoveItErrorCode::SUCCESS);
   
    if (success)
     move_group_interface.move();
    else
    {
      ROS_ERROR("plan failure");
      return ;
    }
  
    current_state = move_group_interface.getCurrentState();
    current_state->copyJointGroupPositions(joint_model_group, joint_group_positions);
    std::cout<<"current joint_group_position:"<<std::endl;
  for(int i=0; i<joint_group_positions.size(); i++)
  {
    std::cout<<joint_group_positions[i]<<" ";
  }
  std::cout<<std::endl;
}

int  readCommand(std::string file_path, std::vector<double>& timestamp_vector, std::vector<std::vector<double>> & points_vector )
{
   std::string line;
    std::ifstream myfile;
    
    myfile.open(file_path);
    
    int points_num = 0;

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
              timestamp_vector.push_back(stod(one_point));

            }
            else
            one_points.push_back(stod(one_point));
          }

            points_vector.push_back(one_points) ;
            points_num++;
        }
        myfile.close();
    }
    else{
      ROS_ERROR("%s file do not exist", file_path);
    }
    return points_num;
}

trajectory_msgs::JointTrajectory formCommand(std::string type, std::vector<double> positions, int seq)
{
  trajectory_msgs::JointTrajectory command;
  command.header.frame_id="";
  command.header.seq =seq;
  command.header.stamp = ros::Time::now();
  if (type=="rh")
  {
    command.joint_names = {"rh_FFJ4","rh_FFJ3","rh_FFJ2","rh_FFJ1","rh_LFJ5","rh_LFJ4","rh_LFJ3","rh_LFJ2","rh_LFJ1",
    "rh_MFJ4","rh_MFJ3","rh_MFJ2","rh_MFJ1","rh_RFJ4","rh_RFJ3","rh_RFJ2","rh_RFJ1","rh_THJ5","rh_THJ4","rh_THJ3","rh_THJ2","rh_THJ1" };
  }
  else if (type=="rh_wr")
  {
    command.joint_names = {"rh_WRJ2", "rh_WRJ1" };
  }
  else if (type=="ra")
  {
    command.joint_names = {"ra_shoulder_pan_joint", "ra_shoulder_lift_joint", "ra_elbow_joint", "ra_wrist_1_joint","ra_wrist_2_joint", "ra_wrist_3_joint" };
  }
  
  std::vector<trajectory_msgs::JointTrajectoryPoint> points;
  trajectory_msgs::JointTrajectoryPoint point;
  
  point.positions = positions;
  if (type =="rh_wr" || type=="ra")
  {
    point.time_from_start.fromNSec(8000000);
  }
  else if (type =="rh")
  {
    point.time_from_start.fromNSec(5000000);
  }
 
  points.push_back(point);
  command.points = points;

  return command;
}

void fake_publish_command(trajectory_msgs::JointTrajectory command)
{
  return;
   std::cout<<"header.frame_id "<<command.header.frame_id<<std::endl<<
                "header.seq "<<command.header.seq<<std::endl
                <<"header.stamp "<<command.header.stamp<<std::endl
                <<"joint_names ";
                for(int i=0; i<command.joint_names.size(); i++)
                std::cout << command.joint_names[i] <<" ";
                std::cout <<std::endl<< "positions ";
                for(int i=0; i<command.points.size(); i++)
                {
                  std::cout<< command.points[i]<<" ";
                }
                std::cout<<std::endl;

}

int main(int argc, char **argv)
{
  if(LOOP_RATE >100)
  return -1;

    ros::init(argc, argv, "command_pub");
    ros::NodeHandle n;
    ros::AsyncSpinner spinner(1);
   

    ros::Publisher ra_command_pub = n.advertise<trajectory_msgs::JointTrajectory>("/ra_trajectory_controller/command", 1000, true);
    ros::Publisher rh_wr_command_pub = n.advertise<trajectory_msgs::JointTrajectory>("/rh_wr_trajectory_controller/command", 1000, true);
    ros::Publisher rh_command_pub = n.advertise<trajectory_msgs::JointTrajectory>("/rh_trajectory_controller/command", 1000, true);
    
    ros::Rate loop_rate(LOOP_RATE);
  std::vector<std::vector<double>> ra_point_vector;
  std::vector<double> ra_timestamp_vector;

  std::vector<std::vector<double>> rh_wr_point_vector;
  std::vector<double> rh_wr_timestamp_vector;

  std::vector<std::vector<double>> rh_point_vector;
  std::vector<double> rh_timestamp_vector;

  int ra_current_points = 0, rh_wr_current_points = ra_current_points, rh_current_points = 0 ;

  std::string prefix = "/home/user/IntelligentHand/drive_ws/bags/yibu_broken_";

    // // road from text
    int ra_points_num=readCommand(prefix + "ra_points.txt", ra_timestamp_vector, ra_point_vector);
    int rh_wr_points_num=readCommand(prefix + "rh_wr_points.txt", rh_wr_timestamp_vector, rh_wr_point_vector);
    int rh_points_num = readCommand(prefix + "rh_points.txt", rh_timestamp_vector, rh_point_vector);

  // start spin in ROS
  spinner.start();


// read desire points from configuration file and reach it through moveit!
  if(!IF_TEST )
  {

      if (IF_CONFIG)
        {
        std::ifstream read_current_value_file("/home/user/IntelligentHand/drive_ws/src/command_pub/config/current_value.json");
        if (read_current_value_file.is_open()) {
        json read_current_value = json::parse(read_current_value_file);
        read_current_value_file.close();
        ra_current_points = read_current_value["ra_value"];
        rh_wr_current_points = read_current_value["rh_wr_value"]; 
        rh_current_points = read_current_value["rh_value"];
      }
      else 
      {
        ROS_ERROR("configuration file open failure");
        return -1;
       }
      }

    std::vector<double> goal_joint_positions;
    goal_joint_positions.insert(goal_joint_positions.end(),ra_point_vector[ra_current_points].begin(), ra_point_vector[ra_current_points].end());
    goal_joint_positions.insert(goal_joint_positions.end(), rh_wr_point_vector[rh_wr_current_points].begin(), rh_wr_point_vector[rh_wr_current_points].end());
    goal_joint_positions.insert(goal_joint_positions.end(), rh_point_vector[rh_current_points].begin(), rh_point_vector[rh_current_points].end());
    back_to_initial("right_arm_and_hand", goal_joint_positions);

  }
  
  //storge the nearest index of arm points
  std::vector<double> nearest_index;
  int rh_prt =0;
  for (int ra_prt=0; ra_prt<ra_timestamp_vector.size(); ra_prt++)
  {
      while(rh_timestamp_vector[rh_prt]<ra_timestamp_vector[ra_prt] && rh_prt<rh_timestamp_vector.size())
        rh_prt++;
      if(rh_prt == 0 || abs(rh_timestamp_vector[rh_prt] - ra_timestamp_vector[ra_prt]) < abs(rh_timestamp_vector[rh_prt-1] - ra_timestamp_vector[ra_prt]))
      {
        nearest_index.push_back(rh_prt);
      }
      else{
        nearest_index.push_back(rh_prt - 1);
      }
      if(ra_prt> nearest_index[ra_prt])
      {
        std::cout<<ra_prt<<" "<<rh_prt<<std::endl;
        return -1;
      }
  }

//   for(int i=0; i<nearest_index.size();i++)
//   std::cout<<i<<" "<<nearest_index[i]<<std::endl;
// return 0;
    int ra_seq =1, rh_wr_seq = 1, rh_seq =1;
    rh_points_num = nearest_index[ra_points_num - 1];
    while(ros::ok())
    {
      if(rh_current_points<nearest_index[ra_current_points])
      {
        trajectory_msgs::JointTrajectory rh_command = formCommand("rh",rh_point_vector[rh_current_points] ,rh_seq++);
        rh_current_points = (rh_current_points + 1) % rh_points_num;
        if(!rh_current_points)
        {
           std::vector<double> goal_joint_positions;
           goal_joint_positions.insert(goal_joint_positions.end(),ra_point_vector[0].begin(), ra_point_vector[0].end());
           goal_joint_positions.insert(goal_joint_positions.end(), rh_wr_point_vector[0].begin(), rh_wr_point_vector[0].end());
           goal_joint_positions.insert(goal_joint_positions.end(), rh_point_vector[0].begin(), rh_point_vector[0].end());
           back_to_initial("right_arm_and_hand", goal_joint_positions);
           rh_current_points =  ra_current_points  = rh_wr_current_points = 1;
        }
    
        if(IF_TEST)
        {
           //fake command_pub
            std::cout<<"rh_command_pub: " <<std::endl;
            fake_publish_command(rh_command);

        }
        else
        {
          rh_command_pub.publish(rh_command);
        }
      }
      if(rh_current_points >= nearest_index[ra_current_points] )
      {
              trajectory_msgs::JointTrajectory ra_command = formCommand("ra",ra_point_vector[ra_current_points], ra_seq++ );
              ra_current_points = (ra_current_points + 1) % ra_points_num;

              trajectory_msgs::JointTrajectory rh_wr_command = formCommand("rh_wr", rh_wr_point_vector[rh_wr_current_points], rh_wr_seq++);
              rh_wr_current_points = (rh_wr_current_points + 1) % rh_wr_points_num;
               if(!rh_wr_current_points)
              {
                std::vector<double> goal_joint_positions;
                goal_joint_positions.insert(goal_joint_positions.end(),ra_point_vector[0].begin(), ra_point_vector[0].end());
                goal_joint_positions.insert(goal_joint_positions.end(), rh_wr_point_vector[0].begin(), rh_wr_point_vector[0].end());
                goal_joint_positions.insert(goal_joint_positions.end(), rh_point_vector[0].begin(), rh_point_vector[0].end());
                back_to_initial("right_arm_and_hand", goal_joint_positions);
                ra_current_points = rh_wr_current_points = 0;
              }

              if(IF_TEST)
              {

                //fake ra_command_pub
                std::cout<<"ra_command_pub:" <<std::endl;
                fake_publish_command(ra_command);
                //fake rh_wr_command_pub
                std::cout<<"rh_wr_command_pub: "<<std::endl;
                fake_publish_command(rh_wr_command);

              }
              else{
                ra_command_pub.publish(ra_command);
                rh_wr_command_pub.publish(rh_wr_command);
              }

      }

      
     
      ros::spinOnce();
      loop_rate.sleep();
      ROS_INFO("ra:%d, rh_wr:%d, rh:%d", ra_current_points -1, rh_wr_current_points -1, rh_current_points -1);
      
    }
     
   
 if(!IF_TEST)
      {
        std::ofstream current_value_record_file("/home/user/IntelligentHand/drive_ws/src/command_pub/config/current_value.json");
        json current_value_record = {
          {"ra_value", ra_current_points},
          {"rh_wr_value", rh_wr_current_points},
          {"rh_value", rh_current_points}
        };
        current_value_record_file<<std::setw(4)<<current_value_record<<std::endl;
        current_value_record_file.close();
  
      }


//     return 0;

}