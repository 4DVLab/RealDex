#include<nlohmann/json.hpp>
#include<iostream>
#include<fstream>
#include<string>
#include <experimental/filesystem>
#include <Eigen/Eigen>

using json = nlohmann::json;
namespace fs = std::experimental::filesystem;
using namespace std;


int main()
{

    ifstream i1("/home/lab4dv/IntelligentHand/calibration_ws/k4a-calibration/input/cn00.json");
   
       if(!i1.is_open())
        {
        std::cout << "open file failed" << std::endl;
        return -1;
        }
        else{
            cout<<"open file cn00.json success"<<endl;
        }
    json j1 = json::parse(i1);
     i1.close();
    Eigen::Affine3d T1=Eigen::Affine3d::Identity();
    T1.rotate( Eigen::Quaterniond( j1["value0"]["rotation"]["w"],  j1["value0"]["rotation"]["x"],  j1["value0"]["rotation"]["y"], j1["value0"]["rotation"]["z"]).toRotationMatrix());
    T1.translation()<<j1["value0"]["translation"]["x"],  j1["value0"]["translation"]["y"], j1["value0"]["translation"]["z"];
   


    for(int i=1; i<4; i++)
    {
        string str = "/home/lab4dv/IntelligentHand/calibration_ws/k4a-calibration/input/cn0"+ to_string(i)+ ".json";
        ifstream i2(str);

        if(!i2.is_open())
        {
        std::cout << "open file failed" << std::endl;
        return -1;
        }
        else{
            cout<<"open file cn0"+to_string(i)+".json success"<<endl;
        }

    json j2 = json::parse(i2);
    i2.close();
    Eigen::Affine3d T2 =Eigen::Affine3d::Identity();
    // T1.matrix() << ,  j1["value0"]["rotation"]["x"],  j1["value0"]["rotation"]["y"],  j1["value0"]["rotation"]["z"],  j1["value0"]["rotation"]["w"],  0,  0,  0,  1;
    // T2.matrix() << j2["value0"]["translation"]["x"],  j2["value0"]["translation"]["y"], j2["value0"]["translation"]["z"],  j2["value0"]["rotation"]["x"],  j2["value0"]["rotation"]["y"],  j2["value0"]["rotation"]["z"],  j2["value0"]["rotation"]["w"],  0,  0,  0,  1;
    

   
    T2.rotate(Eigen::Quaterniond( j2["value0"]["rotation"]["w"],  j2["value0"]["rotation"]["x"],  j2["value0"]["rotation"]["y"], j2["value0"]["rotation"]["z"]).toRotationMatrix());
    T2.translation()<<j2["value0"]["translation"]["x"],  j2["value0"]["translation"]["y"], j2["value0"]["translation"]["z"];

    Eigen::Affine3d T = T1.inverse() * T2;
  
    std::cout << T.matrix() << std::endl;

    Eigen::Quaterniond q(T.rotation());
    Eigen::Vector3d t(T.translation());
    json j3;
    j3["value0"]["rotation"]["x"] = q.x();
    j3["value0"]["rotation"]["y"] = q.y();
    j3["value0"]["rotation"]["z"] = q.z();
    j3["value0"]["rotation"]["w"] = q.w();
    j3["value0"]["translation"]["x"] = t.x();
    j3["value0"]["translation"]["y"] = t.y();
    j3["value0"]["translation"]["z"] = t.z();
    std::cout << j3.dump(4) << std::endl;

    //  Save JSON file
    std::ofstream file("/home/lab4dv/IntelligentHand/calibration_ws/calibration_process/data/cali0"+to_string(i)+".json");
   
    file<< j3.dump(4);

    file.flush();
    file.close();
    }

   

    return 0;
}