#include<nlohmann/json.hpp>
#include<iostream>
#include<fstream>
#include<string>
#include <experimental/filesystem>
#include <Eigen/Eigen>

using json = nlohmann::json;
namespace fs = std::experimental::filesystem;



int main()
{
    std::ifstream i1("/home/lab4dv/intelligentHand/IntelligentHand/calibration_ws/calibration_process/data/calibration.json");

    if(!i1.is_open())
    {
        std::cout << "open file failed" << std::endl;
        return -1;
    }

    json j1 = json::parse(i1);
    i1.close(); 
    Eigen::Affine3d T1=Eigen::Affine3d::Identity(), T2=Eigen::Affine3d::Identity(), TT;

     Eigen::Matrix3d T;
     T.matrix()<< j1["0"]["RT"][0], j1["0"]["RT"][1], j1["0"]["RT"][2],
     j1["0"]["RT"][4], j1["0"]["RT"][5], j1["0"]["RT"][6],
     j1["0"]["RT"][8], j1["0"]["RT"][9], j1["0"]["RT"][10];
     
     T1.rotate(T);

    T1.translation()<<j1["0"]["RT"][3], j1["0"]["RT"][7], j1["0"]["RT"][11];

    // std::cout <<T1.matrix() << std::endl;

    T.matrix()<< j1["1"]["RT"][0], j1["1"]["RT"][1], j1["1"]["RT"][2],
     j1["1"]["RT"][4], j1["1"]["RT"][5], j1["1"]["RT"][6],
     j1["1"]["RT"][8], j1["1"]["RT"][9], j1["1"]["RT"][10];
     
     T2.rotate(T);

    T2.translation()<<j1["1"]["RT"][3], j1["1"]["RT"][7], j1["1"]["RT"][11];

    TT = T2*T1.inverse();

    std::cout<<T1.matrix() << std::endl<< T2.matrix()<<std::endl<<TT.matrix();


    // Eigen::Quaterniond q(T.rotation());
    // Eigen::Vector3d t(T.translation());
    // json j3;
    // j3["value0"]["rotation"]["x"] = q.x();
    // j3["value0"]["rotation"]["y"] = q.y();
    // j3["value0"]["rotation"]["z"] = q.z();
    // j3["value0"]["rotation"]["w"] = q.w();
    // j3["value0"]["translation"]["x"] = t.x();
    // j3["value0"]["translation"]["y"] = t.y();
    // j3["value0"]["translation"]["z"] = t.z();
    // std::cout << j3.dump(4) << std::endl;

     // Save JSON file
    // std::ofstream file("/home/yyxunn/intelligent_hand/k4a-calibration/output/cali01.json");
   
    // file<< j3.dump(4);

    // file.flush();
    // file.close();

    return 0;
}