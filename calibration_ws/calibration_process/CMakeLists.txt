cmake_minimum_required(VERSION 3.16.0)
project(calibration_process VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()
find_package(nlohmann_json 3.2.0 REQUIRED)

find_package(Eigen3 REQUIRED NO_MODULE)


include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(k4acalibration_process src/k4acalibration_trans.cpp)

add_dependencies(k4acalibration_process
nlohmann_json::nlohmann_json
)
target_link_libraries(k4acalibration_process
Eigen3::Eigen
nlohmann_json::nlohmann_json
)
target_include_directories(k4acalibration_process PUBLIC include)

add_executable(cali1_trans src/cali1_trans.cpp)

add_dependencies(cali1_trans
nlohmann_json::nlohmann_json
)
target_link_libraries(cali1_trans
Eigen3::Eigen
nlohmann_json::nlohmann_json
)
target_include_directories(cali1_trans PUBLIC include)




set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
