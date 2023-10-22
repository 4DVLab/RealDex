cmake_minimum_required(VERSION 3.16.0)
project(calibration_process VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(PCL REQUIRED COMPONENTS )
find_package(Open3D REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  tf
  rosbag
)

# set(LIB_VTK_FILTERS_GENERAL_DIR /usr/local/lib)
# add_library(libvtkFiltersGeneral SHARED IMPORTED)
# set_target_properties(libvtkFiltersGeneral PROPERTIES IMPORTED_LOCATION ${LIB_VTK_FILTERS_GENERAL_DIR}/libvtkFiltersGeneral-9.3.so.9.3)

message(STATUS "Open3D : ${Open3D_LIBRARIES}")

include_directories(
# include
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${Eigen3_INCLUDE_DIRS})


add_executable(k4acalibration_process src/k4acalibration_trans.cpp)

add_dependencies(k4acalibration_process
nlohmann_json::nlohmann_json
)
target_link_libraries(k4acalibration_process
Eigen3::Eigen
nlohmann_json::nlohmann_json
)
target_include_directories(k4acalibration_process PUBLIC include)

add_executable(read_bag src/readBag.cpp)
add_dependencies(read_bag 
nlohmann_json::nlohmann_json
)

target_link_libraries(read_bag
# libvtkFiltersGeneral
Eigen3::Eigen
${PCL_LIBRARIES}
${catkin_LIBRARIES}
# ${Open3D_LIBRARIES}
nlohmann_json::nlohmann_json
)
target_include_directories(read_bag PUBLIC include)




set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
