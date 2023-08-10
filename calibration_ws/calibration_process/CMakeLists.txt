cmake_minimum_required(VERSION 3.20.0)
project(testt VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()
find_package(nlohmann_json 3.2.0 REQUIRED)

find_package(Eigen3 REQUIRED NO_MODULE)


include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(testt src/main.cpp)

add_dependencies(testt 
nlohmann_json::nlohmann_json
)
target_link_libraries(testt
Eigen3::Eigen
nlohmann_json::nlohmann_json
)
target_include_directories(testt PUBLIC include)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
