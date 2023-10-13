# 声明cmake最低版本
cmake_minimum_required(VERSION 2.8)
project(3DReconstruction)
include_directories("/usr/include/eigen3")
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(singlereconstruction singlereconstruction.cpp)
add_executable(multiviewreconstruction multiviewreconstruction.cpp)
# 链接OpenCV库
target_link_libraries(singlereconstruction ${OpenCV_LIBS})
target_link_libraries(singlereconstruction ${Pangolin_LIBRARIES})
target_link_libraries(multiviewreconstruction ${OpenCV_LIBS})
target_link_libraries(multiviewreconstruction ${Pangolin_LIBRARIES})
target_link_libraries(multiviewreconstruction ${CERES_LIBRARIES})
