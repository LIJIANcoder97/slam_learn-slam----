# 声明cmake最低版本
cmake_minimum_required(VERSION 2.8)
# 寻找opencv
find_package(OpenCV REQUIRED)

#声明 cmake工程
project(HelloSlam)


include_directories(${OpenCV_INCLUDE_DIRS})

#添加可执行程序
# add_executable(程序名  源代码文件)
add_executable(helloSLAM helloslam.cpp)
add_executable(imageBasics imageBasic.cpp)

# 链接OpenCV库
target_link_libraries(imageBasics ${OpenCV_LIBS})
