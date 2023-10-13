project(edgeExtracting)
find_package(OpenCV REQUIRED)
add_executable(edgeExtracting edgeExtracting.cpp)
# 链接OpenCV库
target_link_libraries(edgeExtracting ${OpenCV_LIBS})
