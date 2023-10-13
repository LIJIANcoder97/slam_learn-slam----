#include<opencv4/opencv2/opencv.hpp>
#include<vector>
#include<string>
#include<eigen3/Eigen/Core>
#include<pangolin/pangolin.h>
#include<unistd.h>

using namespace std;
using namespace Eigen;

//file path
string left_file = "./data/left.png";
string right_file = "./data/right.png";

//pangolin 画图
void showPontCloud(const vector<Vector4d,Eigen::aligned_allocator<Vector4d>> &pointcloud);



//
int main(int argc,char **argv){
    //内参
    double fx = 718.856,fy = 718.856,cx=607.1928,cy=185.2157;
    //基线
    double b = 0.573;

    //read picture
    cv::Mat left = cv::imread(left_file,0);
    cv::Mat right = cv::imread(right_file,0);
}