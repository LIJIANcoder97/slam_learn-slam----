//双视图重建
//SIFI特征匹配计算位姿
//建图

#include <iostream>
#include <algorithm>
#include <Eigen/Core>
// 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Dense>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <vector>
#include <string>
#include "opencv4/opencv2/imgcodecs/legacy/constants_c.h"
#include <pangolin/pangolin.h>
#include <unistd.h>
//#include <pangolin/pangolin.h>
//#include <unistd.h>

using namespace std;
using namespace cv;
using namespace Eigen;


string image_path = "./data_table/";
//String image_path="./image/";
// 内参

double fx =520.9, fy = 521.0, cx =325.1, cy =249.7;
Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

// double fx =2759.48, fy = 2764.16, cx =1520.69, cy =1006.81;
// Mat K = (Mat_<double>(3, 3) << 2759.48, 0, 1520.69, 0, 2764.16, 1006.81, 0, 0, 1);
int imagenums = 2;
vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;
//显示点云图
void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}
void add2Pointcloud(Vector3d Pw,double px){
    Vector4d point(0, 0, 0, px/255); // 前三维为xyz,第四维为颜色
    point(0,0)=Pw(0,0);
    point(1,0)=Pw(1,0);
    point(2,0)=Pw(2,0);
    pointcloud.push_back(point);
}





vector<Point2f> px2cam(vector<Point2f> p){
    vector<Point2f> re;
    for(int i=0;i<p.size();i++){
        re.push_back(
            Point2f
            (
                (p[i].x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                (p[i].y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            ) 
        );
    }
    return re;
}
void maskout_points(vector<Point2f> &points, Mat mask){
    vector<Point2f> copy = points;
	points.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
		{
			points.push_back(copy[i]);
		}
	}
}


Mat restruct(vector<Point2f> p1,vector<Point2f> p2,Mat R,Mat t ){
    //opencv 方式计算深度和不确定性
    Mat T1 = (Mat_<float>(3, 4) <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0);
    Mat T2 = (Mat_<float>(3, 4) <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
    );
    Mat pw;
    //cout<<"调用opencv三角化函数"<<endl;
    triangulatePoints(T1, T2, px2cam(p1), px2cam(p2), pw);
    //cout<<"调用opencv三角化函数结束"<<endl;

    return pw;
}


int main(int argc, char** argv){
    //读取图片路径信息
    vector<Mat> images; 
    
    images.push_back( imread(image_path+to_string(1)+".png",CV_LOAD_IMAGE_COLOR) );
    for ( int i=1; i<imagenums; i++ )
    {
        string path = image_path+to_string(i+1)+".png";
        //string path = image_path+to_string(i+1)+".jpg";
        images.push_back( imread(path,CV_LOAD_IMAGE_COLOR) );
        vector<KeyPoint> keypoints1,keypoints2;
        Mat descriptors1,descriptors2;
        //提取特征点作匹配
        Ptr<FeatureDetector> detector = SIFT::create(0,3,0.04,10);
        Ptr<DescriptorExtractor> descriptor = SIFT::create();
        Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
        detector->detect(images[i-1],keypoints1);
        detector->detect(images[i],keypoints2);
        descriptor->compute(images[i-1],keypoints1,descriptors1);
        descriptor->compute(images[i],keypoints2,descriptors2);
        cout<<"keypoints size :"<<keypoints1.size()<<endl;
        cout<<"keypoints size :"<<keypoints2.size()<<endl;
        cout<<"descriptors1 size :"<<descriptors1.size()<<endl;
        //匹配并筛选优秀的匹配点
        vector<vector<DMatch>>  matches;                   //匹配点列表
        cout<<"计算匹配点"<<endl;
        matcher->knnMatch(descriptors1, descriptors2, matches,2);//每一个匹配返回2个匹配点
        cout<<"matches size "<<matches.size()<<endl;
        vector<Point2f> points1;
        vector<Point2f> points2;//匹配点坐标
        float min_dis=FLT_MAX;
        for(int match_num =0;match_num<matches.size();match_num++){

            if(matches[match_num][0].distance<0.6*matches[match_num][1].distance){
                min_dis = min(min_dis,matches[match_num][0].distance);
                 
            }
        }
        cout<<"min_dis: "<<min_dis<<endl;
        for(int match_num =0;match_num<matches.size();match_num++){

            if(matches[match_num][0].distance>5*max(min_dis,10.0f)) continue;
            if(matches[match_num][0].distance<=0.6*matches[match_num][1].distance){
                //是优秀的匹配点
                points1.push_back(keypoints1[matches[match_num][0].queryIdx].pt);
                points2.push_back(keypoints2[matches[match_num][0].trainIdx].pt);
            }       
        }

                
        //计算位姿态
        Mat essential_matrix;
        Mat R , t; 
        //-- 计算本质矩阵
        cout<<"points1 size :"<<points1.size()<<endl;
        if(points1.size()<8) return 0;
        cout<<points2<<endl;
        Mat mask;
        double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
	    Point2d principle_point(K.at<double>(2), K.at<double>(5));
        essential_matrix = findEssentialMat(points1, points2,focal_length,principle_point,RANSAC,0.999,1.0,mask);
        cout<<"本质矩阵："<<essential_matrix<<endl; 
        
        recoverPose(essential_matrix, points1, points2,R,t,focal_length,principle_point,mask);
        cout<<"R："<<R<<endl; 
        cout<<"t："<<t<<endl; 
        //cout<<"mask :"<<mask<<endl;
        //maskout_points(points1,mask);
        //maskout_points(points2,mask);
        cout<<"points1 size :"<<points1.size()<<endl;
        //重建三维点
        Mat pw = restruct(points1,points2,R,t);
        cout<<"坐标点数量："<<pw.cols<<endl;
        //建图
        cout<<pw.rows<<endl;
        cout<<pw.cols<<endl;
        for(int p_num=0;p_num<pw.cols;p_num++){
            Mat_<float> c =pw.col(p_num);
            Vector3d pwpoint(
                c(0)/c(3),
                c(1)/c(3),
                c(2)/c(3)
            );
            //cout<<c<<endl;
            add2Pointcloud(pwpoint,double(100));
        }
    }
    showPointCloud(pointcloud);
    waitKey(0);
    cout<<"结束";
    return 0;
}