//多视图稀疏重建
//SIFI特征匹配计算位姿
//1,计算特征点和匹配
//2，查询左图匹配点是否有点已重建
//3，如何重建点有4个则p3p计算位资，否则八点法计算位姿
//4，计算深度
//5，判断深度是否重复
//6，存储重建坐标和对应点的在特征点中的位置。
//建图

#include <iostream>
#include <map>
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

using namespace std;
using namespace cv;
using namespace Eigen;



//string image_path = "./data_table/";
String image_path="./image/";
// 内参

//double fx =520.9, fy = 521.0, cx =325.1, cy =249.7;
//Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

double fx =2759.48, fy = 2764.16, cx =1520.69, cy =1006.81;
Mat K = (Mat_<double>(3, 3) << 2759.48, 0, 1520.69, 0, 2764.16, 1006.81, 0, 0, 1);
int imagenums = 10;
vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;

// struct pointmap{
//     Point3f pw;
//     int queryIdx;
// };


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

bool compute_transform(Mat R1,Mat t1,Mat &R,Mat &t,vector<Point3f> &pnp_points1,vector<Point2f> &pnp_points2,vector<Point2f> &points1,vector<Point2f> &points2 ){
    Mat essential_matrix;
    if(pnp_points1.size()>5){
        cout<<"pnp"<<endl;
        Mat r;
        solvePnPRansac(pnp_points1,pnp_points2,K,Mat(),r,t);
        cout<<r<<endl;
        if(r.rows==3){
            Rodrigues(r,R);//旋转向量2旋转矩阵
        }else{
            cout<<"8点法"<<endl;
            if(points1.size()<8) return false;
            double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
	        Point2d principle_point(K.at<double>(2), K.at<double>(5));
            essential_matrix = findEssentialMat(points1, points2,focal_length,principle_point,RANSAC,0.999,1.0);
            recoverPose(essential_matrix, points1, points2,R,t,focal_length,principle_point);
            //将R，t转换为相对世界坐标系的位姿态
            Mat a = R*t1+t;
            a.copyTo(t);
            Mat b = R*R1;
            b.copyTo(R);
        }
        return true;
    }else if(points1.size()>=8){
        //-- 计算本质矩阵
        cout<<"8点法"<<endl;
        cout<<"points1 size :"<<points1.size()<<endl;
        if(points1.size()<8) return 0;
        double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
	    Point2d principle_point(K.at<double>(2), K.at<double>(5));
        essential_matrix = findEssentialMat(points1, points2,focal_length,principle_point,RANSAC,0.999,1.0);
        recoverPose(essential_matrix, points1, points2,R,t,focal_length,principle_point);
        //将R，t转换为相对世界坐标系的位姿态
        Mat a = R*t1+t;
        a.copyTo(t);
        Mat b = R*R1;
        b.copyTo(R);
        return true;
    }
    cout<<"匹配点小于8对，无法计算位姿！！"<<endl;
    return false;   
}

Mat restruct(vector<Point2f> p1,vector<Point2f> p2,Mat R1,Mat t1,Mat R,Mat t ){
    //opencv 方式计算深度和不确定性
    Mat T1 = (Mat_<float>(3, 4) <<
        R1.at<double>(0, 0), R1.at<double>(0, 1), R1.at<double>(0, 2), t1.at<double>(0, 0),
        R1.at<double>(1, 0), R1.at<double>(1, 1), R1.at<double>(1, 2), t1.at<double>(1, 0),
        R1.at<double>(2, 0), R1.at<double>(2, 1), R1.at<double>(2, 2), t1.at<double>(2, 0)
    );
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
    vector<map<int,Point3f>> perpoint_map;
    map<int,Point3f> m0;
    Mat R1(Matx33d(
		1, 0, 0,
		0, 1, 0,
		0, 0, 1));
    Mat t1(Matx31d(
		0,
		0,
		0));
    perpoint_map.push_back(m0);
    images.push_back( imread(image_path+to_string(1)+".png",CV_LOAD_IMAGE_COLOR) );
    for ( int i=1; i<imagenums; i++ )
    {
        string path = image_path+to_string(i+1)+".png";
        //string path = image_path+to_string(i+1)+".jpg";
        images.push_back( imread(path,CV_LOAD_IMAGE_COLOR) );
        cout<<"第"<<i<<endl;
        vector<KeyPoint> keypoints1,keypoints2;
        Mat descriptors1,descriptors2;
        cout<<"perpoint_map size "<<perpoint_map[i-1].size()<<endl;
        cout<<"R1: "<<R1<<endl;
        cout<<"t1: "<<t1<<endl;
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
        vector<Point3f> pnp_points1;
        vector<Point2f> pnp_points2;//p3p计算位姿用点
        vector<DMatch>  goodmatches;
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
                goodmatches.push_back(matches[match_num][0]);
                if(perpoint_map[i-1].size()>5){
                    //perpoint[perpoint_num].queryIdx 存储的是上一次匹配的右图，并不是有序的，这种匹配尝试需要修改       
                    if(perpoint_map[i-1].count(matches[match_num][0].queryIdx)!=0){
                        
                        //该点在上一张图片中已重建
                        pnp_points1.push_back(perpoint_map[i-1][matches[match_num][0].queryIdx]);
                        pnp_points2.push_back(keypoints2[matches[match_num][0].trainIdx].pt);
                    } 
                }
                
            }       
        }               
        //计算位姿态
        Mat R , t; 
        cout<<"pnp_points1 size :"<<pnp_points1.size()<<endl;
        cout<<"pnp_points2 size :"<<pnp_points2.size()<<endl;
        if(!compute_transform(R1,t1,R,t,pnp_points1,pnp_points2,points1,points2)){
            showPointCloud(pointcloud);
            waitKey(0);
            cout<<"结束";
            return 0;
        }
        cout<<"R："<<R<<endl; 
        cout<<"t："<<t<<endl; 
        //cout<<"mask :"<<mask<<endl;
        //maskout_points(points1,mask);
        //maskout_points(points2,mask);
        cout<<"goodmatches size :"<<goodmatches.size()<<endl;
        cout<<"points1 size :"<<points1.size()<<endl;
        //重建三维点
        Mat pw = restruct(points1,points2,R1,t1,R,t);
        //建图
        map<int,Point3f> m2;
        cout<<pw.cols<<endl;
        for(int p_num=0;p_num<pw.cols;p_num++){
            Mat_<float> c =pw.col(p_num);
            Vector3d pwpoint(
                c(0)/c(3),
                c(1)/c(3),
                c(2)/c(3)
            );
            Point3f m_pw = Point3f(
                pwpoint(0,0),
                pwpoint(1,0),
                pwpoint(2,0)
            );
            m2.insert(pair<int,Point3f>(goodmatches[p_num].trainIdx,m_pw));
            //cout<<c<<endl;
            if(perpoint_map[i-1].count(goodmatches[p_num].queryIdx)==0){
                //在上一张图未重建，与上一张图的匹配没有匹配也带表，该点未出线过
                add2Pointcloud(pwpoint,double(100));
            }
            
        }
        R.copyTo(R1);
        t.copyTo(t1);
        perpoint_map.push_back(m2);
    }
    cout<<"坐标点数量："<<pointcloud.size()<<endl;
    showPointCloud(pointcloud);
    waitKey(0);
    cout<<"结束";
    return 0;
}