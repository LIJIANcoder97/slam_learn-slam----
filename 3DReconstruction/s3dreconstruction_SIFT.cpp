// 3D重建（简单版）
// 读取多张（3-5）张图片和信息，读取内参
// 选2张（按顺序/通过比较匹配点数目选择）
// 计算SIFT匹配点，计算位姿，存储匹配点.       每张图片都需要存储自己的位姿，第一张图片的位资应初始化。
// 显示匹配
// 极线搜索，计算深度图
// 显示点云图
// 选择新的图片，与上一张计算匹配点，存储匹配点
// 选择已重建的点，使用p3p计算位姿。计算当位资
// 极线搜索，与上一张图做三角化，计算深度图
// 更新点云图
// 重复

//深度估计的点云图点的存储问题。
//1 ,由于图片按顺序匹配，当深度收敛和像素点没有匹配时记录深度估计后三维点坐标
//2 ,当图片是最后一张时存储三维坐标。

#include <iostream>
//#include <algorithm>
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


//debug 模式
bool debug=false;
// 文件路径
string image_path = "./data_table/";
//string image_path = "./data_building/images/";
//string image_path = "./";
// 内参
double fx =520.9, fy = 521.0, cx =325.1, cy =249.7;
Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
double min_depth=1;
int imagenums = 10;
const int ncc_window_size = 3;    // NCC 取的窗口半宽度
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC窗口面积
const double min_cov = 0.1;     // 收敛判定：最小方差
const double max_cov = 10;      // 发散判定：最大方差

struct dpoint {           //深度图存储 特征点匹配信息，世界坐标，深度信息
    double depth;//当前相机对应P点的深度，深度融合后有效
    double depth_cov2;
    Vector3d camPoisition;//相机坐标系下坐标
    bool ismatch;//是否有匹配点
    Point2d nextImage_poisition;//在下一张图上的匹配点
    vector<DMatch>thismatch;
    bool isreconstruction;
    dpoint():depth(0.0),ismatch(false),isreconstruction(false),depth_cov2(3.0){}
};
vector<vector<vector<dpoint>>> depthimages; //存储每张图片的dpoint信息
vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;
//点云图显示
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
// 像素到相机坐标系
Vector3d px2cam(const Vector2d px) {
    return Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

// 相机坐标系到像素
Vector2d cam2px(const Vector3d p_cam) {
    return Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}

//判断坐标是否在边界内
bool isinpx(int x,int y,int w,int h){
    if(x<ncc_window_size) return false;
    if(x>=w-ncc_window_size) return false;
    if(y<ncc_window_size) return false;
    if(y>=h-ncc_window_size) return false;
    return true;
}
bool epipolarIsinpx(Vector2d p2,int w,int h,Vector2d epipolardirection,Vector2d &sp){
    if(isinpx(p2(0,0),p2(1,0),w,h)){
        sp=p2;
        return true;//在像素平面
    } 
    // y = x*Y/X
    //p2在在左
    if(epipolardirection(0,0)>0 && ncc_window_size-p2(0,0)>=0 && isinpx(ncc_window_size,int(ncc_window_size*epipolardirection(1,0)/epipolardirection(0,0)),w,h)){
        sp(0,0)=ncc_window_size;
        sp(1,0)=ncc_window_size*epipolardirection(1,0)/epipolardirection(0,0);
        return true;
    } 
    //p2在右
    if(epipolardirection(0,0)<0 && w-ncc_window_size-p2(0,0)<=0 && isinpx(w-ncc_window_size,int((w-ncc_window_size)*epipolardirection(1,0)/epipolardirection(0,0)),w,h)){
        sp(0,0)=w-ncc_window_size;
        sp(1,0)=(w-ncc_window_size)*epipolardirection(1,0)/epipolardirection(0,0);
        return true;
    }
    //p2在上
    if(epipolardirection(1,0)>0 && ncc_window_size-p2(1,0)>=0 && isinpx(int(ncc_window_size*epipolardirection(0,0)/epipolardirection(1,0)),ncc_window_size,w,h)){
        sp(0,0)=ncc_window_size*epipolardirection(0,0)/epipolardirection(1,0);
        sp(1,0)=ncc_window_size;
        return true;
    }
    if(epipolardirection(1,0)<0 && h-ncc_window_size-p2(1,0)<=0 && isinpx(int((h-ncc_window_size)*epipolardirection(0,0)/epipolardirection(1,0)),h-ncc_window_size,w,h)){
        sp(0,0)=(h-ncc_window_size)*epipolardirection(0,0)/epipolardirection(1,0);
        sp(1,0)=h-ncc_window_size;
        return true;
    }

    return false;     
}
// ncc匹配度计算
double NCC(Mat &per_img,Mat &curr_img,Vector2d p1,Vector2d p2){
    double p1_mean=0.0;
    double p2_mean=0.0;
    vector<double> p1_values, p2_values;
    for(int y=-ncc_window_size;y<=ncc_window_size;y++){
        for(int x=-ncc_window_size;x<=ncc_window_size;x++){
            double p1_v,p2_v;
            p1_v=double(per_img.ptr<uchar>(int(p1(1,0)+y))[int(p1(0,0)+x)]);
            p1_values.push_back(p1_v);
            p1_mean+=p1_v;

            p2_v=double(curr_img.ptr<uchar>(int(p2(1,0)+y))[int(p2(0,0)+x)]);
            p2_values.push_back(p2_v);
            p2_mean+=p2_v;
        }
    }
    p1_mean/=ncc_area;
    p2_mean/=ncc_area;
    double numerator=0.0;
    double deominator1=0.0;
    double deominator2=0.0;
    for(int i=0;i<p1_values.size();i++){
        numerator += (p1_values[i]-p1_mean)*(p2_values[i]-p2_mean);
        deominator1 += (p1_values[i]-p1_mean)*(p1_values[i]-p1_mean);
        deominator2 += (p2_values[i]-p2_mean)*(p2_values[i]-p2_mean);
    }
    return numerator / sqrt(deominator1*deominator2 + 1e-10);//防止分母等于0

}
//三角化，深度滤波
void depthFilter(int per,int curr,Mat &per_img,Mat &curr_img,Matrix<double, 3, 3> Rm,Vector3d Tm,Vector2d p1,Vector2d p2,Vector2d epipolar_direction,bool islastimg,Matrix<double, 3, 3> per_Rm,Vector3d per_Tm){
    //三角化求深度
    // d_ref * f_ref = d_cur * ( R * f_cur ) + t
    // f2 = R_RC * f_cur 向世界坐标变换
    // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]
    Vector3d f_ref = px2cam(p1);
    f_ref.normalize();
    Vector3d f_curr = px2cam(p2);
    f_curr.normalize();
    Vector3d f2 = Rm.inverse()*f_curr;
    Vector2d b = Vector2d(f_ref.dot(-Tm),f2.dot(-Tm));//方程右侧结果
    Matrix2d A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);
    Vector2d s = A.inverse()*b;
    Vector3d Pcam_esti=(s[0]*f_ref+s[1]*f2-Tm)/2.0;
    double depth_esti = Pcam_esti.norm();
     //Pw_esti = depth_esti*f_ref;
    
    //计算方差（不确定性误差）
    double t_norm = Tm.norm();
    double alpha = acos(f_ref.dot(-Tm)/t_norm);
    Vector3d f_curr_error = px2cam(p2+epipolar_direction);
    f_curr_error.normalize();
    double beta_error = acos(f_curr_error.dot(Tm)/t_norm);
    double gamma = M_PI - alpha - beta_error;
    double d_error = t_norm*sin(beta_error)/sin(gamma);
    double d_cov = d_error - depth_esti;
    double d_cov2 = d_cov*d_cov;

    //opencv 方式计算深度和不确定性

    // vector<Point2f> pts_1, pts_2;
    // Mat pts_3d;
    // Mat T1 = (Mat_<float>(3, 4) <<
    // 1, 0, 0, 0,
    // 0, 1, 0, 0,
    // 0, 0, 1, 0);
    // Mat T2 = (Mat_<float>(3, 4) <<
    // Rm(0, 0), Rm(0, 1), Rm(0, 2), Tm(0, 0),
    // Rm(1, 0), Rm(1, 1), Rm(1, 2), Tm(1, 0),
    // Rm(2, 0), Rm(2, 1), Rm(2, 2), Tm(2, 0)
    // );
    // Vector3d f_ref = px2cam(p1);
    // Vector3d f_curr = px2cam(p2);
    // Vector3d f_curr_error = px2cam(p2+epipolar_direction);
    // Point2f f_p1,f_p2,f_p2_error;
    // f_p1.x=f_ref(0,0);
    // f_p1.y=f_ref(1,0);
    // f_p2.x=f_curr(0,0);
    // f_p2.y=f_curr(1,0);
    // f_p2_error.x=f_curr_error(0,0);
    // f_p2_error.y=f_curr_error(1,0);
    // pts_1.push_back(f_p1);
    // pts_1.push_back(f_p1);
    // pts_2.push_back(f_p2);
    // pts_2.push_back(f_p2_error);
    // //cout<<"调用opencv三角化函数"<<endl;
    // triangulatePoints(T1, T2, pts_1, pts_2, pts_3d);
    // //cout<<"调用opencv三角化函数结束"<<endl;
    // Vector3d Pcam_esti,Pcam_esti_error;
    // Pcam_esti(0,0) = pts_3d.at<double>(0,0);
    // Pcam_esti(1,0) = pts_3d.at<double>(1,0);
    // Pcam_esti(2,0) = pts_3d.at<double>(2,0);
    // double depth_esti = Pcam_esti.norm();
    // Pcam_esti_error(0,0) = pts_3d.at<double>(0,1);
    // Pcam_esti_error(1,0) = pts_3d.at<double>(1,1);
    // Pcam_esti_error(2,0) = pts_3d.at<double>(2,1);
    // double depth_error = Pcam_esti_error.norm();
    // double d_cov = depth_error - depth_esti;
    // double d_cov2 = d_cov*d_cov;

    if(depthimages[per][p1(1,0)][p1(0,0)].isreconstruction){
        //如果当前点已重建进行高斯融合
        double mu = depthimages[per][p1(1,0)][p1(0,0)].depth;
        double sigma2 = depthimages[per][p1(1,0)][p1(0,0)].depth_cov2;
        depth_esti = (d_cov2*mu + sigma2*depth_esti)/(sigma2+d_cov2);
        d_cov2 = sigma2*d_cov2 / (sigma2 + d_cov2);

    }

    Pcam_esti= depth_esti*f_ref;  //P在左图相机坐标系坐标
    Vector3d Pcam2_esti = Rm*Pcam_esti + Tm;//P在右图相机坐标系坐标
    if(debug){
        cout<<p1(0,0)<<","<<p1(1,0)<<"的:左图相机坐标系坐标 "<<Pcam_esti(0,0)<<","<<Pcam_esti(1,0)<<","<<Pcam_esti(2,0)<<endl;
    }
    depthimages[per][p1(1,0)][p1(0,0)].depth=depth_esti;
    depthimages[per][p1(1,0)][p1(0,0)].depth_cov2=d_cov2;
    depthimages[per][p1(1,0)][p1(0,0)].camPoisition=Pcam_esti;
    depthimages[per][p1(1,0)][p1(0,0)].isreconstruction=true;

    depthimages[curr][p2(1,0)][p2(0,0)].depth=depth_esti; //复制左图深度，用于下一次的深度融合
    depthimages[curr][p2(1,0)][p2(0,0)].depth_cov2=d_cov2;
    depthimages[curr][p2(1,0)][p2(0,0)].camPoisition=Pcam2_esti;
    depthimages[curr][p2(1,0)][p2(0,0)].isreconstruction=true;

    if(islastimg){//如果是最后一张图片，加入点云数据。
        Vector3d Pw_esti =per_Rm.inverse()*Pcam_esti-per_Tm;//乘左图位资变换到世界坐标。
        add2Pointcloud(Pw_esti,double(per_img.ptr<uchar>(int(p1(1,0)))[int(p1(0,0))])); 
    }
}

//立体匹配，匹配选定的2张图片
void stereoMatch(int per,int curr,Mat per_img,Mat curr_img,Mat R,Mat t,bool islastimg,Mat &per_R,Mat &per_t){
    //极线搜索，重建3d坐标
    Matrix<double, 3, 3> Rm,per_Rm;
    Vector3d Tm,per_Tm;
    cv2eigen(R,Rm);
    cv2eigen(t,Tm);
    //计算上一张图相对世界坐标系的位姿态
    cv2eigen(per_R,per_Rm);
    cv2eigen(per_t,per_Tm);
    per_Tm = Rm*per_Tm+Tm;
    per_Rm = Rm*per_Rm;
    eigen2cv(per_Rm,per_R);
    eigen2cv(per_Tm,per_t);
    //Vector2d e_point = cam2px(-Tm);
    //遍历图片像素
    for(int y=ncc_window_size;y<per_img.rows-ncc_window_size;y++){
        for(int x=ncc_window_size;x<per_img.cols-ncc_window_size;x++){
            // if(depthimages[per][y][x].ismatch){
            //     cout<<"是匹配点"<<endl;
            //     //计算p2fp1 是否=0，判断p2是否在极线
            //     Matrix<double, 3, 1> p1;
            //     Matrix<double, 3, 1> p2;
            //     Matrix<double, 3, 3> E;
            //     Matrix<double, 3, 3> Km;
            //     p1(0,0)=x;
            //     p1(1,0)=y;
            //     p1(2,0)=1;
            //     p2(0,0)=depthimages[per][y][x].nextImage_poisition.x;
            //     p2(1,0)=depthimages[per][y][x].nextImage_poisition.y;
            //     p2(2,0)=1;
            //     cv2eigen(essential_matrix,E);
            //     cv2eigen(K,Km);
            //     cout<<"p2.transpose()*Km.inverse().transpose()*E*Km.inverse()*p1  "<<p2.transpose()*Km.inverse().transpose()*E*Km.inverse()*p1<<endl;
            //     Mat img_match;
            //     drawMatches(images[per], keypoints_1, images[curr], keypoints_2, depthimages[per][y][x].thismatch, img_match);
            //     namedWindow("test matches", 0);
            //     resizeWindow("test matches", 640, 480);
            //     imshow("test matches", img_match);
            //     waitKey(0);
            // }

            //判断当前点是否已收敛或发散  
            if(depthimages[per][y][x].depth_cov2>max_cov){
                cout<<x<<","<<y<<"当前点深度已发散"<<endl;
                continue;
            }
            if(depthimages[per][y][x].depth_cov2<min_cov){
                cout<<x<<","<<y<<"当前点深度已收敛"<<endl;
                //将估计的世界坐标加入点云数据,per.camPoisition*位姿态
                Vector3d Pw_esti =per_Rm.inverse()*depthimages[per][y][x].camPoisition-per_Tm;//乘左图位资变换到世界坐标。
                add2Pointcloud(Pw_esti,double(per_img.ptr<uchar>(y)[x])); 
                continue;
            }
            //极线搜索
            //通过1图片oP向量的不同深度向2图片投影，确定极线方向。
            if(depthimages[per][y][x].isreconstruction){
                //确定最小深度
                min_depth = depthimages[per][y][x].depth-3*depthimages[per][y][x].depth_cov2;
                if(min_depth<0.1) min_depth=0.1;
            }else {
                min_depth=1;
            }
            Vector2d p1;          
            p1(0,0)=x;
            p1(1,0)=y;
            Vector3d P_p1=px2cam(p1);//图片1坐标系下OP向量
            P_p1.normalize();
            Vector2d p2_1 = cam2px(Rm*(P_p1 * min_depth)+Tm);//在图片2的2个投影
            Vector2d p2_2 = cam2px(Rm*(P_p1 * (min_depth+50))+Tm);
            
            
            Vector2d epipolar_direction = (p2_2 - p2_1);
            epipolar_direction.normalize();
            if(debug){
                cout<<"p2_1"<<p2_1(0,0)<<","<<p2_1(1,0)<<endl;
                cout<<"p2_2 "<<p2_2(0,0)<<","<<p2_2(1,0)<<endl;
                cout<<"极线方向"<<epipolar_direction(0.0)<<"++"<<epipolar_direction(1.0)<<endl; 
            }            
            
            //1，判断最小深度投影的极线是否在像素平面
            Vector2d startpoint;
            if(!epipolarIsinpx(p2_1,curr_img.cols,curr_img.rows,epipolar_direction,startpoint)){
                //cout<<p2_1(0.0)<<","<<p2_1(1,0)<<"当前投影点极线不在像素平面上，没有匹配点"<<endl;
                //如果当前点已重建深度，加入点云数据
                if(depthimages[per][y][x].isreconstruction){
                    Vector3d Pw_esti =per_Rm.inverse()*depthimages[per][y][x].camPoisition-per_Tm;//乘左图位资变换到世界坐标。
                    add2Pointcloud(Pw_esti,double(per_img.ptr<uchar>(y)[x])); 
                }
                continue;
            }
            if(debug){
                cout<<p2_1(0.0)<<","<<p2_1(1,0)<<"当前投影点极线在像素平面上"<<endl;
            }
            
            //3，沿极线从进入像素平面的点开始搜索。
            double search_length = 0.0;
            if(!depthimages[per][y][x].isreconstruction){
                search_length = sqrt(curr_img.cols*curr_img.cols+curr_img.rows*curr_img.rows);//a^2+b^2=c^2
            }else {
                search_length=6*depthimages[per][y][x].depth_cov2;
            }
            if(debug){
                cout<<"极线搜索长度为："<<search_length<<endl;
                cout<<"startpoint:"<<startpoint(0,0)<<" , "<<startpoint(1,0)<<endl;
            }
            
            //NCC匹配
            double bestncc = -1.0;
            Vector2d bestmatchpoint;
            for(int l=0;l<=search_length;l++){
                startpoint+=l*epipolar_direction;
                if(!isinpx(int(startpoint(0,0)),int(startpoint(1,0)),curr_img.cols,curr_img.rows)){
                    if(debug){
                        cout<<"失败点坐标:"<<startpoint(0,0)<<" , "<<startpoint(1,0)<<endl;
                        cout<<"失败 l :"<<l<<endl;
                    }
                    
                    break;
                }
                double ncc = NCC(per_img,curr_img,p1,startpoint);
                if(debug){
                    cout<<"ncc:"<<ncc<<endl;
                }
                
                if(ncc>bestncc){
                    bestncc=ncc;
                    bestmatchpoint=startpoint;
                }
            }
            if(bestncc<0.85f){
                //cout<<x<<","<<y<<" ncc匹配失败,无匹配点"<<endl;
                //如果当前点已重建深度，加入点云数据
                if(depthimages[per][y][x].isreconstruction){
                    Vector3d Pw_esti =per_Rm.inverse()*depthimages[per][y][x].camPoisition-per_Tm;//乘左图位资变换到世界坐标。
                    add2Pointcloud(Pw_esti,double(per_img.ptr<uchar>(y)[x])); 
                }
                continue;
            }
            //记录匹配点
            bestmatchpoint(0,0)=int(bestmatchpoint(0,0));
            bestmatchpoint(1,0)=int(bestmatchpoint(1,0));
            depthimages[per][y][x].nextImage_poisition.x=bestmatchpoint(0,0);
            depthimages[per][y][x].nextImage_poisition.y=bestmatchpoint(1,0);
            depthimages[per][y][x].ismatch=true;
            //深度估计：第一张图进行深度初始化，其余图片进行高斯融合
            depthFilter(per,curr,per_img,curr_img,Rm,Tm,p1,bestmatchpoint,epipolar_direction,islastimg,per_Rm,per_Tm);

        }
        cout<<y<<endl;
        //waitKey(0);
    }
}


int main(int argc, char** argv){
    //读取图片路径信息
    vector<Mat> images; 
    vector<vector<KeyPoint>> keypoints;   // 存储所有图片关键点
    vector<Mat> descriptors;               //存储所以图片关键点的描述子
    for ( int i=0; i<imagenums; i++ )
    {
        string path = image_path+to_string(i+1)+".png";
        //string path = image_path+to_string(i+1)+".jpg";
        images.push_back( imread(path,CV_LOAD_IMAGE_COLOR) );
    }
    cout<<"图片读取完毕"<<endl;
    //
    Ptr<FeatureDetector> detector = SIFT::create();
    Ptr<DescriptorExtractor> descriptor = SIFT::create();
    Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
    cout<<"matcher"<<endl;
    //初始化
    vector<KeyPoint> keypoints_0;
    Mat descriptors_0;
    detector->detect(images[0],keypoints_0);
    descriptor->compute(images[0], keypoints_0, descriptors_0);
    keypoints.push_back(keypoints_0);
    descriptors.push_back(descriptors_0);
    depthimages.push_back(vector<vector<dpoint>>(images[0].rows,vector<dpoint>(images[0].cols)));//存储第一张图片的世界坐标，匹配点信息
    Mat per_R  = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    Mat per_t  = (Mat_<double>(3,1)<< 0,0,0); //上一张图片相对于世界坐标系的位姿
    for(int i = 1;i<imagenums;i++){
        bool islastimg = false;
        int curr=i;
        int per=i-1;
        vector<KeyPoint> keypoints_1, keypoints_2;//SIFT关键点
        Mat descriptors_1, descriptors_2;         //描述点
        if(i == imagenums){
            //最后一张图与第一张图比较
            islastimg = true;
            curr=0;
            per=imagenums-1;
            keypoints_1=keypoints[per];
            keypoints_2=keypoints[0];
            descriptors_1=descriptors[per];
            descriptors_2=descriptors[0];
        }else{
            vector<vector<dpoint>>  dimg2(images[curr].rows,vector<dpoint>(images[curr].cols));//
            depthimages.push_back(dimg2);
            keypoints_1=keypoints[per];
            descriptors_1=descriptors[per];
            detector->detect(images[curr], keypoints_2);
            descriptor->compute(images[curr], keypoints_2, descriptors_2);

            keypoints.push_back(keypoints_2);
            descriptors.push_back(descriptors_2);
        }
        
        //计算匹配点
        cout<<"计算匹配点"<<endl;
        vector<vector<DMatch>>  matches;                   //匹配点列表
        matcher->knnMatch(descriptors_1, descriptors_2, matches,2);
        cout<<"完成计算匹配点"<<endl;
        // Mat outimg1,outimg2;
        // drawKeypoints(images[per], keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // drawKeypoints(images[curr], keypoints_2, outimg2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // namedWindow("ORB features1", 0);
        // resizeWindow("ORB features1", 640, 480);
        // imshow("ORB features1", outimg1);
        // namedWindow("ORB features2", 0);
        // resizeWindow("ORB features2", 640, 480);
        // imshow("ORB features2", outimg2);
        //剔除错误匹配。因为match会将最接近的点记录，在这里剔除匹配出的2个结果差距过小的点
        std::vector<DMatch> good_matches;
        //std::vector<DMatch> p3p_matches;
        vector<Point2f> points1;
        vector<Point2f> points2;//匹配点坐标
        vector<Point3f> p3p_wpoints;
        vector<Point2f> p3p_points;
        for (int i = 0; i < descriptors_1.rows; i++) {
            if (matches[i][0].distance < 0.6*matches[i][1].distance) {
                good_matches.push_back(matches[i][0]);
                //记录匹配点坐标，用计算位姿
                points1.push_back(keypoints_1[matches[i][0].queryIdx].pt);
                points2.push_back(keypoints_2[matches[i][0].trainIdx].pt);

                //记录筛选后匹配点
                depthimages[per][keypoints[per][matches[i][0].queryIdx].pt.y][keypoints[per][matches[i][0].queryIdx].pt.x].nextImage_poisition=keypoints[per][matches[i][0].queryIdx].pt;
                depthimages[per][keypoints[per][matches[i][0].queryIdx].pt.y][keypoints[per][matches[i][0].queryIdx].pt.x].ismatch = true;

                depthimages[per][keypoints[per][matches[i][0].queryIdx].pt.y][keypoints[per][matches[i][0].queryIdx].pt.x].thismatch.push_back(matches[i][0]); 

                //记录已确认世界坐标点的匹配点
                if(p3p_wpoints.size()<4&&depthimages[per][keypoints[per][matches[i][0].queryIdx].pt.y][keypoints[per][matches[i][0].queryIdx].pt.x].isreconstruction){
                    //p3p_matches.push_back(matches[i][0]);
                    Point3f wposition(
                       depthimages[per][keypoints[per][matches[i][0].queryIdx].pt.y][keypoints[per][matches[i][0].queryIdx].pt.x].camPoisition(0,0),
                       depthimages[per][keypoints[per][matches[i][0].queryIdx].pt.y][keypoints[per][matches[i][0].queryIdx].pt.x].camPoisition(1,0),
                       depthimages[per][keypoints[per][matches[i][0].queryIdx].pt.y][keypoints[per][matches[i][0].queryIdx].pt.x].camPoisition(2,0)
                    );
                    //存储p3p需要的已重建的世界坐标
                    p3p_wpoints.push_back(wposition);
                    p3p_points.push_back(keypoints_2[matches[i][0].trainIdx].pt);
                }
            }
        }
        Mat img_goodmatch;
        cout<<"绘制匹配点图像"<<endl;
        drawMatches(images[per], keypoints_1, images[curr], keypoints_2, good_matches, img_goodmatch);
        namedWindow("good matches", 0);
        resizeWindow("good matches", 640, 480);
        imshow("good matches", img_goodmatch);
        //waitKey(0);

        //计算位姿
        //通过筛选匹配点时记录的已确认世界坐标匹配点数目，决策是否使用P3P
        Mat essential_matrix;
        Mat R , t;  //相对于上一张图片的位姿态
        if(p3p_wpoints.size()>=4){
            cout<<"使用P3P计算位姿"<<endl;
            cout<<"p3p_wpoints.size()"<<p3p_wpoints.size()<<endl;
            //CV_P3P = 2
            Mat r;
            solvePnP(p3p_wpoints,p3p_points,K,Mat(),r,t,false,2);
            Rodrigues(r,R);//旋转向量2旋转矩阵
        }else if(good_matches.size()<8){
            //计算本质矩阵
            cout<<"匹配点过少，无法计算位姿"<<endl;
            showPointCloud(pointcloud);
            waitKey(0);
            cout<<"结束";
            return 0;  
        }else{
            //-- 计算本质矩阵
            essential_matrix = findEssentialMat(points1, points2,K,RANSAC,0.99,1);
            cout<<"本质矩阵："<<essential_matrix<<endl;
            recoverPose(essential_matrix, points1, points2, K, R,t);
        }

        cout<<t<<endl;   
        //立体匹配，深度估计
        stereoMatch(per,curr,images[per],images[curr],R,t,islastimg,per_R,per_t);
    }
    showPointCloud(pointcloud);
    waitKey(0);
    cout<<"结束";
    return 1;
}