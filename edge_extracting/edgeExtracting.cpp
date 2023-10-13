#include<iostream>

using namespace std;

#include<opencv4/opencv2/core/core.hpp>
#include<opencv4/opencv2/highgui/highgui.hpp>

cv::Mat edgeExtract_byArray(cv::Mat image_gray){
    cv::Mat image_edge = image_gray.clone();

    //使用循环方式提取边缘
    //遍历图
    //行
    for(size_t y = 0;y < image_gray.rows;y++){
        //cv::Mat::ptr 获取行指针
        unsigned char *row_ptr = image_gray.ptr<unsigned char>(y);//row_ptr是第y行头指针
        for(size_t x = 0;x < image_gray.cols;x++){
            float sum = 0;
            //左边界
            if(x>0)sum += image_gray.at<unsigned char>(y,x-1);
            //右边界
            if(x<image_gray.cols-1) sum += image_gray.at<unsigned char>(y,x+1);
            sum += image_gray.at<unsigned char>(y,x);
            image_edge.at<unsigned char>(y,x)=sum/3.0;
        }
    }
    //列
    cv::Mat edge_tem = image_edge.clone();
    for(size_t y = 0;y < image_edge.rows;y++){
        //cv::Mat::ptr 获取行指针
        unsigned char *row_ptr = image_edge.ptr<unsigned char>(y);//row_ptr是第y行头指针
        for(size_t x = 0;x < image_edge.cols;x++){
            float sum = 0;
            //上边界
            if(y>0)sum += image_edge.at<unsigned char>(y-1,x);
            //下边界
            if(y<image_edge.rows-1) sum += image_edge.at<unsigned char>(y+1,x);
            sum += image_edge.at<unsigned char>(y,x);
            edge_tem.at<unsigned char>(y,x)=sum/3.0;
        }
    }
    //原图-平滑去噪 = 边缘图
    image_edge = (image_gray - edge_tem)*2;
    return image_edge;

}
int main(int argc , char **argv){
    cv::Mat image = cv::imread(argv[1]); //cv::imread 读取指定路径图像

    if(image.data == nullptr){//数据不存在
        cerr<<"file "<<argv[1]<<" not exit. "<<endl;
        return 0;
    }

    cout<<"  "<<image.cols<<", height "<<image.rows<<", channels "<<image.channels()<<endl;
    cv::imshow("image",image);
    cv::waitKey(0); 
    
    vector<cv::Mat> channels;
    split(image,channels); 

    cv::Mat blue,green,red;
    blue = edgeExtract_byArray(channels.at(0));
    cv::imshow("image_gray",blue);
    cv::waitKey(0);

    green = edgeExtract_byArray(channels.at(1));
    red = edgeExtract_byArray(channels.at(2));

    vector<cv::Mat> channels2;
    channels2.push_back(blue);
    channels2.push_back(green);
    channels2.push_back(red);
    merge(channels2,image);

    cv::imshow("image_eage",image);
    cv::waitKey(0);
return 1;
}