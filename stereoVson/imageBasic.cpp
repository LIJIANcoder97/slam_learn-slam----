#include<iostream>
#include<chrono>

using namespace std;

#include<opencv4/opencv2/core/core.hpp>
#include<opencv4/opencv2/highgui/highgui.hpp>

int main(int argc , char **argv){
    //
    cv::Mat image;
    image = cv::imread(argv[1]); //cv::imread 读取指定路径图像
    //
    if(image.data == nullptr){//数据不存在
        cerr<<"file "<<argv[1]<<" not exit. "<<endl;
        return 0;
    }

    //
    cout<<"  "<<image.cols<<", height "<<image.rows<<", channels "<<image.channels()<<endl;
    cv::imshow("image",image);
    cv::waitKey(0);             //暂停程序

    //判断image类型
    if(image.type() != CV_8UC1 && image.type() != CV_8UC3){
        cout<<"请输入彩色图或灰色图"<<endl;
        return 0;
    }

    //遍历图像
    //计时 std::chrono
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for(size_t y = 0;y < image.rows;y++){
        //cv::Mat::ptr 获取行指针
        unsigned char *row_ptr = image.ptr<unsigned char>(y);//row_ptr是第y行头指针
        for(size_t x = 0;x < image.cols;x++){
            unsigned char *data_ptr = &row_ptr[x*image.channels()];//指向待访问的像素数据
            //访问像素每个通道
            for(int c = 0;c != image.channels();c++){
                unsigned char data = data_ptr[c];
            }
        }
    }

    
}