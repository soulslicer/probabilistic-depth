#ifndef MAT_EXT_H
#define MAT_EXT_H

// C++
#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include <map>
#include <ostream>

// ROS
#ifdef ROS_FOUND
#include <ros/ros.h>
#include <ros/package.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <cv_bridge/cv_bridge.h>
#endif

// Eigen
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/StdVector>
#include <Eigen/Dense>

// Opencv
#include <opencv2/opencv.hpp>

// Opencv GPU
#ifdef CUDA_FOUND
#include "opencv2/opencv_modules.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#endif

class MatExt : public cv::Mat
{

public:
    //boost::posix_time::ptime time;
    std::string name;

    MatExt();
    ~MatExt();
    MatExt(const cv::Mat& inputImage);
    #ifdef ROS_FOUND
    MatExt(const sensor_msgs::CompressedImageConstPtr& inputImage);
    MatExt(const sensor_msgs::ImageConstPtr& inputImage);
    MatExt(const sensor_msgs::Image& inputImage);
    #endif
    MatExt clone();
    bool empty();

    void performInvert();
    void performTranspose();
    void performScale(double scale);
    void performResize(double scale);

    MatExt extractInvert();
    MatExt extractTranspose();
    MatExt extractResize(double scale);

    void performSobel(int kernel, double scale, double delta);

    bool performGrayConvert();
    bool performColorConvert();
    MatExt extractGrayConvert();
    MatExt extractColorConvert();

    std::string getType();
    #ifdef ROS_FOUND
    std::string getROSType();
    sensor_msgs::Image getROSMsg();
    #endif
};

#endif
