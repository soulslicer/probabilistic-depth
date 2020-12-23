#ifndef VISUALIZER_H
#define VISUALIZER_H

// C++
#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include <csignal>

// ROS
#ifdef ROS_FOUND
#include <ros/ros.h>
#include <ros/package.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>
#endif

// Viz
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/visualization/point_cloud_geometry_handlers.h>
#include <vtkRenderWindow.h>
#include <vtkWindowToImageFilter.h>

#include "PointCloudExt.h"
#include "MatExt.h"

#include <vtkPolyLine.h>

class VisualizerExtBuffer{
public:
    struct ArrowData{
        pcl::PointXYZ p1;
        pcl::PointXYZ p2;
        int r, g, b;
        bool head;
        ArrowData(pcl::PointXYZ p1, pcl::PointXYZ p2, int r, int g, int b, bool head){
            this->p1 = p1; this->p2 = p2;
            this->r = r; this->g = g; this->b = b;
            this->head = head;
        }
    };

    struct LineData{
        pcl::PointXYZ p1;
        pcl::PointXYZ p2;
        int r, g, b;
        LineData(pcl::PointXYZ p1, pcl::PointXYZ p2, int r, int g, int b){
            this->p1 = p1; this->p2 = p2;
            this->r = r; this->g = g; this->b = b;
        }
    };

    std::vector< std::vector<PointCloudExt<pcl::PointXYZ> > > bufferXYZ;
    std::vector< std::vector<PointCloudExt<pcl::PointXYZRGB> > > bufferXYZRGB;
    std::vector< std::vector<PointCloudExt<pcl::PointXYZRGBNormal> > > bufferXYZRGBNormal;
    std::vector< std::vector<PointCloudExt<pcl::PointXYZRGBNormal> > > bufferNormals;
    std::vector< std::vector<PointCloudExtParams::OBB> > bufferOBB;
    std::vector< std::vector<std::string> > bufferString;
    std::vector< std::vector<pcl::PointXYZ> > bufferStringPos;
    std::vector< std::vector<MatExt> > bufferMat;
    std::vector< std::vector<VisualizerExtBuffer::ArrowData> > bufferArrow;
    std::vector< std::vector<VisualizerExtBuffer::LineData> > bufferLine;
    std::vector< std::vector<pcl::PolygonMesh> > bufferMesh;
    std::vector< std::vector<pcl::TextureMesh> > bufferTextureMesh;
    boost::mutex bufferMutex;

    VisualizerExtBuffer();
    void swapBuffer();
    void updateViewer(boost::shared_ptr<pcl::visualization::PCLVisualizer>& viewer);
};

class VisualizerExt
{
public:
    static boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    static VisualizerExtBuffer buffer;
    static boost::mutex bufferMutex;
    static bool stopFlag;

    static bool mouseCallbackMode, keyboardCallbackMode;
    static boost::function<void(int, int, int)> mouseCallbackExt;
    static boost::function<void(int)> keyboardCallbackExt;
    static void setMouseCallback(boost::function<void(int, int, int)>& callback);
    static void setKeyboardCallback(boost::function<void (int)> &callback);
    static void mouseCallback(int event, int x, int y, int flags, void* param);
    static void keyboardCallback(int kb);

    static void loop();
    static void start();
    static void stop();
    static void saveScreenshot(std::string file);
    static void addPointCloud(PointCloudExt<pcl::PointXYZ> cloud, int size);
    static void addPointCloud(PointCloudExt<pcl::PointXYZRGB> cloud, int size);
    static void addPointCloud(PointCloudExt<pcl::PointXYZRGBNormal> cloud, int size);
    static void addPoint(pcl::PointXYZ point, int size);
    static void addPoint(pcl::PointXYZRGB point, int size);
    static void addPoint(pcl::PointXYZRGBNormal point, int size);
    static void addPoint(float x, float y, float z, int r, int g, int b, int size);
    static void addNormals(PointCloudExt<pcl::PointXYZRGBNormal> cloud);
    static void addText(std::string text, pcl::PointXYZ position);
    static void addOBB(PointCloudExtParams::OBB obb);
    static void addArrow(VisualizerExtBuffer::ArrowData arrow);
    static void addLine(VisualizerExtBuffer::LineData line);
    static void addMesh(pcl::PolygonMesh mesh);
    static void addTextureMesh(pcl::TextureMesh mesh);
    static cv::Mat getRenderedImage();

    static void swapBuffer();
    static void addImage(MatExt img, float resize);
    static void addImage(MatExt img, float resize, std::string name);

    static bool enabled;
    static int count;

    static void signalHandler(int signum) {
        cout << "Interrupt signal (" << signum << ") received.\n";
        if(signum == 2) {
            VisualizerExt::stopFlag = true;
            exit(-1);
        }
    }
};

#endif
