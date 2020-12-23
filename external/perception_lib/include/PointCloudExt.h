#ifndef POINT_CLOUD_EXT_H
#define POINT_CLOUD_EXT_H

// C++
#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include <map>

// ROS
#ifdef ROS_FOUND
#include <ros/ros.h>
#include <ros/package.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#endif

// PCL Conversions
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/pcl_config.h>
#include <pcl/point_types_conversion.h>
#include <pcl/console/parse.h>

// Filters/Features/Algorithms
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/surface/mls.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/sample_consensus/sac_model_normal_plane.h>
#include <pcl/filters/passthrough.h>
#include <pcl/surface/convex_hull.h>

// Viz
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/visualization/point_cloud_geometry_handlers.h>

// Eigen
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/StdVector>
#include <Eigen/Dense>

// Opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <pcl/segmentation/supervoxel_clustering.h>
#include "MatExt.h"

extern "C" Eigen::Matrix4f getTransformMatrix(float roll, float pitch, float yaw, float x, float y, float z);
extern "C" pcl::PointCloud<pcl::PointXYZ> getBoxCorners(float x, float y, float z, float xl, float yl, float zl, float roll, float pitch, float yaw);

class PointCloudExtParams
{
public:
    struct OBB{
        Eigen::Vector3f bboxTransform;
        Eigen::Quaternionf bboxQuaternion;
        float xDist;
        float yDist;
        float zDist;
        pcl::PointCloud<pcl::PointXYZ> points;
    };

    struct PointDistance{
        int index;
        float distance;
        Eigen::Vector3f direction;
        PointDistance(int i, float d) {index=i; distance=d;}
        PointDistance(int i, pcl::PointXYZ a, pcl::PointXYZ b) {
            index=i;
            direction(0) = a.x-b.x;
            direction(1) = a.y-b.y;
            direction(2) = a.z-b.z;
        }
        static bool distanceSort(PointCloudExtParams::PointDistance i, PointCloudExtParams::PointDistance j){
            return (i.distance < j.distance);
        }
        static bool directionSort(PointCloudExtParams::PointDistance i, PointCloudExtParams::PointDistance j){
            return (std::fabs(i.direction(2)) < std::fabs(j.direction(2)));
        }
    };

    enum Mode{
      EXTRACT_NORMALS_RADIUSSEARCH,
      EXTRACT_NORMALS_KSEARCH
    };
};

template<typename PointT>
class PointCloudExt : public pcl::PointCloud<PointT>
{
    typedef typename pcl::search::KdTree<PointT>::Ptr KdTreePtr;
    typedef typename pcl::PointCloud<PointT>::Ptr PointCloudExtPtr;
    typedef typename pcl::PointCloud<PointT>::iterator Iterator;

public:

    int visualizerSize;
    //boost::posix_time::ptime time;

    /**************************************************************************************
     * Constructors/ROS Functions
    **************************************************************************************/

    PointCloudExt();
    ~PointCloudExt();
    #ifdef ROS_FOUND
    explicit PointCloudExt(const sensor_msgs::PointCloud2ConstPtr& inputCloud);
    explicit PointCloudExt(const sensor_msgs::PointCloud2& inputCloud);
    #endif
    explicit PointCloudExt(const pcl::PointCloud<PointT>& inputCloud);
    PointCloudExt<PointT> clone();
    bool empty();
    #ifdef ROS_FOUND
    sensor_msgs::PointCloud2 getROSMsg();
    #endif

    /**************************************************************************************
     * Transformations
    **************************************************************************************/

    void performTransform(Eigen::Matrix4f transformMatrix);
    void performTransform(MatExt transformMatrix);
    void performScale(float xScale, float yScale, float zScale);

    PointCloudExt<PointT> extractTransform(Eigen::Matrix4f transformMatrix);
    PointCloudExt<PointT> extractTransform(MatExt transformMatrix);
    PointCloudExt<PointT> extractScale(float xScale, float yScale, float zScale);

    /**************************************************************************************
     * Filters
    **************************************************************************************/

    void performCurvatureCompute(float searchRadius);
    void performRangeThresholding(std::string axis, float startRange, float endRange, bool negative = false);
    void performDownsampling(float leafSize);
    void performDownsamplingKeepingNormals(float leafSize);
    void performSmoothing(float searchRadius, int threadCount);
    pcl::ModelCoefficients::Ptr performModelFitting(int model, float error);

    /**************************************************************************************
     * Extract Data
    **************************************************************************************/

    pcl::PointXYZ extractCentroid();
    Eigen::Vector3f extractNormalCentroid();
    Eigen::Vector3f extractColorCentroid();
    PointCloudExtParams::OBB extractMVBB(bool extrude, bool collapse = false, float zVal = 0.0);
    PointCloudExt<PointT> extractEuclidianClusterCloud(float tolerance, int minClusterSize, int maxClusterSize);
    std::vector<PointCloudExt<PointT> > extractEuclidianClusters(float tolerance, int minClusterSize, int maxClusterSize);
    PointCloudExt<pcl::PointXYZRGBNormal> extractNormals(float searchRadius, PointCloudExtParams::Mode searchMode);
    PointCloudExt<pcl::PointXYZRGBNormal> extractNormals(float searchRadius, PointCloudExtParams::Mode searchMode, int threadCount);
    std::vector<PointCloudExt<PointT> > extractModelFittedClusters(int model, float error, float eucDistance, int minPoints);
    PointT extractClosestPoint(PointT point);
    float extractTotalDistanceMean();
    PointCloudExt<pcl::PointXYZRGBNormal> extractSmoothing(float searchRadius, int threadCount);

    /**************************************************************************************
     * Space Conversion
    **************************************************************************************/

    void performPolarConversion();
    void performCartesianConversion();
    void swapNormalsAndXYZ();

    /**************************************************************************************
     * Color
    **************************************************************************************/

    void colorize(int r, int g, int b);
    void colorize();

    /**************************************************************************************
     * Pixel Operations
    **************************************************************************************/

    std::vector<cv::Point2i> projectPoints(MatExt worldToCamera, MatExt rgbCameraMatrix);
    void projectPoints(MatExt worldToCamera, MatExt rgbCameraMatrix, MatExt& image, bool cloudColor, bool circleFill, cv::Scalar fillColor = cv::Scalar(255,255,0));

    /**************************************************************************************
     * Other Operations
    **************************************************************************************/

    void removeNANPoints();

    PointCloudExtPtr getPtrCopy(){
        boost::shared_ptr<pcl::PointCloud<PointT> > aCopy = boost::make_shared<pcl::PointCloud<PointT> >(*this);
        return aCopy;
    }

    static bool clusterSizeSort(PointCloudExt<PointT> i, PointCloudExt<PointT> j){
        return (i.size()>j.size());
    }

    static bool pointZSort(PointT i, PointT j){
        return (i.z>j.z);
    }

    void swap(float& a, float& b){
        float temp = b;
        b = a;
        a = temp;
    }

    static float l2norm(PointT a, PointT b){
        return sqrt(pow((a.x - b.x),2) + pow((a.y - b.y),2) + pow((a.z - b.z),2));
    }

    static PointT transform(PointT& point, MatExt transformMatrix);
    static PointT transform(PointT& point, Eigen::Matrix4f transformMatrix);
    static PointT backproject(cv::Point2i& pixel, uint16_t depth, MatExt cameraMatrix);
    static cv::Point2i project(PointT& point, MatExt cameraMatrix);

};

#endif
