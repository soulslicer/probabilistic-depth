#include "PointCloudExt.h"

Eigen::Matrix4f getTransformMatrix(float roll, float pitch, float yaw, float x, float y, float z){
    Eigen::AngleAxisf rollAngle(roll / 180.0 * M_PI, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(pitch / 180.0 * M_PI, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle(yaw / 180.0 * M_PI, Eigen::Vector3f::UnitZ());
    Eigen::Quaternion<float> q = yawAngle * pitchAngle * rollAngle;

    Eigen::Matrix3f rotationMatrix = q.matrix();
    Eigen::Matrix4f transformMatrix = Eigen::Matrix4f::Identity();
    transformMatrix(0,0) = rotationMatrix(0,0);
    transformMatrix(0,1) = rotationMatrix(0,1);
    transformMatrix(0,2) = rotationMatrix(0,2);
    transformMatrix(0,3) = x;
    transformMatrix(1,0) = rotationMatrix(1,0);
    transformMatrix(1,1) = rotationMatrix(1,1);
    transformMatrix(1,2) = rotationMatrix(1,2);
    transformMatrix(1,3) = y;
    transformMatrix(2,0) = rotationMatrix(2,0);
    transformMatrix(2,1) = rotationMatrix(2,1);
    transformMatrix(2,2) = rotationMatrix(2,2);
    transformMatrix(2,3) = z;
    transformMatrix(3,0) = 0;
    transformMatrix(3,1) = 0;
    transformMatrix(3,2) = 0;
    transformMatrix(3,3) = 1;
    return transformMatrix;
}

pcl::PointCloud<pcl::PointXYZ> getBoxCorners(float x, float y, float z, float xl, float yl, float zl, float roll, float pitch, float yaw){
    PointCloudExt<pcl::PointXYZ> boxCloud;
    boxCloud.push_back(pcl::PointXYZ(0 + xl/2, 0 + yl/2, 0 + zl/2));
    boxCloud.push_back(pcl::PointXYZ(0 - xl/2, 0 - yl/2, 0 + zl/2));
    boxCloud.push_back(pcl::PointXYZ(0 + xl/2, 0 - yl/2, 0 + zl/2));
    boxCloud.push_back(pcl::PointXYZ(0 - xl/2, 0 + yl/2, 0 + zl/2));
    boxCloud.push_back(pcl::PointXYZ(0 + xl/2, 0 + yl/2, 0 - zl/2));
    boxCloud.push_back(pcl::PointXYZ(0 - xl/2, 0 - yl/2, 0 - zl/2));
    boxCloud.push_back(pcl::PointXYZ(0 + xl/2, 0 - yl/2, 0 - zl/2));
    boxCloud.push_back(pcl::PointXYZ(0 - xl/2, 0 + yl/2, 0 - zl/2));
    pcl::transformPointCloud(boxCloud, boxCloud, getTransformMatrix(roll,pitch,yaw,x,y,z));
    return boxCloud;
}

/**************************************************************************************
 * Constructors/ROS Functions
**************************************************************************************/

template<class PointT> PointCloudExt<PointT>::PointCloudExt(){
    //kdTreeFLANN = KdTreeFLANNPtr(new pcl::KdTreeFLANN<PointT>);
}

#ifdef ROS_FOUND
template<class PointT> PointCloudExt<PointT>::PointCloudExt(const sensor_msgs::PointCloud2ConstPtr& inputCloud)
{
    pcl::fromROSMsg(*inputCloud, *this);
}

template<class PointT> PointCloudExt<PointT>::PointCloudExt(const sensor_msgs::PointCloud2& inputCloud)
{
    pcl::fromROSMsg(inputCloud, *this);
}
#endif

template<class PointT> PointCloudExt<PointT>::PointCloudExt(const pcl::PointCloud<PointT>& inputCloud)
{
    pcl::copyPointCloud(inputCloud,*this);
}

template<class PointT> PointCloudExt<PointT>::~PointCloudExt(){

}

template<class PointT> PointCloudExt<PointT> PointCloudExt<PointT>::clone(){
    PointCloudExt<PointT> cloudCopy;
    pcl::copyPointCloud(*this, cloudCopy);
    //cloudCopy.time = this->time;
    cloudCopy.visualizerSize = this->visualizerSize;
    return cloudCopy;
}

template<class PointT> bool PointCloudExt<PointT>::empty(){
    if(this->points.size() == 0) return true;
    return false;
}

#ifdef ROS_FOUND
template<class PointT> sensor_msgs::PointCloud2 PointCloudExt<PointT>::getROSMsg(){
    sensor_msgs::PointCloud2 outputCloud;
    pcl::toROSMsg(*this,outputCloud);
    return outputCloud;
}
#endif

/**************************************************************************************
 * Transformations
**************************************************************************************/

void cv2eigen(cv::Mat& mat, Eigen::Matrix4f& eigen){
    eigen(0,0) = mat.at<float>(0,0);
    eigen(0,1) = mat.at<float>(0,1);
    eigen(0,2) = mat.at<float>(0,2);
    eigen(0,3) = mat.at<float>(0,3);
    eigen(1,0) = mat.at<float>(1,0);
    eigen(1,1) = mat.at<float>(1,1);
    eigen(1,2) = mat.at<float>(1,2);
    eigen(1,3) = mat.at<float>(1,3);
    eigen(2,0) = mat.at<float>(2,0);
    eigen(2,1) = mat.at<float>(2,1);
    eigen(2,2) = mat.at<float>(2,2);
    eigen(2,3) = mat.at<float>(2,3);
    eigen(3,0) = mat.at<float>(3,0);
    eigen(3,1) = mat.at<float>(3,1);
    eigen(3,2) = mat.at<float>(3,2);
    eigen(3,3) = mat.at<float>(3,3);
}

template<class PointT> void PointCloudExt<PointT>::performTransform(Eigen::Matrix4f transformMatrix){
    pcl::transformPointCloud(*this, *this, transformMatrix);
}

template<> void PointCloudExt<pcl::PointXYZRGBNormal>::performTransform(Eigen::Matrix4f transformMatrix){
    pcl::transformPointCloudWithNormals(*this, *this, transformMatrix);
}

template<class PointT> void PointCloudExt<PointT>::performTransform(MatExt transformMatrix){
    Eigen::Matrix4f transformMatrixEigen;
    cv2eigen(transformMatrix, transformMatrixEigen);
    pcl::transformPointCloud(*this, *this, transformMatrixEigen);
}

template<> void PointCloudExt<pcl::PointXYZRGBNormal>::performTransform(MatExt transformMatrix){
    Eigen::Matrix4f transformMatrixEigen;
    cv2eigen(transformMatrix, transformMatrixEigen);
    pcl::transformPointCloudWithNormals(*this, *this, transformMatrixEigen);
}

template<class PointT> void PointCloudExt<PointT>::performScale(float xScale, float yScale, float zScale){
    for(Iterator it = this->begin(); it!= this->end(); it++){
        it->x *= xScale;
        it->y *= yScale;
        it->z *= zScale;
    }
}

template<class PointT> PointCloudExt<PointT> PointCloudExt<PointT>::extractTransform(Eigen::Matrix4f transformMatrix){
    PointCloudExt<PointT> output;
    pcl::transformPointCloud(*this, output, transformMatrix);
    return output;
}

template<> PointCloudExt<pcl::PointXYZRGBNormal> PointCloudExt<pcl::PointXYZRGBNormal>::extractTransform(Eigen::Matrix4f transformMatrix){
    PointCloudExt<pcl::PointXYZRGBNormal> output;
    pcl::transformPointCloudWithNormals(*this, output, transformMatrix);
    return output;
}

template<class PointT> PointCloudExt<PointT> PointCloudExt<PointT>::extractTransform(MatExt transformMatrix){
    PointCloudExt<PointT> output;
    Eigen::Matrix4f transformMatrixEigen;
    cv2eigen(transformMatrix, transformMatrixEigen);
    pcl::transformPointCloud(*this, output, transformMatrixEigen);
    return output;
}

template<> PointCloudExt<pcl::PointXYZRGBNormal> PointCloudExt<pcl::PointXYZRGBNormal>::extractTransform(MatExt transformMatrix){
    PointCloudExt<pcl::PointXYZRGBNormal> output;
    Eigen::Matrix4f transformMatrixEigen;
    cv2eigen(transformMatrix, transformMatrixEigen);
    pcl::transformPointCloudWithNormals(*this, output, transformMatrixEigen);
    return output;
}

template<class PointT> PointCloudExt<PointT> PointCloudExt<PointT>::extractScale(float xScale, float yScale, float zScale){
    PointCloudExt<PointT> output = this->clone();
    for(Iterator it = output.begin(); it!= output.end(); it++){
        it->x *= xScale;
        it->y *= yScale;
        it->z *= zScale;
    }
    return output;
}

/**************************************************************************************
 * Filters
**************************************************************************************/

template<> void PointCloudExt<pcl::PointXYZRGBNormal>::performCurvatureCompute(float searchRadius){
    PointCloudExt<pcl::PointXYZRGBNormal>& objectCloud = *this;
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::PrincipalCurvatures> pc;
    pcl::PointCloud<pcl::PrincipalCurvatures> cloud_c;
    pc.setInputCloud(objectCloud.getPtrCopy());
    pc.setInputNormals(objectCloud.getPtrCopy());
    pc.setSearchMethod(tree);
    pc.setRadiusSearch(searchRadius);
    pc.compute(cloud_c);
    for (int i=0;i<objectCloud.size();i++){
        objectCloud[i].curvature = 0.5*(cloud_c[i].pc1 + cloud_c[i].pc2);
    }
}

template<class PointT> void PointCloudExt<PointT>::performRangeThresholding(std::string axis, float startRange, float endRange, bool negative){
    pcl::PassThrough<PointT> passthroughFilter;
    passthroughFilter.setInputCloud(getPtrCopy());
    passthroughFilter.setFilterFieldName(axis);
    passthroughFilter.setFilterLimits(startRange, endRange);
    passthroughFilter.setNegative(negative);
    passthroughFilter.filter(*this);
}

template<class PointT> void PointCloudExt<PointT>::performDownsampling(float leafSize){
    pcl::VoxelGrid<PointT> filter;
    filter.setInputCloud(getPtrCopy());
    filter.setLeafSize(leafSize, leafSize, leafSize);
    filter.filter(*this);
}

template<> void PointCloudExt<pcl::PointXYZRGBNormal>::performDownsamplingKeepingNormals(float leafSize){
    PointCloudExt<pcl::PointXYZRGBNormal>& inputCloud = *this;
    PointCloudExt<pcl::PointXYZRGBNormal> filteredCloud;
    pcl::VoxelGrid<pcl::PointXYZRGBNormal> filter;
    filter.setInputCloud(getPtrCopy());
    filter.setLeafSize(leafSize, leafSize, leafSize);
    filter.filter(filteredCloud);
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
    kdTree->setInputCloud(getPtrCopy());
    #pragma omp parallel for shared(filteredCloud)
    for(int i=0; i<filteredCloud.size(); i++){
        pcl::PointXYZRGBNormal& point = filteredCloud[i];
        std::vector<int> treeIndices (1);
        std::vector<float> treeDists (1);
        kdTree->nearestKSearch(point, 1, treeIndices, treeDists);
        pcl::PointXYZRGBNormal closestPoint = inputCloud[treeIndices[0]];
        point.normal[0] = closestPoint.normal[0];
        point.normal[1] = closestPoint.normal[1];
        point.normal[2] = closestPoint.normal[2];
    }
    *this = filteredCloud;
}

template<> void PointCloudExt<pcl::PointXYZRGB>::performSmoothing(float searchRadius, int threadCount){
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr smoothedCloudNormal(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::MovingLeastSquaresOMP<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> smoothingFilter;
    smoothingFilter.setNumberOfThreads(threadCount);
    smoothingFilter.setInputCloud(getPtrCopy());
    smoothingFilter.setSearchRadius(searchRadius);
    smoothingFilter.setPolynomialFit(true);
    smoothingFilter.setComputeNormals(true);
    smoothingFilter.setSearchMethod (tree);
    smoothingFilter.process(*smoothedCloudNormal);
    copyPointCloud(*smoothedCloudNormal, *this);
}

#if PCL_VERSION_COMPARE(>=, 1, 8, 0)
template<> void PointCloudExt<pcl::PointXYZRGBNormal>::performSmoothing(float searchRadius, int threadCount){
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr smoothedCloudNormal(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::MovingLeastSquaresOMP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> smoothingFilter(threadCount);
    smoothingFilter.setInputCloud(getPtrCopy());
    smoothingFilter.setSearchRadius(searchRadius);
    smoothingFilter.setPolynomialFit(true);
    smoothingFilter.setComputeNormals(true);
    smoothingFilter.setSearchMethod (tree);
    smoothingFilter.process(*this);
    copyPointCloud(*smoothedCloudNormal, *this);
}
#endif

template<class PointT> pcl::ModelCoefficients::Ptr PointCloudExt<PointT>::performModelFitting(int model, float error){
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (model);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (error);
    seg.setInputCloud (this->getPtrCopy());
    seg.segment (*inliers, *coefficients);

    pcl::ProjectInliers<PointT> proj;
    proj.setModelType (model);
    proj.setIndices (inliers);
    proj.setInputCloud (this->getPtrCopy());
    proj.setModelCoefficients (coefficients);
    proj.filter (*this);
    return coefficients;
}

/**************************************************************************************
 * Extract Data
**************************************************************************************/

template<> PointCloudExt<pcl::PointXYZRGBNormal> PointCloudExt<pcl::PointXYZRGB>::extractSmoothing(float searchRadius, int threadCount){
    PointCloudExt<pcl::PointXYZRGBNormal> smoothedCloudNormal;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    pcl::MovingLeastSquaresOMP<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> smoothingFilter;
    smoothingFilter.setNumberOfThreads(threadCount);
    smoothingFilter.setInputCloud(getPtrCopy());
    smoothingFilter.setSearchRadius(searchRadius);
    smoothingFilter.setPolynomialFit(true);
    smoothingFilter.setComputeNormals(true);
    smoothingFilter.setSearchMethod (tree);
    smoothingFilter.process(smoothedCloudNormal);
    return smoothedCloudNormal;
}

template<class PointT> pcl::PointXYZ PointCloudExt<PointT>::extractCentroid(){
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*this, centroid);
    pcl::PointXYZ pointCentroid(centroid[0],centroid[1],centroid[2]);
    return pointCentroid;
}


template<class PointT> PointCloudExt<PointT> PointCloudExt<PointT>::extractEuclidianClusterCloud(float tolerance, int minClusterSize, int maxClusterSize){
    PointCloudExt<PointT> cleanedCloud;
    PointCloudExtPtr ptrCloud = getPtrCopy();
    KdTreePtr segkdtree(new pcl::search::KdTree<PointT>);
    segkdtree->setInputCloud(ptrCloud);
    pcl::EuclideanClusterExtraction<PointT> clustering;
    clustering.setClusterTolerance(tolerance);
    clustering.setMinClusterSize(minClusterSize);
    clustering.setMaxClusterSize(maxClusterSize);
    clustering.setSearchMethod(segkdtree);
    clustering.setInputCloud(ptrCloud);
    std::vector<pcl::PointIndices> clusters;
    clustering.extract(clusters);

    int label = 0;
    for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
    {
        for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
            cleanedCloud.push_back(this->points[*point]);
        label++;
    }
    return cleanedCloud;
}

template<class PointT> std::vector<PointCloudExt<PointT> > PointCloudExt<PointT>::extractEuclidianClusters(float tolerance, int minClusterSize, int maxClusterSize){
    PointCloudExtPtr ptrCloud = getPtrCopy();
    KdTreePtr segkdtree(new pcl::search::KdTree<PointT>);
    segkdtree->setInputCloud(ptrCloud);
    pcl::EuclideanClusterExtraction<PointT> clustering;
    clustering.setClusterTolerance(tolerance);
    clustering.setMinClusterSize(minClusterSize);
    clustering.setMaxClusterSize(maxClusterSize);
    clustering.setSearchMethod(segkdtree);
    clustering.setInputCloud(ptrCloud);
    std::vector<pcl::PointIndices> clusters;
    clustering.extract(clusters);

    int currentClusterNum = 1;
    std::vector<PointCloudExt<PointT> > clustersAllocated;
    for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
    {
        PointCloudExt<PointT> cluster;
        for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
            cluster.push_back(this->points[*point]);
        cluster.width = cluster.size();
        cluster.height = 1;
        cluster.is_dense = true;
        clustersAllocated.push_back(cluster);

        if (cluster.size() <= 0)
            break;
        currentClusterNum++;
    }

    return clustersAllocated;
}

template<class PointT> PointCloudExt<pcl::PointXYZRGBNormal> PointCloudExt<PointT>::extractNormals(float searchRadius, PointCloudExtParams::Mode searchMode){
    PointCloudExt<pcl::PointXYZRGBNormal> normals;
    pcl::copyPointCloud (*this, normals);
    pcl::NormalEstimation<PointT, pcl::PointXYZRGBNormal> normalEstimation;

    pcl::PointXYZ centroid = this->extractCentroid();
    normalEstimation.setViewPoint(centroid.data[0], centroid.data[1], centroid.data[2]);

    normalEstimation.setInputCloud(getPtrCopy());
    switch(searchMode)
    {
        case PointCloudExtParams::EXTRACT_NORMALS_RADIUSSEARCH: normalEstimation.setRadiusSearch(searchRadius);
        case PointCloudExtParams::EXTRACT_NORMALS_KSEARCH: normalEstimation.setKSearch((int)searchRadius);
        default: {}
    }
    KdTreePtr kdtree(new pcl::search::KdTree<PointT>);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(normals);

    for(size_t i = 0; i < normals.size (); ++i){
        normals.points[i].normal[0] *= -1;
        normals.points[i].normal[1] *= -1;
        normals.points[i].normal[2] *= -1;
    }

    return normals;
}

template<class PointT> PointCloudExt<pcl::PointXYZRGBNormal> PointCloudExt<PointT>::extractNormals(float searchRadius, PointCloudExtParams::Mode searchMode, int threadCount){
    PointCloudExt<pcl::PointXYZRGBNormal> normals;
    pcl::copyPointCloud (*this, normals);
    pcl::NormalEstimationOMP<PointT, pcl::PointXYZRGBNormal> normalEstimation(threadCount);

    pcl::PointXYZ centroid = this->extractCentroid();
    centroid.z -= 0.2;
    normalEstimation.setViewPoint(centroid.data[0], centroid.data[1], centroid.data[2]);

    normalEstimation.setInputCloud(getPtrCopy());
    switch(searchMode)
    {
        case PointCloudExtParams::EXTRACT_NORMALS_RADIUSSEARCH: normalEstimation.setRadiusSearch(searchRadius);
        case PointCloudExtParams::EXTRACT_NORMALS_KSEARCH: normalEstimation.setKSearch((int)searchRadius);
        default: {}
    }
    KdTreePtr kdtree(new pcl::search::KdTree<PointT>);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(normals);

    for(size_t i = 0; i < normals.size (); ++i){
        normals.points[i].normal[0] *= -1;
        normals.points[i].normal[1] *= -1;
        normals.points[i].normal[2] *= -1;
    }

    return normals;
}

template<class PointT> PointT PointCloudExt<PointT>::extractClosestPoint(PointT point){
    KdTreePtr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(getPtrCopy());
    std::vector<int> treeIndices (1);
    std::vector<float> treeDists (1);
    tree->nearestKSearch(point, 1, treeIndices, treeDists);
    PointT closestPoint = this->points[treeIndices[0]];
    return closestPoint;
}

template<class PointT> float PointCloudExt<PointT>::extractTotalDistanceMean(){
    pcl::PointXYZ centroid = this->extractCentroid();
    float totalDistanceMean = 0.0;
    for(Iterator it = this->begin(); it!= this->end(); it++){
        totalDistanceMean += sqrt(pow((it->x - centroid.x),2) + pow((it->y - centroid.y),2) + pow((it->z - centroid.z),2));
    }
    totalDistanceMean /= this->size();
    return totalDistanceMean;
}

// pcl::SACMODEL_LINE, 0.01, 50
template<class PointT> std::vector<PointCloudExt<PointT> > PointCloudExt<PointT>::extractModelFittedClusters(int model, float error, float eucDistance, int minPoints){
    PointCloudExt<PointT> topSurface = this->clone();
    std::vector<PointCloudExt<PointT> > topSurfaceClusters;
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (model);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (error);

    int j=0, nr_points = (int) topSurface.size ();
    while (topSurface.size () > 0.01 * nr_points)
    {
        // Segment the largest model component from the remaining cloud
        seg.setInputCloud (topSurface.getPtrCopy());
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () == 0)
        {
          std::cout << "Could not estimate a model for the given dataset." << std::endl;
          break;
        }

        // Extract the model inliers from the input cloud
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud (topSurface.getPtrCopy());
        extract.setIndices (inliers);
        extract.setNegative (false);

        // Get the points associated with the model
        PointCloudExt<PointT> modelPoints;
        extract.filter (modelPoints);

        // THIS PART NEEDS SOME WORK
        if(eucDistance){
            std::vector<PointCloudExt<PointT> > modelPointsClusters = modelPoints.extractEuclidianClusters(eucDistance, minPoints, 2500000);
            for(int j=0; j<modelPointsClusters.size(); j++) topSurfaceClusters.push_back(modelPointsClusters[j]);
        }else{
            if(modelPoints.size() > minPoints) topSurfaceClusters.push_back(modelPoints);
        }

        // Remove the planar inliers, extract the rest
        extract.setNegative (true);
        extract.filter (topSurface);
    }
    return topSurfaceClusters;
    cout << "Iterative RANSAC complete" << endl;
}

template<> Eigen::Vector3f PointCloudExt<pcl::PointXYZRGBNormal>::extractNormalCentroid(){
    Eigen::Vector3f normalCentroid;
    float totalX = 0., totalY = 0., totalZ = 0.;
    for(PointCloudExt<pcl::PointXYZRGBNormal>::iterator it = this->begin(); it!= this->end(); it++){
        totalX+=it->normal[0];
        totalY+=it->normal[1];
        totalZ+=it->normal[2];
    }
    normalCentroid(0) = totalX/this->size();
    normalCentroid(1) = totalY/this->size();
    normalCentroid(2) = totalZ/this->size();
    return normalCentroid.normalized();
}

template<> Eigen::Vector3f PointCloudExt<pcl::PointXYZRGBNormal>::extractColorCentroid(){
    Eigen::Vector3f colorCentroid;
    float totalR = 0., totalG = 0., totalB = 0.;
    for(PointCloudExt<pcl::PointXYZRGBNormal>::iterator it = this->begin(); it!= this->end(); it++){
        totalR+=it->r;
        totalG+=it->g;
        totalB+=it->b;
    }
    colorCentroid(0) = totalR/this->size();
    colorCentroid(1) = totalG/this->size();
    colorCentroid(2) = totalB/this->size();
    return colorCentroid;
}

/**************************************************************************************
 * Space Conversion
**************************************************************************************/

template<class PointT> void PointCloudExt<PointT>::performPolarConversion(){
    for(Iterator it = this->begin(); it!= this->end(); it++){
        float r =  sqrt(pow((it->x),2) + pow((it->y),2) + pow((it->z),2));
        float theta = acos(it->z / r);
        float phi = atan2(it->y, it->x);
        it->x = r; it->y = theta; it->z = phi;
    }
}

template<class PointT> void PointCloudExt<PointT>::performCartesianConversion(){
    for(Iterator it = this->begin(); it!= this->end(); it++){
        float r = it->x; float theta = it->y; float phi = it->z;
        it->x = r*sin(theta)*cos(phi);
        it->y = r*sin(theta)*sin(phi);
        it->z = r*cos(theta);
    }
}

template<> void PointCloudExt<pcl::PointXYZRGBNormal>::swapNormalsAndXYZ(){
    for(PointCloudExt<pcl::PointXYZRGBNormal>::iterator it = this->begin(); it!= this->end(); it++){
        swap(it->x, it->normal[0]);
        swap(it->y, it->normal[1]);
        swap(it->z, it->normal[2]);
    }
}

/**************************************************************************************
 * Color
**************************************************************************************/

template<> void PointCloudExt<pcl::PointXYZRGB>::colorize(int r, int g, int b){
    for(PointCloudExt<pcl::PointXYZRGB>::iterator it = this->begin(); it!= this->end(); it++){
        it->r = r; it->g = g; it->b = b;
    }
}

template<> void PointCloudExt<pcl::PointXYZRGBNormal>::colorize(int r, int g, int b){
    for(PointCloudExt<pcl::PointXYZRGBNormal>::iterator it = this->begin(); it!= this->end(); it++){
        it->r = r; it->g = g; it->b = b;
    }
}

template<> void PointCloudExt<pcl::PointXYZRGB>::colorize(){
    int rColor = rand() % 255;
    int gColor = rand() % 255;
    int bColor = rand() % 255;
    for(PointCloudExt<pcl::PointXYZRGB>::iterator it = this->begin(); it!= this->end(); it++){
        it->r = rColor; it->g = gColor; it->b = bColor;
    }
}

template<> void PointCloudExt<pcl::PointXYZRGBNormal>::colorize(){
    int rColor = rand() % 255;
    int gColor = rand() % 255;
    int bColor = rand() % 255;
    for(PointCloudExt<pcl::PointXYZRGBNormal>::iterator it = this->begin(); it!= this->end(); it++){
        it->r = rColor; it->g = gColor; it->b = bColor;
    }
}

/**************************************************************************************
 * Pixel Operations
**************************************************************************************/

template<class PointT> std::vector<cv::Point2i> PointCloudExt<PointT>::projectPoints(MatExt worldToCamera, MatExt rgbCameraMatrix){
    PointCloudExt<PointT>& pointCloud = *this;
    std::vector<cv::Point2i> cameraPoints;
    for(Iterator it = pointCloud.begin(); it!= pointCloud.end(); it++){
        PointT currPoint = *it;
        PointT currPointCamera = PointCloudExt<PointT>::transform(currPoint, worldToCamera);
        cv::Point2i currPixelRGB= PointCloudExt<PointT>::project(currPointCamera, rgbCameraMatrix);
        cameraPoints.push_back(currPixelRGB);
    }
    return cameraPoints;
}

template<> void PointCloudExt<pcl::PointXYZRGBNormal>::projectPoints(MatExt worldToCamera, MatExt rgbCameraMatrix, MatExt& image, bool cloudColor, bool circleFill, cv::Scalar fillColor){
    PointCloudExt<pcl::PointXYZRGBNormal> cameraPoints = this->extractTransform(worldToCamera);
    float fx = rgbCameraMatrix.at<float>(0,0);
    float fy = rgbCameraMatrix.at<float>(1,1);
    float cx = rgbCameraMatrix.at<float>(0,2);
    float cy = rgbCameraMatrix.at<float>(1,2);
    for(PointCloudExt<pcl::PointXYZRGBNormal>::iterator it = cameraPoints.begin(); it!= cameraPoints.end(); it++){
        int u = (int)(((fx*it->x)/it->z) + cx);
        int v = (int)(((fy*it->y)/it->z) + cy);
        if (v < image.rows && v >= 0 && u >= 0 && u < image.cols) {
            if(circleFill){
                if(cloudColor) { cv::circle(image, cv::Point(u,v), 3, cv::Scalar(it->r, it->g, it->b), CV_FILLED); }
                else { cv::circle(image, cv::Point(u,v), 3, fillColor, CV_FILLED); }
            }else{
                if(cloudColor) { cv::Vec3b& color = image.at<cv::Vec3b>(cv::Point(u,v)); color[0] = it->r; color[1] = it->g; color[2] = it->b; }
                else { cv::Vec3b& color = image.at<cv::Vec3b>(cv::Point(u,v)); color[0] = fillColor[0]; color[1] = fillColor[1]; color[2] = fillColor[2]; }
            }
        }
    }
}

template<> void PointCloudExt<pcl::PointXYZRGB>::projectPoints(MatExt worldToCamera, MatExt rgbCameraMatrix, MatExt& image, bool cloudColor, bool circleFill, cv::Scalar fillColor){
    PointCloudExt<pcl::PointXYZRGB> cameraPoints = this->extractTransform(worldToCamera);
    float fx = rgbCameraMatrix.at<float>(0,0);
    float fy = rgbCameraMatrix.at<float>(1,1);
    float cx = rgbCameraMatrix.at<float>(0,2);
    float cy = rgbCameraMatrix.at<float>(1,2);
    for(PointCloudExt<pcl::PointXYZRGB>::iterator it = cameraPoints.begin(); it!= cameraPoints.end(); it++){
        int u = (int)(((fx*it->x)/it->z) + cx);
        int v = (int)(((fy*it->y)/it->z) + cy);
        if (v < image.rows && v >= 0 && u >= 0 && u < image.cols) {
            if(circleFill){
                if(cloudColor) { cv::circle(image, cv::Point(u,v), 3, cv::Scalar(it->r, it->g, it->b), CV_FILLED); }
                else { cv::circle(image, cv::Point(u,v), 3, fillColor, CV_FILLED); }
            }else{
                if(cloudColor) { cv::Vec3b& color = image.at<cv::Vec3b>(cv::Point(u,v)); color[0] = it->r; color[1] = it->g; color[2] = it->b; }
                else { cv::Vec3b& color = image.at<cv::Vec3b>(cv::Point(u,v)); color[0] = fillColor[0]; color[1] = fillColor[1]; color[2] = fillColor[2]; }
            }
        }
    }
}

template<> void PointCloudExt<pcl::PointXYZ>::projectPoints(MatExt worldToCamera, MatExt rgbCameraMatrix, MatExt& image, bool cloudColor, bool circleFill, cv::Scalar fillColor){
    PointCloudExt<pcl::PointXYZ> cameraPoints = this->extractTransform(worldToCamera);
    float fx = rgbCameraMatrix.at<float>(0,0);
    float fy = rgbCameraMatrix.at<float>(1,1);
    float cx = rgbCameraMatrix.at<float>(0,2);
    float cy = rgbCameraMatrix.at<float>(1,2);
    for(PointCloudExt<pcl::PointXYZ>::iterator it = cameraPoints.begin(); it!= cameraPoints.end(); it++){
        int u = (int)(((fx*it->x)/it->z) + cx);
        int v = (int)(((fy*it->y)/it->z) + cy);
        if (v < image.rows && v >= 0 && u >= 0 && u < image.cols) {
            if(circleFill){
                if(cloudColor) { cv::circle(image, cv::Point(u,v), 3, fillColor, CV_FILLED); }
                else { cv::circle(image, cv::Point(u,v), 3, fillColor, CV_FILLED); }
            }else{
                if(cloudColor) { cv::Vec3b& color = image.at<cv::Vec3b>(cv::Point(u,v)); color[0] = fillColor[0]; color[1] = fillColor[1]; color[2] = fillColor[2]; }
                else { cv::Vec3b& color = image.at<cv::Vec3b>(cv::Point(u,v)); color[0] = fillColor[0]; color[1] = fillColor[1]; color[2] = fillColor[2]; }
            }
        }
    }
}

/**************************************************************************************
 * Other Operations
**************************************************************************************/

template<class PointT> void PointCloudExt<PointT>::removeNANPoints(){
    PointCloudExt<PointT>& topSurface = *this;
    PointCloudExt<PointT> tempSurface;
    std::vector<int> nanIndices;
    pcl::removeNaNFromPointCloud(topSurface, tempSurface, nanIndices); topSurface = tempSurface;
}

template<> void PointCloudExt<pcl::PointXYZRGBNormal>::removeNANPoints(){
    PointCloudExt<pcl::PointXYZRGBNormal>& topSurface = *this;
    PointCloudExt<pcl::PointXYZRGBNormal> tempSurface;
    std::vector<int> nanIndices;
    pcl::removeNaNFromPointCloud(topSurface, tempSurface, nanIndices); topSurface = tempSurface;
    pcl::removeNaNNormalsFromPointCloud(topSurface, tempSurface, nanIndices); topSurface = tempSurface;
}

template<class PointT> PointT PointCloudExt<PointT>::transform(PointT& point, MatExt transformMatrix){
    PointT transformedPoint(point);
    transformedPoint.x = point.x*transformMatrix.at<float>(0,0) + point.y*transformMatrix.at<float>(0,1) + point.z*transformMatrix.at<float>(0,2) + transformMatrix.at<float>(0,3);
    transformedPoint.y = point.x*transformMatrix.at<float>(1,0) + point.y*transformMatrix.at<float>(1,1) + point.z*transformMatrix.at<float>(1,2) + transformMatrix.at<float>(1,3);
    transformedPoint.z = point.x*transformMatrix.at<float>(2,0) + point.y*transformMatrix.at<float>(2,1) + point.z*transformMatrix.at<float>(2,2) + transformMatrix.at<float>(2,3);
    return transformedPoint;
}

template<class PointT> PointT PointCloudExt<PointT>::transform(PointT& point, Eigen::Matrix4f transformMatrix){
    PointT transformedPoint(point);
    transformedPoint.x = (point.x*transformMatrix(0,0) + point.y*transformMatrix(0,1) + point.z*transformMatrix(0,2) + transformMatrix(0,3));
    transformedPoint.y = (point.x*transformMatrix(1,0) + point.y*transformMatrix(1,1) + point.z*transformMatrix(1,2) + transformMatrix(1,3));
    transformedPoint.z = (point.x*transformMatrix(2,0) + point.y*transformMatrix(2,1) + point.z*transformMatrix(2,2) + transformMatrix(2,3));
    return transformedPoint;
}

template<class PointT> PointT PointCloudExt<PointT>::backproject(cv::Point2i& pixel, uint16_t depth, MatExt cameraMatrix){
    PointT point;
    float fx = cameraMatrix.at<float>(0,0);
    float fy = cameraMatrix.at<float>(1,1);
    float cx = cameraMatrix.at<float>(0,2);
    float cy = cameraMatrix.at<float>(1,2);
    point.z = (float)depth*0.001;
    point.y = (((float)pixel.y - cy) * point.z) / (fy);
    point.x = (((float)pixel.x - cx) * point.z) / (fx);
    return point;
}

template<class PointT> cv::Point2i PointCloudExt<PointT>::project(PointT& point, MatExt cameraMatrix){
    cv::Point2i pixel;
    float fx = cameraMatrix.at<float>(0,0);
    float fy = cameraMatrix.at<float>(1,1);
    float cx = cameraMatrix.at<float>(0,2);
    float cy = cameraMatrix.at<float>(1,2);
    pixel.x = (int)(((fx*point.x)/point.z) + cx);
    pixel.y = (int)(((fy*point.y)/point.z) + cy);
    return pixel;
}

template class PointCloudExt<pcl::PointXYZ>;
template class PointCloudExt<pcl::PointXYZI>;
template class PointCloudExt<pcl::PointXYZRGB>;
template class PointCloudExt<pcl::PointXYZRGBNormal>;
