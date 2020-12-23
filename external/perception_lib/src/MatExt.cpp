#include "MatExt.h"

MatExt::MatExt(){

}

MatExt::MatExt(const cv::Mat& inputImage)
{
    inputImage.copyTo(*this);
}

#ifdef ROS_FOUND
MatExt::MatExt(const sensor_msgs::CompressedImageConstPtr& inputImage)
{
    cv::Mat img = cv::imdecode(cv::Mat(inputImage->data), CV_LOAD_IMAGE_UNCHANGED);
    img.copyTo(*this);
}

MatExt::MatExt(const sensor_msgs::ImageConstPtr& inputImage)
{
    cv::Mat img = cv_bridge::toCvShare(inputImage, inputImage->encoding)->image;
    img.copyTo(*this);
}

MatExt::MatExt(const sensor_msgs::Image& inputImage)
{
    sensor_msgs::ImageConstPtr inputImageConv = boost::make_shared<const sensor_msgs::Image>(inputImage);
    cv::Mat img = cv_bridge::toCvShare(inputImageConv, inputImage.encoding)->image;
    img.copyTo(*this);
}
#endif

MatExt::~MatExt(){

}

MatExt MatExt::clone()
{
    MatExt img;
    this->copyTo(img);
    //img.time = this->time;
    img.name = this->name;
    return img;
}

bool MatExt::empty()
{
    if(this->rows == 0 || this->cols == 0) return true;
    return false;
}

void MatExt::performInvert()
{
    cv::invert(*this,*this);
}

void MatExt::performTranspose()
{
    cv::transpose(*this,*this);
}

MatExt MatExt::extractInvert()
{
    MatExt inverted;
    try{
        cv::invert(*this,inverted);
    }catch(cv::Exception & e){
        std::cout << *this << std::endl;
        //ROS_ERROR("Invert failure. Returning identity");
        cv::Mat eye = cv::Mat::eye(4,4,CV_32FC1);
        return MatExt(eye);
    }

    return inverted;
}

MatExt MatExt::extractTranspose()
{
    MatExt transposed;
    cv::transpose(*this,transposed);
    return transposed;
}

MatExt MatExt::extractResize(double scale)
{
    MatExt resized;
    cv::resize(*this, resized, cv::Size(0,0), scale, scale);
    return resized;
}

void MatExt::performScale(double scale)
{
    MatExt& img = *this;
    int type = img.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch(depth){
    case CV_64F:{
        cv::MatIterator_<double> it, end;
        for( it = img.begin<double>(), end = img.end<double>(); it != end; ++it){
            if(*it==1 || *it==0) continue;
            *it *= scale;
        }
        break;
    }
    case CV_32F:{
        cv::MatIterator_<float> it, end;
        for( it = img.begin<float>(), end = img.end<float>(); it != end; ++it){
            if(*it==1 || *it==0) continue;
            *it *= scale;
        }
        break;
    }
    default: break;
    }
}

void MatExt::performResize(double scale)
{
    MatExt& img = *this;
    cv::resize(img, img, cv::Size(0,0), scale, scale);
}

bool MatExt::performColorConvert()
{
    MatExt& img = *this;
    if(img.channels() != 1) return false;
    MatExt color;
    cv::cvtColor( img, color, CV_GRAY2RGB );
    color.copyTo(*this);
    return true;
}

bool MatExt::performGrayConvert()
{
    MatExt& img = *this;
    if(img.channels() != 3) return false;
    MatExt gray;
    cv::cvtColor( img, gray, CV_RGB2GRAY );
    gray.copyTo(*this);
    return true;
}

MatExt MatExt::extractColorConvert()
{
    MatExt& img = *this;
    MatExt color;
    cv::cvtColor( img, color, CV_GRAY2RGB );
    return color;
}

MatExt MatExt::extractGrayConvert()
{
    MatExt& img = *this;
    MatExt gray;
    cv::cvtColor( img, gray, CV_RGB2GRAY );
    return gray;
}

void MatExt::performSobel(int kernel, double scale, double delta)
{
    MatExt& img = *this;
    img.performGrayConvert();

    /// Generate grad_x and grad_y
    MatExt grad_x, grad_y, grad;
    MatExt abs_grad_x, abs_grad_y;

    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    cv::Sobel( img, grad_x, CV_16S, 1, 0, kernel, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    cv::Sobel( img, grad_y, CV_16S, 0, 1, kernel, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    grad.copyTo(*this);
}

std::string MatExt::getType(){
    MatExt& img = *this;
    int type = img.type();
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

#ifdef ROS_FOUND
std::string MatExt::getROSType(){
    std::string type = getType();
    std::string rosType;
    if(type == "8UC1") return "mono8";
    if(type == "16UC1") return "mono16";
    if(type == "8UC3") return "bgr8";
    if(type == "8UC4") return "bgra8";
    return "User";
}

sensor_msgs::Image MatExt::getROSMsg(){
    return *cv_bridge::CvImage(std_msgs::Header(), getROSType(), *this).toImageMsg();
}
#endif

