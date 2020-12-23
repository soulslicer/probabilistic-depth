#ifndef PYLC_LIB_HPP
#define PYLC_LIB_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <pybind11/eigen.h>
#include <opencv2/opencv.hpp>
#include <opencv/cxeigen.hpp>
#ifdef ROS_FOUND
#include <ros/ros.h>
#include <ros/package.h>
#endif
#include <VisualizerExt.h>

void test(PointCloudExt<pcl::PointXYZRGBNormal>& cloud){
    std::cout << cloud.size() << std::endl;
}

class Visualizer
{
    public:

    void start(){
        VisualizerExt::start();
    }

    void loop(){
        VisualizerExt::loop();
    }

    void stop(){
        VisualizerExt::stop();
    }

    void swapBuffer(){
        VisualizerExt::swapBuffer();
    }

    void addCloud(PointCloudExt<pcl::PointXYZRGBNormal>& cloud, int size){
        VisualizerExt::addPointCloud(cloud, size);
    }

    void saveScreenshot(std::string file){
        VisualizerExt::saveScreenshot(file);
    }

    cv::Mat getRenderedImage(){
        return VisualizerExt::getRenderedImage();
    }

};

namespace py = pybind11;

PYBIND11_MODULE(pyperception_lib, m) {

   // Output Object
   py::class_<Visualizer, std::shared_ptr<Visualizer>>(m, "Visualizer")
        .def(py::init<>())
        .def("start", &Visualizer::start)
        .def("loop", &Visualizer::loop)
        .def("stop", &Visualizer::stop)
        .def("swapBuffer", &Visualizer::swapBuffer)
        .def("addCloud", &Visualizer::addCloud)
        .def("saveScreenshot", &Visualizer::saveScreenshot)
        .def("getRenderedImage", &Visualizer::getRenderedImage)
        ;

    m.def("test", &test, "test");

    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif
}

// Numpy - op::Array<float> interop
namespace pybind11 { namespace detail {

template <> struct type_caster<PointCloudExt<pcl::PointXYZRGBNormal>> {
    public:

        PYBIND11_TYPE_CASTER(PointCloudExt<pcl::PointXYZRGBNormal>, _("numpy.ndarray"));

        // Cast numpy to PointCloudExt
        bool load(handle src, bool imp)
        {
            try
            {
                // array b(src, true);
                array b = reinterpret_borrow<array>(src);
                buffer_info info = b.request();

                if (info.format != format_descriptor<float>::format())
                    throw std::runtime_error("only supports float32 now");

                //std::vector<int> a(info.shape);
                std::vector<int> shape(std::begin(info.shape), std::end(info.shape));

                if(shape[1] != 9)
                    throw std::runtime_error("x y z r g b nx ny nz format");

                PointCloudExt<pcl::PointXYZRGBNormal>& cloud = value;

                float* dataPtr = (float*)info.ptr;
                for(int i=0; i<shape[0]; i++){
                    float x = dataPtr[i*shape[1] + 0];
                    float y = dataPtr[i*shape[1] + 1];
                    float z = dataPtr[i*shape[1] + 2];
                    float r = dataPtr[i*shape[1] + 3];
                    float g = dataPtr[i*shape[1] + 4];
                    float b = dataPtr[i*shape[1] + 5];
                    float nx = dataPtr[i*shape[1] + 6];
                    float ny = dataPtr[i*shape[1] + 7];
                    float nz = dataPtr[i*shape[1] + 8];

                    //printf("%f %f %f %f %f %f %f %f %f \n", x, y, z, r, g, b, nx, ny, nz);

                    pcl::PointXYZRGBNormal pt;
                    pt.x = x;
                    pt.y = y;
                    pt.z = z;
                    pt.r = r;
                    pt.g = g;
                    pt.b = b;
                    pt.normal_x = nx;
                    pt.normal_y = ny;
                    pt.normal_z = nz;
                    cloud.push_back(pt);
                }

                return true;
            }
            catch (const std::exception& e)
            {
                std::cout << e.what() << std::endl;
                return {};
            }
        }

        // Cast op::Array<float> to numpy
        static handle cast(const PointCloudExt<pcl::PointXYZRGBNormal> &m, return_value_policy, handle defval)
        {
            // UNUSED(defval);
            // std::string format = format_descriptor<float>::format();
            // return array(buffer_info(
            //     m.getPseudoConstPtr(),/* Pointer to buffer */
            //     sizeof(float),        /* Size of one scalar */
            //     format,               /* Python struct-style format descriptor */
            //     m.getSize().size(),   /* Number of dimensions */
            //     m.getSize(),          /* Buffer dimensions */
            //     m.getStride()         /* Strides (in bytes) for each index */
            //     )).release();
        }

    };
}} // namespace pybind11::detail

// Numpy - cv::Mat interop
namespace pybind11 { namespace detail {

        template <> struct type_caster<cv::Mat> {
        public:

            PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

            // Cast numpy to cv::Mat
            bool load(handle src, bool)
            {
                /* Try a default converting into a Python */
                //array b(src, true);
                array b = reinterpret_borrow<array>(src);
                buffer_info info = b.request();

                int ndims = info.ndim;

                decltype(CV_32F) dtype;
                size_t elemsize;
                if (info.format == format_descriptor<float>::format()) {
                    if (ndims == 3) {
                        dtype = CV_32FC3;
                    } else {
                        dtype = CV_32FC1;
                    }
                    elemsize = sizeof(float);
                } else if (info.format == format_descriptor<double>::format()) {
                    if (ndims == 3) {
                        dtype = CV_64FC3;
                    } else {
                        dtype = CV_64FC1;
                    }
                    elemsize = sizeof(double);
                } else if (info.format == format_descriptor<unsigned char>::format()) {
                    if (ndims == 3) {
                        dtype = CV_8UC3;
                    } else {
                        dtype = CV_8UC1;
                    }
                    elemsize = sizeof(unsigned char);
                } else {
                    throw std::logic_error("Unsupported type");
                    return false;
                }

                std::vector<int> shape = {(int)info.shape[0], (int)info.shape[1]};

                value = cv::Mat(cv::Size(shape[1], shape[0]), dtype, info.ptr, cv::Mat::AUTO_STEP);
                return true;
            }

            // Cast cv::Mat to numpy
            static handle cast(const cv::Mat &m, return_value_policy, handle defval)
            {
                std::string format = format_descriptor<unsigned char>::format();
                size_t elemsize = sizeof(unsigned char);
                int dim, channels;
                switch(m.type()) {
                    case CV_8U:
                        format = format_descriptor<unsigned char>::format();
                        elemsize = sizeof(unsigned char);
                        dim = 2;
                        channels = 0;
                        break;
                    case CV_8UC3:
                        format = format_descriptor<unsigned char>::format();
                        elemsize = sizeof(unsigned char);
                        dim = 3;
                        channels = 3;
                        break;
                    case CV_8UC4:
                        format = format_descriptor<unsigned char>::format();
                        elemsize = sizeof(unsigned char);
                        dim = 3;
                        channels = 4;
                        break;
                    case CV_32F:
                        format = format_descriptor<float>::format();
                        elemsize = sizeof(float);
                        dim = 2;
                        channels = 0;
                        break;
                    case CV_64F:
                        format = format_descriptor<double>::format();
                        elemsize = sizeof(double);
                        dim = 2;
                        channels = 0;
                        break;
                    case CV_32FC4:
                        format = format_descriptor<float>::format();
                        elemsize = sizeof(float);
                        dim = 3;
                        channels = 4;
                        break;
                    default:
                        throw std::logic_error("Unsupported type");
                }

                std::vector<size_t> bufferdim;
                std::vector<size_t> strides;
                if (dim == 2) {
                    bufferdim = {(size_t) m.rows, (size_t) m.cols};
                    strides = {elemsize * (size_t) m.cols, elemsize};
                } else if (dim == 3) {
                    bufferdim = {(size_t) m.rows, (size_t) m.cols, (size_t) channels};
                    strides = {(size_t) elemsize * m.cols * channels, (size_t) elemsize * channels, (size_t) elemsize};
                }
                return array(buffer_info(
                        m.data,         /* Pointer to buffer */
                        elemsize,       /* Size of one scalar */
                        format,         /* Python struct-style format descriptor */
                        dim,            /* Number of dimensions */
                        bufferdim,      /* Buffer dimensions */
                        strides         /* Strides (in bytes) for each index */
                )).release();
            }

        };
    }} // namespace pybind11::detail

#endif


//std::vector<int> strides(bufferdim.size());
//if (!strides.empty())
//{
//    strides.back() = sizeof(float);
//    for (auto i = (int)strides.size()-2 ; i > -1 ; i--)
//        strides[i] = strides[i+1] * bufferdim[i+1];
//}
