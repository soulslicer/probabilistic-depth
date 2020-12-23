#ifndef PYLC_LIB_HPP
#define PYLC_LIB_HPP

#include <evaluate_depth.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <pybind11/eigen.h>


namespace py = pybind11;

//void test(DepthImage &D_gt){
//    std::cout << D_gt.width() << " " << D_gt.height() << std::endl;
//}

PYBIND11_MODULE(pyevaluatedepth_lib, m) {

    m.def("depthError", &depthError, "depthError");
    m.def("evaluateErrors", &evaluateErrors, "evaluateErrors");

    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif
}

// Numpy - op::Array<float> interop
namespace pybind11 { namespace detail {

template <> struct type_caster<DepthImage> {
    public:

        PYBIND11_TYPE_CASTER(DepthImage, _("numpy.ndarray"));

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

                float* dataPtr = (float*)info.ptr;

                DepthImage& dImg = value;
                value = DepthImage (dataPtr, shape[1], shape[0]);

                return true;
            }
            catch (const std::exception& e)
            {
                std::cout << e.what() << std::endl;
                return {};
            }
        }

        // Cast op::Array<float> to numpy
        static handle cast(const DepthImage &m, return_value_policy, handle defval)
        {
            throw std::runtime_error("Not implemented");
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

#endif


