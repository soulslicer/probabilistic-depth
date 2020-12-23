cmake_minimum_required(VERSION 2.8.3)
project(deval)


include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


find_package(OpenCV REQUIRED)
find_package( Eigen3 REQUIRED )

include_directories(
    "/usr/local/include"
    "${CMAKE_SOURCE_DIR}/src"
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

add_library(evaluate_depth STATIC "src/evaluate_depth.h" )
target_link_libraries(
  evaluate_depth
  png
  ${OpenCV_LIBRARIES}
)
set_target_properties(evaluate_depth PROPERTIES LINKER_LANGUAGE CXX)

add_subdirectory(pybind11)

pybind11_add_module(pyevaluatedepth_lib python/pyevaluatedepth_lib.cpp)
target_link_libraries(pyevaluatedepth_lib PRIVATE pybind11::module
    evaluate_depth png ${OpenCV_LIBRARIES}
)
