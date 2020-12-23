cmake_minimum_required(VERSION 2.8.3)
project(perception_lib)

set (CMAKE_CXX_STANDARD 11)

if(NOT "${PYTHON_EXECUTABLE}" STREQUAL "/usr/bin/python")
  find_package(PythonLibs 3)
  include_directories(${PYTHON_INCLUDE_DIRS})
endif()

find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
   ${CMAKE_CURRENT_SOURCE_DIR}/src
  python
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

file(GLOB perception_lib_SRC
    "include/*.h"
    "include/*.hpp"
    "include/*.cl"
    "src/*.cpp"
    #"../vision_launch/launch/*"
    #"../vision_launch/scripts/*"
)

add_library(perception_lib SHARED ${perception_lib_SRC})
target_link_libraries(
  perception_lib
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
set_target_properties(perception_lib PROPERTIES LINKER_LANGUAGE CXX)


# ROS Flag
if (${ROS})
  add_definitions(-DROS_FOUND)
  message(STATUS "Building with ROS support.")
  find_package(catkin REQUIRED COMPONENTS
      pcl_conversions
      pcl_ros
      roscpp
      rospy
      cv_bridge
      geometry_msgs
      std_msgs
  )
  include_directories(${catkin_INCLUDE_DIRS})
else()
  set(catkin_LIBRARIES "")
endif()

# Pylc
add_subdirectory(pybind11)
pybind11_add_module(pyperception_lib python/pyperception_lib.cpp)
target_link_libraries(pyperception_lib PRIVATE pybind11::module
    perception_lib ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${catkin_LIBRARIES}
)