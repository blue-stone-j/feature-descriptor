cmake_minimum_required(VERSION 3.16.3)
project(sift)

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED QUIET)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# include_directories(include ./common)
set(TARGET_LIBRARIES "")
list(APPEND TARGET_LIBRARIES ${GLOG_LIBRARIES})
list(APPEND TARGET_LIBRARIES ${OpenCV_LIBRARIES})

add_executable(sift sift/main.cpp sift/display.cpp sift/match.cpp sift/sift.cpp)
target_link_libraries(sift ${TARGET_LIBRARIES} gflags)
