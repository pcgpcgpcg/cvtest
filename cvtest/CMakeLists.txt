# CMakeList.txt : CMake project for cvtest, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
# Add source to this project's executable.
add_executable (cvtest "cvtest.cpp" "cvtest.h")
set (OpenCV_DIR D:/Downloads/opencv/build)
find_package(OpenCV 4 REQUIRED)
  # Additional Include Directories
  include_directories(${OpenCV_INCLUDE_DIRS})
  # Additional Library Directories
  link_directories(${OpenCV_LIBRARY_DIRS})
  # Additional Dependencies
  target_link_libraries( cvtest ${OpenCV_LIBS} )

# TODO: Add tests and install targets if needed.
