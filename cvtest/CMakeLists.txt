# CMakeList.txt : CMake project for cvtest, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

add_subdirectory ("stitching")
# Add source to this project's executable.
add_executable (cvtest "cvtest.cpp" "cvtest.h")
set (OpenCV_DIR D:/pcgdev/cpp/opencv/build/install)
#requires opencv
find_package(OpenCV 4 REQUIRED)
#show a message with the opencv version detected
MESSAGE("opencv version: ${OpenCV_VERSION}")
  # Additional Include Directories
  include_directories(${OpenCV_INCLUDE_DIRS})
  # Additional Library Directories
  link_directories(${OpenCV_LIBRARY_DIRS})
  # Additional Dependencies
  target_link_libraries( cvtest ${OpenCV_LIBS} StitchingFunctions )

# TODO: Add tests and install targets if needed.
