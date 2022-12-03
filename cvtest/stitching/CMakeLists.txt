# find all src at this directory
# and save to DIR_LIB_SRCS
aux_source_directory(. DIR_LIB_SRCS)
# generate StitchingFunctions lib
add_library (StitchingFunctions ${DIR_LIB_SRCS})

set (OpenCV_DIR D:/Downloads/opencv/build)
#requires opencv
find_package(OpenCV 4 REQUIRED)
#show a message with the opencv version detected
MESSAGE("opencv version: ${OpenCV_VERSION}")
  # Additional Include Directories
  include_directories(${OpenCV_INCLUDE_DIRS})
  # Additional Library Directories
  link_directories(${OpenCV_LIBRARY_DIRS})
  # Additional Dependencies
  target_link_libraries( StitchingFunctions ${OpenCV_LIBS} )
