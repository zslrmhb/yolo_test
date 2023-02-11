cmake_minimum_required(VERSION 3.22)
project(CV Test_Code)

set(CMAKE_CXX_STANDARD 11)
find_library( CAMERA_LIB NAMES MVSDK HINTS ./camera_lib/x86 REQUIRED)
find_package( OpenCV REQUIRED )



include_directories(camera_include)
include_directories( ${OpenCV_INCLUDE_DIRS} )




add_executable(YOLO main.cpp Camera.cpp Camera.h)
message(STATUS ${CAMERA_LIB})
target_link_libraries(test  ${CAMERA_LIB} ${OpenCV_LIBS} rt pthread)

