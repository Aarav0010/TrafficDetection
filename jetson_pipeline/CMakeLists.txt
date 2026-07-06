cmake_minimum_required(VERSION 3.15)
project(TrafficDetectionCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Jetson Orin L4T Native Dependencies
# ============================================================================
# OpenCV (CUDA, DNN, and GStreamer are pre-compiled into L4T OpenCV)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn highgui)

# cpp-httplib (header-only)
set(HTTPLIB_INCLUDE "${CMAKE_SOURCE_DIR}/deps")

# ============================================================================
# Executable
# ============================================================================
add_executable(traffic_detection main.cpp)

target_include_directories(traffic_detection PRIVATE
    ${HTTPLIB_INCLUDE}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(traffic_detection PRIVATE
    ${OpenCV_LIBS}
    pthread
)
