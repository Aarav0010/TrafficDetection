cmake_minimum_required(VERSION 3.15)
project(TrafficDetectionCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Dependency Paths (relative to cpp_pipeline/)
# ============================================================================
set(DEPS_DIR "${CMAKE_SOURCE_DIR}/deps")

# ONNX Runtime GPU SDK
set(ONNXRUNTIME_DIR "${DEPS_DIR}/onnxruntime-win-x64-gpu-1.20.0")
set(ONNXRUNTIME_INCLUDE "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

# OpenCV
set(OpenCV_DIR "${DEPS_DIR}/opencv/build")
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn highgui)

# cpp-httplib (header-only)
set(HTTPLIB_INCLUDE "${DEPS_DIR}")

# ============================================================================
# Executable
# ============================================================================
add_executable(traffic_detection main.cpp)

target_include_directories(traffic_detection PRIVATE
    ${ONNXRUNTIME_INCLUDE}
    ${HTTPLIB_INCLUDE}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(traffic_detection PRIVATE
    ${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib
    ${OpenCV_LIBS}
    ws2_32  # Required by cpp-httplib on Windows
)

# ============================================================================
# Copy DLLs to output directory
# ============================================================================
# Copy OpenCV DLLs automatically
add_custom_command(TARGET traffic_detection POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/bin/opencv_world4100.dll"
        "$<TARGET_FILE_DIR:traffic_detection>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/bin/opencv_videoio_ffmpeg4100_64.dll"
        "$<TARGET_FILE_DIR:traffic_detection>"
    COMMENT "Copying runtime DLLs to output directory..."
)

# Copy model files and video to build directory
add_custom_command(TARGET traffic_detection POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/../yolov8n.onnx"
        "$<TARGET_FILE_DIR:traffic_detection>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/../best.onnx"
        "$<TARGET_FILE_DIR:traffic_detection>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/../best (3).onnx"
        "$<TARGET_FILE_DIR:traffic_detection>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/../edited_ultimate_video.mp4"
        "$<TARGET_FILE_DIR:traffic_detection>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/../index.html"
        "$<TARGET_FILE_DIR:traffic_detection>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/../image.png"
        "$<TARGET_FILE_DIR:traffic_detection>"
    COMMENT "Copying models and video to output directory..."
)
