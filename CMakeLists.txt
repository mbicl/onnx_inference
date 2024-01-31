cmake_minimum_required(VERSION 3.13)
project(onnx_modelzoo)

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/include/onnxruntime)

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/lib/onnxruntime)

add_executable(onnx_modelzoo onnx_modelzoo.cpp)
target_include_directories(onnx_modelzoo PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(onnx_modelzoo PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES} Threads::Threads)