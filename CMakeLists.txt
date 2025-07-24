cmake_minimum_required(VERSION 3.18)
project(mimo_unet_trtInfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# 路径根据实际情况修改！
set(TENSORRT_ROOT /usr/local/TensorRT-8.6.1.6)
set(CUDA_ROOT /usr/local/cuda-11.8)

find_package(OpenCV REQUIRED)

include_directories(
	include
	${TENSORRT_ROOT}/include
	${CUDA_ROOT}/include
	${OpenCV_INCLUDE_DIRS}
)

link_directories(
        ${TENSORRT_ROOT}/lib
        ${CUDA_ROOT}/lib64 )

add_executable(mimo_unet_trtInfer src/main.cpp src/mimo_unet.cpp)

target_link_libraries(mimo_unet_trtInfer nvinfer cudart ${OpenCV_LIBS})

target_compile_options(mimo_unet_trtInfer PRIVATE -Wno-deprecated-declarations)