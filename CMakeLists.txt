cmake_minimum_required(VERSION 3.10)
project(CUDA_11_TOY LANGUAGES CUDA CXX)

# C++ 11 标准编译
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 11)

# 包含CUDA 11 库文件
include(CheckLanguage)
check_language(CUDA)

set(BIN_DIR ${PROJECT_SOURCE_DIR}/bin)

# enable_language(CUDA)

message(STATUS "cuda version: " ${CUDA_VERSION_STRING})

# 控制输出的二进制文件位置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR})

# 加入要输出的可执行文件
add_executable(cuda_add cudaAdd/add.cu)
add_executable(cuda_deviceprop cudaDeviceProp/deviceprop.cu)
add_executable(cuda_dotproduct cudaDotProduct/dotproduct.cu)