cmake_minimum_required(VERSION 3.10)
project(CUDA_11_TOY LANGUAGES CUDA CXX)

# C++ 11 标准编译
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 11)

# 包含CUDA 11 库文件
include(CheckLanguage)
check_language(CUDA)

find_package(CUDA REQUIRED)
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
message(STATUS "cuda dir: " ${CUDA_TOOLKIT_ROOT_DIR})

set(CMAKE_CUDA_ARCHITECTURES 75)
set(BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

# enable_language(CUDA)
# 包含头文件util.cuh
include_directories(${PROJECT_SOURCE_DIR}/include)

find_library(CUDA_NVTX_LIB 
    NAME nvToolsExt
    PATH ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES "lib64"
    DOC "Location of the CUDA Toolkit Extension (NVTX) library"
    NO_DEFAULT_PATH)
link_directories(/usr/local/cuda/lib64)

# 控制输出的二进制文件位置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR})

# 加入要输出的可执行文件
set(EXE_OBJ_LIST "cudaAdd;cudaDeviceProp;cudaDotProduct;cudaPinMem;\
cudaAsync;cudaZeroCopy;elementwise;cudaMatMul;cudaReduce")

foreach(EXE_OBJ ${EXE_OBJ_LIST})
    set(SRC_LIST)
    aux_source_directory(${SOURCE_DIR}/${EXE_OBJ} SRC_LIST)
    add_executable(${EXE_OBJ} ${SRC_LIST})
    target_link_libraries(${EXE_OBJ} nvToolsExt)
endforeach()
