#!/bin/bash  

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <cuda_executable> <nsys_output_file>"
    exit 1
fi


EXECUTABLE=$1
NSYS_OUTPUT_REP=$2.nsys-rep
NSYS_OUTPUT_CSV=$2.csv

  
# 执行 ncu 并获取性能信息  
nsys profile -f true  -o $2 -d 1000 -c cudaProfilerApi -t cuda,cudnn,cublas,nvtx,osrt,nvvideo \
--cudabacktrace all:1 --cuda-memory-usage true  --stats true $EXECUTABLE

nsys stats --force-overwrite true --report gputrace -q -f csv -o $NSYS_OUTPUT_CSV $NSYS_OUTPUT_REP
