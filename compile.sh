#!/bin/bash

DEBUG=$1

# If DEBUG is not empty, then build with debug mode
if [ ! -z "$DEBUG" ];
then
    FLAGS="-DCMAKE_BUILD_TYPE=Debug"
else
    FLAGS="-DCMAKE_BUILD_TYPE=Release"
fi

if [ ! -d "bin" ];
then
    mkdir bin
fi

if [ ! -d "build" ];
then
    mkdir build
fi

cd build/
cmake .. $FLAGS

make