#!/bin/bash

if [ ! -d "bin" ];
then
    mkdir bin
fi

if [ ! -d "build" ];
then
    mkdir build
fi

cd build/
cmake ..

make