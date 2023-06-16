#!/bin/bash

clang++ -std=c++11 \
  -I/opt/homebrew/Cellar/opencv/4.7.0_2/include/opencv4 \
  -L/opt/homebrew/Cellar/opencv/4.7.0_2/lib \
  -lopencv_core \
  -lopencv_highgui \
  -lopencv_imgcodecs \
  -lopencv_stitching \
  -lopencv_video \
  -lopencv_videoio \
  -lopencv_imgproc \
  -g \
  ./src/opencv-stitcher.cpp \
  -o ./bin/opencv-stitcher
