#!/bin/bash
set -e

# Set OpenCV version
OPENCV_VERSION="4.5.4"

# Install dependencies
sudo apt update
sudo apt install -y \
    build-essential cmake git unzip pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev v4l-utils \
    libxvidcore-dev libx264-dev \
    libgtk-3-dev \
    libcanberra-gtk* \
    libatlas-base-dev gfortran \
    python3-dev python3-numpy \
    libtbb2 libtbb-dev \
    libdc1394-22-dev

# Get OpenCV source
cd ~
git clone --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git
git clone --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git

# Build OpenCV
cd ~/opencv
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D ENABLE_NEON=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D CUDA_ARCH_BIN="8.7" \
      -D WITH_GSTREAMER=ON \
      -D WITH_LIBV4L=ON \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D BUILD_opencv_python3=ON \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF ..

make -j$(nproc)
sudo make install
sudo ldconfig
