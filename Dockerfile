# Docker environment for machine learning research
FROM nvidia/cuda:7.5-cudnn4-devel
ENV DEBIAN_FRONTEND noninteractive
MAINTAINER Matt Liston <s1580449@ed.ac.uk>

# prerequisites
RUN apt-get update
RUN apt-get install -y build-essential cmake git yasm pkg-config
RUN apt-get install -y eog vim mplayer2 emacs24 libav-tools
RUN apt-get install -y libjpeg-dev libtiff5-dev libjpeg8-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python-tk
RUN apt-get install -y python-pip
RUN apt-get install -y --upgrade python-dev python-numpy python-scipy python-pip python-nose g++ libopenblas-dev git
RUN apt-get install -y libatlas-base-dev libboost-all-dev libprotobuf-dev libgoogle-glog-dev libgflags-dev protobuf-compiler libhdf5-serial-dev libleveldb-dev libsnappy-dev liblmdb-dev
RUN apt-get install -y gfortran
RUN apt-get install -y hdf5-tools
RUN apt-get install -y curl

# python libraries
RUN pip install --upgrade pip
RUN pip install --upgrade cython
RUN pip install --upgrade scikit-image
RUN pip install --upgrade scipy
RUN pip install h5py
RUN pip install scikit-learn
RUN pip install matplotlib
RUN pip install protobuf
ENV PYTHONPATH /usr/local/python

# BUILD CAFFE
RUN git clone -b rc3 https://github.com/BVLC/caffe.git /tmp/caffe
RUN cd /tmp/caffe && mkdir build && cd build && cmake -DUSE_OPENCV:BOOL=OFF -DCMAKE_INSTALL_PREFIX=/usr/local .. && make install -j && make clean
RUN ldconfig
