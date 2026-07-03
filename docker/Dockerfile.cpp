# =============================================================================
# Fast-BEV C++ TensorRT inference build environment (Stage C)
#
# Extends fastbev-deploy:trt (TensorRT 8.6.1 + cuDNN 8.9 on the cu113 stack) with
# the C++ toolchain and libraries needed to build deploy/cpp (the ported nuScenes
# inference app): OpenCV, Eigen, yaml-cpp, Qhull, OpenMP, and gcc-10 as the CUDA
# host compiler (CUDA 11.3 supports gcc <= 10; the default focal gcc-9 is too old
# for some of the C++ used, gcc-11 too new for nvcc 11.3).
# =============================================================================
FROM fastbev-deploy:trt

ENV DEBIAN_FRONTEND=noninteractive

# C++ build toolchain + libraries. libyaml-cpp-dev/libeigen3-dev/libopencv-dev are
# the focal (20.04) versions; opencv 4.2 is plenty for imread/imwrite/draw here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-10 g++-10 \
        libopencv-dev libeigen3-dev libyaml-cpp-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# nvcc must call gcc-10 (not focal's default gcc-9) as the host compiler.
RUN ln -sf /usr/bin/gcc-10 /usr/local/cuda/bin/gcc \
    && ln -sf /usr/bin/g++-10 /usr/local/cuda/bin/g++
ENV CC=/usr/bin/gcc-10 \
    CXX=/usr/bin/g++-10 \
    CUDAHOSTCXX=/usr/bin/g++-10

# Qhull from source: provides the reentrant qhull_r lib + QhullConfig.cmake that
# find_package(Qhull) needs (focal's libqhull-dev predates the cmake config).
RUN git clone --depth 1 -b 2020.2 https://github.com/qhull/qhull.git /tmp/qhull \
    && cmake -S /tmp/qhull -B /tmp/qhull/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/qhull/build --target install -j"$(nproc)" \
    && ldconfig \
    && rm -rf /tmp/qhull

# This is a pure C++ build/run image -- skip the mmdet3d editable install that the
# base entrypoint does (only the python deploy path needs it).
WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
