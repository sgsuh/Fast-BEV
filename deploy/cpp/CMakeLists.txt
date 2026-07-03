# Fast-BEV C++ TensorRT inference (nuScenes app).
#
# Ported from ThorDrive's revolt.fast-bev-infer (Jetson Orin / sm_87) to the
# Fast-BEV deploy env: x86_64, CUDA 11.3, TensorRT 8.6.1, RTX 4070 (Ada/sm_89).
# The DDS/Lucid camera path (apps/fastdds.cpp + external/thor) is dropped; only
# the self-contained nuScenes app is built. The custom TRT plugins
# (project_2d_to_3d / emc) are compiled straight into this executable, so the
# engine is built/loaded here without the mmdeploy plugin .so.
cmake_minimum_required(VERSION 3.18)
project(fastbev VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# nvcc 11.3 cannot emit sm_89; build sm_86 SASS + compute_86 PTX so the driver
# JITs to the 4070's sm_89 (same PTX-forward-compat trick as the rest of the repo).
if(NOT CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES "86-real;86-virtual")
endif()

find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Qhull REQUIRED)

file(GLOB CPP_SOURCES "src/*.cpp")
file(GLOB CUDA_SOURCES "src/*.cu")

include_directories(include
	${OpenCV_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	${TensorRT_INCLUDE_DIRS}
	${YAML_CPP_INCLUDE_DIRS})

add_executable(fastbev
	apps/nuscenes.cpp
	${CPP_SOURCES}
	${CUDA_SOURCES})

# Qhull's reentrant lib is `qhull_r` from a source build; apt's libqhull-dev ships
# it as `qhullstatic_r` / `libqhull_r`. Prefer the imported target when present.
set(_QHULL_LIB qhull_r)
if(TARGET Qhull::qhull_r)
	set(_QHULL_LIB Qhull::qhull_r)
endif()

target_link_libraries(fastbev
	${OpenCV_LIBS}
	${CUDA_LIBRARIES}
	${TensorRT_LIBRARIES}
	${YAML_CPP_LIBRARIES}
	Eigen3::Eigen
	OpenMP::OpenMP_CXX
	cublas
	curand
	${_QHULL_LIB})
