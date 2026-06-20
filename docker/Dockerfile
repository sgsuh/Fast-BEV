# =============================================================================
# Fast-BEV deploy environment for NVIDIA RTX 4070 (Ada / sm_89)
#
# The repo pins the legacy stack MMCV-full 1.4.0 / MMDet 2.14.0 / MMSeg 0.14.1,
# which only builds against PyTorch ~1.10 (CUDA 11.3). An Ada GPU normally needs
# CUDA 11.8+, but the 4070 runs this old stack via PTX forward-compatibility
# (the driver JIT-compiles compute_86 PTX to sm_89). Verified working: cuDNN
# conv, batchnorm, indexing and matmul all execute on the 4070.
#
# To keep that guarantee for the custom CUDA ops we COMPILE (MMCV + this repo's
# mmdet3d), we set TORCH_CUDA_ARCH_LIST with an explicit "+PTX" entry so the
# built .so files carry JIT-able PTX for sm_89.
# =============================================================================
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FORCE_CUDA=1 \
    # 7.0/7.5/8.0 cover Volta..Ampere data-center; 8.6+PTX gives Ada (sm_89) via JIT
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX" \
    MMCV_WITH_OPS=1

# --- system packages + python 3.8 -------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip \
        git ninja-build build-essential \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
        ca-certificates wget \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Old build backends need an old setuptools; pin pip toolchain up front.
RUN python -m pip install --upgrade "pip==22.3.1" "setuptools==59.5.0" "wheel==0.37.1" "Cython==0.29.33"

# --- numpy first so everything else links against the pinned ABI ------------
RUN pip install numpy==1.19.5

# --- PyTorch 1.10.1 + cu113 (matches MMCV 1.4.0; runs on Ada via PTX JIT) ----
RUN pip install \
        torch==1.10.1+cu113 \
        torchvision==0.11.2+cu113 \
        -f https://download.pytorch.org/whl/torch_stable.html

# --- MMCV-full 1.4.0 built FROM SOURCE so its CUDA ops include sm_89 PTX -----
# (prebuilt wheels may lack PTX and fail with "no kernel image" on the 4070)
RUN git clone -b v1.4.0 --depth 1 https://github.com/open-mmlab/mmcv.git /tmp/mmcv \
    && cd /tmp/mmcv \
    && pip install -r requirements/optional.txt || true \
    && pip install . --no-build-isolation -v \
    && cd / && rm -rf /tmp/mmcv

# --- remaining deploy/inference dependencies --------------------------------
COPY docker/requirements-deploy.txt /tmp/requirements-deploy.txt
RUN pip install --no-build-isolation -r /tmp/requirements-deploy.txt

# numpy can get bumped by transitive deps; force it back to the pinned ABI.
RUN pip install --no-deps --force-reinstall numpy==1.19.5

# MMCV/nuscenes-devkit pull opencv>=4.7, whose cv2/typing stub crashes on import
# (AttributeError: module 'cv2.dnn' has no attribute 'DictValue'). Collapse to a
# single clean headless build that predates that stub.
RUN pip uninstall -y opencv-python opencv-python-headless \
        opencv-contrib-python opencv-contrib-python-headless || true \
    && pip install --no-deps opencv-python-headless==4.5.5.64

WORKDIR /workspace
ENV MMDET3D=/workspace

# The repo (and its mmdet3d CUDA ops) is bind-mounted and built on first start.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
