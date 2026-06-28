#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <memory>

#include <cuda_runtime_api.h>

#include <torch/torch.h>
#include <torch/extension.h>

#include "view_transform.h"



FeatureMeta configToMeta() {
    FeatureMeta meta;

    meta.n_img_ = 6;
    meta.height_ = 256;
    meta.width_ = 704;
    meta.stride_ = 4;
    meta.n_ch_ = 64;
    meta.n_vx_ = 200 + 2 * 125;
    meta.n_vy_ = 200 + 2 * 125;
    meta.n_vz_ = 4;
    meta.vs_x_ = 0.5f;
    meta.vs_y_ = 0.5f;
    meta.vs_z_ = 1.5f;
    meta.origin_x_ = 0.0f;
    meta.origin_y_ = 0.0f;
    meta.origin_z_ = -1.0f;

    return meta;
}

const std::string getInputName(int i) {
    switch(i) {
        case 0:
            return std::string("backbone_output");
        default:
            std::cerr << "Invalid index given in getInputName: " << i << std::endl;

            return std::string();
    }
}

std::vector<int> getInputDims(FeatureMeta& meta, int i) {
    std::vector<int> input_size;

    switch(i) {
        case 0:
            // Backbone output
            input_size.push_back(meta.n_img_);
            input_size.push_back(meta.n_ch_);
            input_size.push_back(meta.height_ / meta.stride_);
            input_size.push_back(meta.width_ / meta.stride_);

            break;
        default:
            // Do nothing
            std::cerr << "Invalid index given in getInputDims: " << i << std::endl;

            break;
    }

    return input_size;
}

const std::string getOutputName(int i) {
    switch(i) {
        case 0:
            return std::string("bulk_bev_feature");
        default:
            std::cerr << "Invalid index given in getOutputName: " << i << std::endl;

            return std::string();
    }
}

const std::vector<int> getOutputDims(FeatureMeta& meta, int n_ts, int i) {
    std::vector<int> output_size;

    switch(i) {
        case 0:
            // Padded & un-compensated version of inpupt of neck_3d (bulk bev feature)
            output_size.push_back(n_ts);
            output_size.push_back(meta.n_ch_);
            output_size.push_back(meta.n_vx_);
            output_size.push_back(meta.n_vy_);
            output_size.push_back(meta.n_vz_);

            break;
        default:
            // Do nothing
            std::cerr << "Invalid index given in getOutputDims: " << i << std::endl;

            break;
    }

    return output_size;
}

const int getOutputSize(FeatureMeta& meta, int n_ts, int i) {
    if(i < 0 || i >= 1) {
        std::cerr << "Invalid index given in getOutputSize: " << i << std::endl;

        return 0;
    } else {
        int output_size = 1;
        std::vector<int> output_dims = getOutputDims(meta, n_ts, i);

        for(unsigned int axis = 0; axis < output_dims.size(); ++axis) {
            output_size *= output_dims[axis];
        }

        return output_size;
    }
}

void deviceMemoryAlloc(FeatureMeta& meta, int n_ts, int* lut_coord_idx, int* lut_img_idx) {
    // Display input and output list
    unsigned int n_in = 1;
    unsigned int n_out = 1;

    

    // Allocating GPU memories for the look-up table
    const unsigned int n_voxels = meta.n_vx_ * meta.n_vy_ * meta.n_vz_;

    

    GPU_CHECK(cudaMalloc((void**)&(lut_coord_idx), n_voxels * sizeof(int)));
    GPU_CHECK(cudaMalloc((void**)&(lut_img_idx), n_voxels * sizeof(int)));
}

void deviceMemoryDealloc(int* lut_coord_idx, int* lut_img_idx) {
    // Deallocating look-up table
    GPU_CHECK(cudaFree(lut_coord_idx));
    GPU_CHECK(cudaFree(lut_img_idx));
}

void vt_builder(at::Tensor fv_feat, at::Tensor lidar2img_host, at::Tensor vt_output) {
    FeatureMeta meta = configToMeta();
    int n_ts = 4;
    unsigned char buff_tail_idx = 0;
    std::string module_name = "view_transformer";

    float* out_buff = vt_output.data_ptr<float>();

    // Preparing output buffering
    

    int* lut_coord_idx;
    int* lut_img_idx;

    

    // Allocating GPU memories
    

    const unsigned int n_voxels = meta.n_vx_ * meta.n_vy_ * meta.n_vz_;

    GPU_CHECK(cudaMalloc((void**)&lut_coord_idx, n_voxels * sizeof(int)));
    GPU_CHECK(cudaMalloc((void**)&lut_img_idx, n_voxels * sizeof(int)));

    std::shared_ptr<int> lut_coord;
    std::shared_ptr<int> lut_img;

    lut_coord.reset(lut_coord_idx, [](int* p) {
        GPU_CHECK(cudaFree(p));
    });

    lut_img.reset(lut_img_idx, [](int* p) {
        GPU_CHECK(cudaFree(p));
    });

    // Construct LUT using lidar2img
    

    
    buildLUT(meta, lidar2img_host.data_ptr<float>(), lut_img.get(), lut_coord.get());

    

    // View transform core
    
    backproject(meta, fv_feat.data_ptr<float>(), out_buff, lut_img.get(), lut_coord.get());

    
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vt_builder", &vt_builder, "vt_builder");
}