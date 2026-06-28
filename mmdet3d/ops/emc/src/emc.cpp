#include <vector>

#include <cuda_runtime_api.h>

#include <torch/torch.h>
#include <torch/extension.h>

#include "emc_utils.h"

std::vector<int> getInputDims(ReMapMeta& config, int n_ts, int i) {
    std::vector<int> input_size;

    switch(i) {
        case 0:
            // Padded bev feature
            input_size.push_back(n_ts);
            input_size.push_back(config.n_ch_);
            input_size.push_back(config.n_vx_ + 2 * config.m_vx_);
            input_size.push_back(config.n_vy_ + 2 * config.m_vy_);
            input_size.push_back(config.n_vz_);

            break;
        default:
            break;
    }

    return input_size;
}

std::vector<int> getOutputDims(ReMapMeta& config, int n_ts, int i) {
    std::vector<int> output_size;

    switch(i) {
        case 0:
            output_size.push_back(1);
            output_size.push_back(n_ts * config.n_ch_);
            output_size.push_back(config.n_vx_);
            output_size.push_back(config.n_vy_);
            output_size.push_back(config.n_vz_);

            break;
        default:
            break;
    }

    return output_size;
}

int getOutputSize(ReMapMeta& config, int n_ts, int i) {
    if(i < 0 || i >= 1) {
        return 0;
    } else {
        int output_size = 1;
        std::vector<int> output_dims = getOutputDims(config, n_ts, i);

        for(unsigned int axis = 0; axis < output_dims.size(); ++axis) {
            output_size *= output_dims[axis];
        }

        return output_size;
    }
}

void deviceMemoryAlloc(ReMapMeta& meta, int n_ts, float* stack_buff_dev, float* tf_dev) {
    unsigned int n_in = 1;
    unsigned int n_out = 1;

    for(unsigned int i = 0; i < n_in; ++i) {
        std::vector<int> input_dims = getInputDims(meta, n_ts, i);
    }

    for(unsigned int i = 0; i < n_out; ++i) {
        std::vector<int> output_dims = getOutputDims(meta, n_ts, i);
    }

    // Allocating GPU memories for output buffering
    GPU_CHECK(cudaMalloc((void**)&stack_buff_dev, getOutputSize(meta, n_ts, 0) * sizeof(float)));
    GPU_CHECK(cudaMalloc((void**)&tf_dev, 4 * 4 * sizeof(float)));
}

ReMapMeta configToMeta() {
    ReMapMeta meta;

    meta.n_vx_ = 200;
    meta.n_vy_ = 200;
    meta.n_vz_ = 4;
    meta.m_vx_ = 125;
    meta.m_vy_ = 125;
    meta.n_ch_ = 64;
    meta.vs_x_ = 0.5f;
    meta.vs_y_ = 0.5f;
    meta.vs_z_ = 1.5f;
    meta.origin_x_ = 0.0f;
    meta.origin_y_ = 0.0f;
    meta.origin_z_ = -1.0f;

    return meta;
}

int getInputSize(ReMapMeta& config, int n_ts, int i) {
    if(i < 0 || i >= 1) {
        return 0;
    } else {
        int input_size = 1;

        std::vector<int> input_dims = getInputDims(config, n_ts, i);

        for(unsigned int axis = 0; axis < input_dims.size(); ++axis) {
            input_size *= input_dims[axis];
        }

        return input_size;

    }
}

void emc(at::Tensor vt_output, at::Tensor tf_vec, at::Tensor bev_feat) {
    ReMapMeta config = configToMeta();
    int n_ts = 4;

    float* stack_buff_dev = bev_feat.data_ptr<float>();
    float* tf_dev = tf_vec.data_ptr<float>();

    

    // Emc core part
    const int nrof_voxels = config.n_vx_ * config.n_vy_ * config.n_vz_;
    const int single_buff_size = config.n_ch_ * nrof_voxels;

    // Just crop for current frame
    GPU_CHECK(cudaMemset(stack_buff_dev, 0.0f, single_buff_size * sizeof(float)));

    simpleCrop(config, vt_output.data_ptr<float>(), stack_buff_dev);

    // Rotate & translate, then crop for previous frames
    for(unsigned int ts = 1; ts < n_ts; ++ts) {
        

        // Transform and crop
        GPU_CHECK(cudaMemset(stack_buff_dev + single_buff_size * ts, 0.0f, single_buff_size * sizeof(float)));
        transformCrop(config, tf_dev, vt_output.data_ptr<float>() + ts * getInputSize(config, 1, 0), stack_buff_dev + single_buff_size * ts);
    }


}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("emc", &emc, "emc");
}