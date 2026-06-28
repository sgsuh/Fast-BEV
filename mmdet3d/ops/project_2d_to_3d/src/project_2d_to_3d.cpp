#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda_runtime_api.h>


void build_lut_gpu(std::vector<int> n_voxels, float* voxel_size_dev, float* origin_dev, float* projection, int* lut, int n_images, int height, int width);

void backproject_lut_gpu(float* features_bev, int* lut_dev, float* volume_dev, int n_images, int n_channels, std::vector<int> n_voxels, int height, int width);

void build_lut_cpu(std::vector<int> n_voxels, at::Tensor voxel_size, at::Tensor origin, at::Tensor projection, int n_images, int height, int width, int n_channels, int* lutp);

void backproject_lut_cpu(at::Tensor features, int* lutp, float* volumep, std::vector<int> n_voxels);

// n_voxels, voxel_size, origin, projection, volume

void project_2d_to_3d(at::Tensor _features, at::Tensor _n_voxels, at::Tensor _voxel_size, at::Tensor _origin, at::Tensor _projection, at::Tensor _volume) {

    int n_images = _features.size(0);
    int height = _features.size(2);
    int width = _features.size(3);
    int n_channels = _features.size(1);

    float* features_bev = _features.data_ptr<float>();

    int* n_voxels = _n_voxels.data_ptr<int>();
    float* voxel_size_dev = _voxel_size.data_ptr<float>();
    float* origin_dev = _origin.data_ptr<float>();
    float* projection = _projection.data_ptr<float>();
    float* volume_dev = _volume.data_ptr<float>();



    std::vector<int> n_voxels_cpu(3);



    cudaMemcpy(n_voxels_cpu.data(), n_voxels, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    size_t nrof_elements = n_voxels_cpu[0] * n_voxels_cpu[1] * n_voxels_cpu[2];

    int* lut_p;

    cudaMalloc((void**)&lut_p, nrof_elements * sizeof(int));

    std::shared_ptr<int> lut;

    lut.reset(lut_p, [](int* p) {
        cudaFree(p);
    });

    std::vector<int> tmp1(nrof_elements, -1);

    cudaMemcpy(lut.get(), tmp1.data(), nrof_elements * sizeof(int), cudaMemcpyHostToDevice);

    build_lut_gpu(n_voxels_cpu, voxel_size_dev, origin_dev, projection, lut.get(), n_images, height, width);
    backproject_lut_gpu(features_bev, lut.get(), volume_dev, n_images, n_channels, n_voxels_cpu, height, width);


}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project_2d_to_3d", &project_2d_to_3d, "project_2d_to_3d");
}