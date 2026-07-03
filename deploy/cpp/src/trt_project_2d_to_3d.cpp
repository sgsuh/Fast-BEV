#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "trt_plugin_helper.hpp"
#include "trt_project_2d_to_3d.h"
#include "trt_project_2d_to_3d_kernel.h"
#include "trt_serialize.hpp"

namespace {
static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"project_2d_to_3d"};
}  // namespace

TRTProject2DTo3D::TRTProject2DTo3D(const std::string& name, int out_dim_0, int out_dim_1, int out_dim_2, int out_dim_3)
    : TRTPluginBase(name)
    , out_dim_0_(out_dim_0)
    , out_dim_1_(out_dim_1)
    , out_dim_2_(out_dim_2)
    , out_dim_3_(out_dim_3) {
	init_ = false;
	n_voxels_cpu_.resize(3);
}

TRTProject2DTo3D::TRTProject2DTo3D(const std::string name, const void* data, size_t length)
    : TRTPluginBase(name) {
	deserializeValue(&data, &length, &out_dim_0_);
	deserializeValue(&data, &length, &out_dim_1_);
	deserializeValue(&data, &length, &out_dim_2_);
	deserializeValue(&data, &length, &out_dim_3_);
}

nvinfer1::IPluginV2DynamicExt* TRTProject2DTo3D::clone() const TRT_NOEXCEPT {
	TRTProject2DTo3D* plugin = new TRTProject2DTo3D(layer_name_, out_dim_0_, out_dim_1_, out_dim_2_, out_dim_3_);

	plugin->setPluginNamespace(getPluginNamespace());

	return plugin;
}

nvinfer1::DimsExprs TRTProject2DTo3D::getOutputDimensions(int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs, nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
	nvinfer1::DimsExprs ret;

	ret.nbDims = 4;

	ret.d[0] = expr_builder.constant(out_dim_0_);
	ret.d[1] = expr_builder.constant(out_dim_1_);
	ret.d[2] = expr_builder.constant(out_dim_2_);
	ret.d[3] = expr_builder.constant(out_dim_3_);

	return ret;
}

bool TRTProject2DTo3D::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* io_desc, int nb_inputs, int nb_outputs) TRT_NOEXCEPT {
	if(pos == 1) {
		return (io_desc[pos].type == nvinfer1::DataType::kINT32 && io_desc[pos].format == nvinfer1::TensorFormat::kLINEAR);
	} else {
		return (io_desc[pos].type == nvinfer1::DataType::kFLOAT && io_desc[pos].format == nvinfer1::TensorFormat::kLINEAR);
	}
}

void TRTProject2DTo3D::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs, int nb_inputs, const nvinfer1::DynamicPluginTensorDesc* outputs, int nb_outputs) TRT_NOEXCEPT {
	ASSERT(nb_inputs == 5);
	ASSERT(nb_outputs == 1);
}

size_t TRTProject2DTo3D::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nb_inputs, const nvinfer1::PluginTensorDesc* outputs, int nb_outputs) const TRT_NOEXCEPT {
	return 0;
}

int TRTProject2DTo3D::enqueue(const nvinfer1::PluginTensorDesc* input_desc, const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs, void* const* outputs, void* work_space, cudaStream_t stream) TRT_NOEXCEPT {
	nvinfer1::Dims feat_dims = input_desc[0].dims;

	int n_images   = feat_dims.d[0];
	int height     = feat_dims.d[2];
	int width      = feat_dims.d[3];
	int n_channels = feat_dims.d[1];

	if(!init_) {
		cudaMemcpy(n_voxels_cpu_.data(), (int*)inputs[1], 3 * sizeof(int), cudaMemcpyDeviceToHost);
		int  nrof_elements = n_voxels_cpu_[0] * n_voxels_cpu_[1] * n_voxels_cpu_[2];
		int* lut_p;
		cudaMalloc((void**)&lut_p, nrof_elements * sizeof(int));
		lut_.reset(lut_p, [](int* p) {
			cudaFree(p);
		});
		std::vector<int> tmp1(nrof_elements, -1);
		cudaMemcpy(lut_.get(), tmp1.data(), nrof_elements * sizeof(int), cudaMemcpyHostToDevice);
		buildLutGpu(n_voxels_cpu_, (float*)inputs[2], (float*)inputs[3], (float*)inputs[4], lut_.get(), n_images, height, width, stream);

		init_ = true;
	}

	nvinfer1::Dims out_dims = output_desc[0].dims;

	int n_size = out_dims.d[0] * out_dims.d[1] * out_dims.d[2] * out_dims.d[3];

	setVolumeZero(n_size, (float*)outputs[0], stream);

	backprojectLutGpu((float*)inputs[0], lut_.get(), (float*)outputs[0], n_images, n_channels, n_voxels_cpu_, height, width, stream);

	return 0;
}

nvinfer1::DataType TRTProject2DTo3D::getOutputDataType(int index, const nvinfer1::DataType* input_types, int nb_inputs) const TRT_NOEXCEPT {
	return input_types[0];
}

const char* TRTProject2DTo3D::getPluginType() const TRT_NOEXCEPT {
	return PLUGIN_NAME;
}

const char* TRTProject2DTo3D::getPluginVersion() const TRT_NOEXCEPT {
	return PLUGIN_VERSION;
}

int TRTProject2DTo3D::getNbOutputs() const TRT_NOEXCEPT {
	return 1;
}

size_t TRTProject2DTo3D::getSerializationSize() const TRT_NOEXCEPT {
	return serializedSize(out_dim_0_) + serializedSize(out_dim_1_) + serializedSize(out_dim_2_) + serializedSize(out_dim_3_);
}

void TRTProject2DTo3D::serialize(void* buffer) const TRT_NOEXCEPT {
	serializeValue(&buffer, out_dim_0_);
	serializeValue(&buffer, out_dim_1_);
	serializeValue(&buffer, out_dim_2_);
	serializeValue(&buffer, out_dim_3_);
}

TRTProject2DTo3DCreator::TRTProject2DTo3DCreator() {
	plugin_attributes_ = std::vector<nvinfer1::PluginField>({nvinfer1::PluginField("out_dim_0"), nvinfer1::PluginField("out_dim_1"), nvinfer1::PluginField("out_dim_2"), nvinfer1::PluginField("out_dim_3")});
	fc_.nbFields       = plugin_attributes_.size();
	fc_.fields         = plugin_attributes_.data();
}

const char* TRTProject2DTo3DCreator::getPluginName() const TRT_NOEXCEPT {
	return PLUGIN_NAME;
}

const char* TRTProject2DTo3DCreator::getPluginVersion() const TRT_NOEXCEPT {
	return PLUGIN_VERSION;
}

nvinfer1::IPluginV2* TRTProject2DTo3DCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT {
	int out_dim_0 = 64;
	int out_dim_1 = 450;
	int out_dim_2 = 450;
	int out_dim_3 = 4;

	for(int i = 0; i < fc->nbFields; ++i) {
		if(fc->fields[i].data == nullptr) {
			continue;
		}

		std::string field_name(fc->fields[i].name);

		if(field_name.compare("out_dim_0") == 0) {
			out_dim_0 = static_cast<const int*>(fc->fields[i].data)[0];
		}

		if(field_name.compare("out_dim_1") == 0) {
			out_dim_1 = static_cast<const int*>(fc->fields[i].data)[0];
		}

		if(field_name.compare("out_dim_2") == 0) {
			out_dim_2 = static_cast<const int*>(fc->fields[i].data)[0];
		}

		if(field_name.compare("out_dim_3") == 0) {
			out_dim_3 = static_cast<const int*>(fc->fields[i].data)[0];
		}
	}

	ASSERT(out_dim_0 > 0);
	ASSERT(out_dim_1 > 0);
	ASSERT(out_dim_2 > 0);
	ASSERT(out_dim_3 > 0);

	TRTProject2DTo3D* plugin = new TRTProject2DTo3D(name, out_dim_0, out_dim_1, out_dim_2, out_dim_3);

	plugin->setPluginNamespace(getPluginNamespace());

	return plugin;
}

nvinfer1::IPluginV2* TRTProject2DTo3DCreator::deserializePlugin(const char* name, const void* serial_data, size_t serial_length) TRT_NOEXCEPT {
	auto plugin = new TRTProject2DTo3D(name, serial_data, serial_length);

	plugin->setPluginNamespace(getPluginNamespace());

	return plugin;
}

REGISTER_TENSORRT_PLUGIN(TRTProject2DTo3DCreator);
