#include "trt_emc.h"
#include "trt_serialize.hpp"

namespace {
static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"emc"};
}  // namespace

TRTEmc::TRTEmc(const std::string& name, int out_dim_0, int out_dim_1, int out_dim_2, int out_dim_3, int out_dim_4)
    : TRTPluginBase(name)
    , out_dim_0_(out_dim_0)
    , out_dim_1_(out_dim_1)
    , out_dim_2_(out_dim_2)
    , out_dim_3_(out_dim_3)
    , out_dim_4_(out_dim_4) {
	config_ = configToMeta();
	n_ts_   = 4;
}

TRTEmc::TRTEmc(const std::string name, const void* data, size_t length)
    : TRTPluginBase(name) {
	deserializeValue(&data, &length, &out_dim_0_);
	deserializeValue(&data, &length, &out_dim_1_);
	deserializeValue(&data, &length, &out_dim_2_);
	deserializeValue(&data, &length, &out_dim_3_);
	deserializeValue(&data, &length, &out_dim_4_);
}

ReMapMeta TRTEmc::configToMeta() {
	ReMapMeta meta;

	meta.n_vx_     = 200;
	meta.n_vy_     = 200;
	meta.n_vz_     = 4;
	meta.m_vx_     = 125;
	meta.m_vy_     = 125;
	meta.n_ch_     = 64;
	meta.vs_x_     = 0.5f;
	meta.vs_y_     = 0.5f;
	meta.vs_z_     = 1.5f;
	meta.origin_x_ = 0.0f;
	meta.origin_y_ = 0.0f;
	meta.origin_z_ = -1.0f;

	return meta;
}

std::vector<int> TRTEmc::getInputDims(ReMapMeta& config, int n_ts, int i) {
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

int TRTEmc::getInputSize(ReMapMeta& config, int n_ts, int i) {
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

nvinfer1::IPluginV2DynamicExt* TRTEmc::clone() const TRT_NOEXCEPT {
	TRTEmc* plugin = new TRTEmc(layer_name_, out_dim_0_, out_dim_1_, out_dim_2_, out_dim_3_, out_dim_4_);

	plugin->setPluginNamespace(getPluginNamespace());

	return plugin;
}

nvinfer1::DimsExprs TRTEmc::getOutputDimensions(int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs, nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
	nvinfer1::DimsExprs ret;

	ret.nbDims = 5;

	ret.d[0] = expr_builder.constant(out_dim_0_);
	ret.d[1] = expr_builder.constant(out_dim_1_);
	ret.d[2] = expr_builder.constant(out_dim_2_);
	ret.d[3] = expr_builder.constant(out_dim_3_);
	ret.d[4] = expr_builder.constant(out_dim_4_);

	return ret;
}

bool TRTEmc::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* io_desc, int nb_inputs, int nb_outputs) TRT_NOEXCEPT {
	return (io_desc[pos].type == nvinfer1::DataType::kFLOAT && io_desc[pos].format == nvinfer1::TensorFormat::kLINEAR);
}

void TRTEmc::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs, int nb_inputs, const nvinfer1::DynamicPluginTensorDesc* outputs, int nb_outputs) TRT_NOEXCEPT {
	ASSERT(nb_inputs == 2);
	ASSERT(nb_outputs == 1);
}

size_t TRTEmc::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nb_inputs, const nvinfer1::PluginTensorDesc* outputs, int nb_outputs) const TRT_NOEXCEPT {
	return 0;
}

int TRTEmc::enqueue(const nvinfer1::PluginTensorDesc* input_desc, const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs, void* const* outputs, void* work_space, cudaStream_t stream) TRT_NOEXCEPT {
	float* stack_buff_dev = (float*)outputs[0];
	float* tf_dev         = (float*)inputs[1];
	float* vt_output      = (float*)inputs[0];

	// Emc core part
	const int nrof_voxels      = config_.n_vx_ * config_.n_vy_ * config_.n_vz_;
	const int single_buff_size = config_.n_ch_ * nrof_voxels;

	// Just crop for current frame
	cudaMemset(stack_buff_dev, 0.0f, single_buff_size * sizeof(float));

	simpleCrop(config_, vt_output, stack_buff_dev);

	// Rotate & translate, then crop for previous frames
	for(unsigned int ts = 1; ts < n_ts_; ++ts) {
		// Transform and crop
		cudaMemset(stack_buff_dev + single_buff_size * ts, 0.0f, single_buff_size * sizeof(float));
		transformCrop(config_, tf_dev, vt_output + ts * getInputSize(config_, 1, 0), stack_buff_dev + single_buff_size * ts);
	}

	return 0;
}

nvinfer1::DataType TRTEmc::getOutputDataType(int index, const nvinfer1::DataType* input_types, int nb_inputs) const TRT_NOEXCEPT {
	return input_types[0];
}

const char* TRTEmc::getPluginType() const TRT_NOEXCEPT {
	return PLUGIN_NAME;
}

const char* TRTEmc::getPluginVersion() const TRT_NOEXCEPT {
	return PLUGIN_VERSION;
}

int TRTEmc::getNbOutputs() const TRT_NOEXCEPT {
	return 1;
}

size_t TRTEmc::getSerializationSize() const TRT_NOEXCEPT {
	return serializedSize(out_dim_0_) + serializedSize(out_dim_1_) + serializedSize(out_dim_2_) + serializedSize(out_dim_3_) + serializedSize(out_dim_4_);
}

void TRTEmc::serialize(void* buffer) const TRT_NOEXCEPT {
	serializeValue(&buffer, out_dim_0_);
	serializeValue(&buffer, out_dim_1_);
	serializeValue(&buffer, out_dim_2_);
	serializeValue(&buffer, out_dim_3_);
	serializeValue(&buffer, out_dim_4_);
}

TRTEmcCreator::TRTEmcCreator() {
	plugin_attributes_ = std::vector<nvinfer1::PluginField>({nvinfer1::PluginField("out_dim_0"), nvinfer1::PluginField("out_dim_1"), nvinfer1::PluginField("out_dim_2"), nvinfer1::PluginField("out_dim_3"), nvinfer1::PluginField("out_dim_4")});
	fc_.nbFields       = plugin_attributes_.size();
	fc_.fields         = plugin_attributes_.data();
}

const char* TRTEmcCreator::getPluginName() const TRT_NOEXCEPT {
	return PLUGIN_NAME;
}

const char* TRTEmcCreator::getPluginVersion() const TRT_NOEXCEPT {
	return PLUGIN_VERSION;
}

nvinfer1::IPluginV2* TRTEmcCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT {
	int out_dim_0 = 1;
	int out_dim_1 = 256;
	int out_dim_2 = 200;
	int out_dim_3 = 200;
	int out_dim_4 = 4;

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

		if(field_name.compare("out_dim_4") == 0) {
			out_dim_4 = static_cast<const int*>(fc->fields[i].data)[0];
		}
	}

	ASSERT(out_dim_0 > 0);
	ASSERT(out_dim_1 > 0);
	ASSERT(out_dim_2 > 0);
	ASSERT(out_dim_3 > 0);
	ASSERT(out_dim_4 > 0);

	TRTEmc* plugin = new TRTEmc(name, out_dim_0, out_dim_1, out_dim_2, out_dim_3, out_dim_4);

	plugin->setPluginNamespace(getPluginNamespace());

	return plugin;
}

nvinfer1::IPluginV2* TRTEmcCreator::deserializePlugin(const char* name, const void* serial_data, size_t serial_length) TRT_NOEXCEPT {
	auto plugin = new TRTEmc(name, serial_data, serial_length);

	plugin->setPluginNamespace(getPluginNamespace());

	return plugin;
}

REGISTER_TENSORRT_PLUGIN(TRTEmcCreator);
