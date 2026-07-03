#pragma once

#include "trt_plugin_base.hpp"

class TRTProject2DTo3D: public TRTPluginBase {
   public:
	TRTProject2DTo3D(const std::string& name, int out_dim_0, int out_dim_1, int out_dim_2, int out_dim_3);
	TRTProject2DTo3D(const std::string name, const void* data, size_t length);
	TRTProject2DTo3D() = delete;

	~TRTProject2DTo3D() TRT_NOEXCEPT override = default;

	nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override;
	nvinfer1::DimsExprs            getOutputDimensions(int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs, nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT override;

	bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* io_desc, int nb_inputs, int nb_outputs) TRT_NOEXCEPT override;

	void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nb_inputs, const nvinfer1::DynamicPluginTensorDesc* out, int nb_outputs) TRT_NOEXCEPT override;

	size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nb_inputs, const nvinfer1::PluginTensorDesc* outputs, int nb_outputs) const TRT_NOEXCEPT override;

	int enqueue(const nvinfer1::PluginTensorDesc* input_desc, const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;

	nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* input_types, int nb_inputs) const TRT_NOEXCEPT override;

	const char* getPluginType() const TRT_NOEXCEPT override;
	const char* getPluginVersion() const TRT_NOEXCEPT override;

	int getNbOutputs() const TRT_NOEXCEPT override;

	size_t getSerializationSize() const TRT_NOEXCEPT override;

	void serialize(void* buffer) const TRT_NOEXCEPT override;

   private:
	int out_dim_0_;
	int out_dim_1_;
	int out_dim_2_;
	int out_dim_3_;

	bool                 init_;
	std::vector<int>     n_voxels_cpu_;
	std::shared_ptr<int> lut_;
};

class TRTProject2DTo3DCreator: public TRTPluginCreatorBase {
   public:
	TRTProject2DTo3DCreator();

	~TRTProject2DTo3DCreator() TRT_NOEXCEPT override = default;

	const char* getPluginName() const TRT_NOEXCEPT override;
	const char* getPluginVersion() const TRT_NOEXCEPT override;

	nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT override;
	nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serial_data, size_t serial_length) TRT_NOEXCEPT override;
};
