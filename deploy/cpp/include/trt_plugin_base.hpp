#pragma once

#include <NvInferRuntime.h>
#include <NvInferVersion.h>

#include "trt_plugin_helper.hpp"

#define TRT_NOEXCEPT noexcept

class TRTPluginBase: public nvinfer1::IPluginV2DynamicExt {
   public:
	TRTPluginBase(const std::string& name)
	    : layer_name_(name) {
	}

	// IPluginV2 Methods
	const char* getPluginVersion() const TRT_NOEXCEPT override {
		return "1";
	}

	int initialize() TRT_NOEXCEPT override {
		return STATUS_SUCCESS;
	}

	void terminate() TRT_NOEXCEPT override {
	}

	void destroy() TRT_NOEXCEPT override {
		delete this;
	}

	void setPluginNamespace(const char* plugin_namespace) TRT_NOEXCEPT override {
		namespace_ = plugin_namespace;
	}

	const char* getPluginNamespace() const TRT_NOEXCEPT override {
		return namespace_.c_str();
	}

	virtual void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) TRT_NOEXCEPT override {
	}

	virtual size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const TRT_NOEXCEPT override {
		return 0;
	}

	virtual void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) TRT_NOEXCEPT override {
	}

   protected:
	const std::string layer_name_;
	std::string       namespace_;
};

class TRTPluginCreatorBase: public nvinfer1::IPluginCreator {
   public:
	const char* getPluginVersion() const TRT_NOEXCEPT override {
		return "1";
	}

	const nvinfer1::PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override {
		return &fc_;
	}

	void setPluginNamespace(const char* plugin_namespace) TRT_NOEXCEPT override {
		namespace_ = plugin_namespace;
	}

	const char* getPluginNamespace() const TRT_NOEXCEPT override {
		return namespace_.c_str();
	}

   protected:
	nvinfer1::PluginFieldCollection    fc_;
	std::vector<nvinfer1::PluginField> plugin_attributes_;
	std::string                        namespace_;
};
