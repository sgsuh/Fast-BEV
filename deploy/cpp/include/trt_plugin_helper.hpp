#pragma once

#include <NvInferRuntime.h>
#include <cudnn.h>
#include <iostream>
#include <stdexcept>

#define ASSERT(assertion)                                                          \
	{                                                                              \
		if(!(assertion)) {                                                         \
			std::cerr << "#assertion" << __FILE__ << "," << __LINE__ << std::endl; \
			abort();                                                               \
		}                                                                          \
	}

#define CUDASEERT(status_)                                                                                          \
	{                                                                                                               \
		auto s_ = status_;                                                                                          \
		if(s_ != cudaSuccess) {                                                                                     \
			std::cerr << __FILE__ << ", " << __LINE__ << ", " << s_ << ", " << cudaGetErrorString(s_) << std::endl; \
		}                                                                                                           \
	}

#define CUBLASASSERT(status_)                                                     \
	{                                                                             \
		auto s_ = status_;                                                        \
		if(s_ != CUBLAS_STATUS_SUCCESS) {                                         \
			std::cerr << __FILE__ << ", " << __LINE__ << ", " << s_ << std::endl; \
		}                                                                         \
	}

#define CUERRORMSG(status_)                                                       \
	{                                                                             \
		auto s_ = status;                                                         \
		if(s_ != 0)                                                               \
			std::cerr << __FILE__ << ", " << __LINE__ << ", " << s_ << std::endl; \
	}

#define CHECK(status)   \
	do {                \
		if(status != 0) \
			abort();    \
	} while(0)

#define ASSERT_FAILURE(exp)        \
	do {                           \
		if(!(exp))                 \
			return STATUS_FAILURE; \
	} while(0)

#define CSC(call, err)                  \
	do {                                \
		cudaError_t cudaStatus = call;  \
		if(cudaStatus != cudaSuccess) { \
			return err;                 \
		}                               \
	} while(0);

#define DEBUF_PRINTF(...) \
	do {                  \
	} while(0)

cudnnStatus_t convertTrt2cudnnDtype(nvinfer1::DataType trt_dtype, cudnnDataType_t* cudnn_dtype);

// Eunmerator for status
typedef enum {
	STATUS_SUCCESS         = 0,
	STATUS_FAILURE         = 1,
	STATUS_BAD_PARAM       = 2,
	STATUS_NOT_SUPPORTED   = 3,
	STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

const int max_tensor_dims = 10;

struct TensorDesc {
	int shape[max_tensor_dims];
	int stride[max_tensor_dims];
	int dim;
};

inline unsigned int getElementSize(nvinfer1::DataType t) {
	switch(t) {
	case nvinfer1::DataType::kINT32:
		return 4;

	case nvinfer1::DataType::kFLOAT:
		return 4;

	case nvinfer1::DataType::kHALF:
		return 2;

	case nvinfer1::DataType::kINT8:
		return 1;

	default:
		throw std::runtime_error("Invalid Datatype.");
	}

	throw std::runtime_error("Invalid DataType.");

	return 0;
}

inline size_t getAlignedSize(size_t origin_size, size_t aligned_number = 16) {
	return size_t((origin_size + aligned_number - 1) / aligned_number) * aligned_number;
}
