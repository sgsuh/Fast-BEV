#pragma once

#include <cassert>
#include <cstring>
#include <fstream>
#include <iterator>
#include <numeric>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

class IBatchStream {
   public:
	virtual void           reset(int first_batch) = 0;
	virtual bool           next()                 = 0;
	virtual void           skip(int skip_count)   = 0;
	virtual float*         getBatch()             = 0;
	virtual float*         getLabels()            = 0;
	virtual int            getBatchesRead() const = 0;
	virtual int            getBatchSize() const   = 0;
	virtual nvinfer1::Dims getDims() const        = 0;
};

class MCODStream: public IBatchStream {
   public:
	MCODStream(int batch_size, int max_batch_size, std::string& data_fold, std::vector<std::string>& fold_list, std::string& calib_path)
	    : batch_size_(batch_size)
	    , max_batch_size_(max_batch_size)
	    , dims_{5, batch_size, 6, 3, 256, 704} {
	}

	void reset(int first_batch) override {
		batch_count_ = first_batch;
	}

	bool next() override {
		if(batch_count_ >= max_batch_size_) {
			return false;
		}

		batch_count_++;

		return true;
	}

	void skip(int skip_count) override {
		batch_count_ += skip_count;
	}

	float* getBatch() override {
		data_.clear();

		int input_len = 1;

		for(int i = 0; i < dims_.nbDims; ++i) {
			input_len *= dims_.d[i];
		}

		data_.resize(input_len);

		return data_.data();
	}

	int getBatchesRead() const override {
		return batch_count_;
	}

	int getBatchSize() const override {
		return batch_size_;
	}

	nvinfer1::Dims getDims() const override {
		return dims_;
	}

	float* getLabels() override {
		return label_.data() + (batch_count_ * batch_size_);
	}

	std::vector<float> getExtrinsic() {
		return extrinsic_;
	}

   protected:
	int batch_size_{0};
	int max_batch_size_{0};
	int batch_count_{-1};

	nvinfer1::Dims dims_;

	std::array<std::vector<std::string>, 6> data_list_{};
	std::vector<float>                      label_{};
	std::vector<float>                      data_{};

	std::vector<float> extrinsic_{};
};

inline long volume(const nvinfer1::Dims& d) {
	return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<long>());
}

template<typename TBatchStream>
class EntropyCalibratorImpl {
   public:
	EntropyCalibratorImpl(TBatchStream stream, int first_batch, std::string network_name, const char* input_blob_name, const char* input_param_name, bool read_cache = true)
	    : stream_{stream}
	    , calibration_table_name_(network_name)
	    , input_blob_name_(input_blob_name)
	    , input_param_name_(input_param_name)
	    , read_cache_(read_cache) {
		nvinfer1::Dims dims = stream_.getDims();
		input_count_        = volume(dims);

		std::vector<float> param = stream_.getExtrinsic();
		param_count_             = param.size();

		cudaMalloc(&device_input_, input_count_ * sizeof(float));
		cudaMalloc(&device_param_, param_count_ * sizeof(float));

		stream_.reset(first_batch);
	}

	virtual ~EntropyCalibratorImpl() {
		cudaFree(device_input_);
		cudaFree(device_param_);
	}

	int getBatchSize() const noexcept {
		return stream_.getBatchSize();
	}

	bool getBatch(void* bindings[], const char* names[], int nb_bindings) noexcept {
		if(!stream_.next()) {
			return false;
		}

		std::vector<float> param = stream_.getExtrinsic();

		cudaMemcpy(device_input_, stream_.getBatch(), input_count_ * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(device_param_, param.data(), param_count_ * sizeof(float), cudaMemcpyHostToDevice);

		assert(!strcmp(names[0], input_blob_name_));
		assert(!strcmp(names[1], input_param_name_));

		bindings[0] = device_input_;
		bindings[1] = device_param_;

		return true;
	}

	const void* readCalibrationCache(size_t& length) noexcept {
		calibration_cache_.clear();

		std::ifstream input(calibration_table_name_, std::ios::binary);

		input >> std::noskipws;

		if(read_cache_ && input.good()) {
			std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calibration_cache_));
		}

		length = calibration_cache_.size();

		return length ? calibration_cache_.data() : nullptr;
	}

	void writeCalibrationCache(const void* cache, size_t length) noexcept {
		std::ofstream output(calibration_table_name_, std::ios::binary);

		output.write(reinterpret_cast<const char*>(cache), length);
	}

   private:
	TBatchStream stream_;

	size_t input_count_;
	size_t param_count_;

	std::string calibration_table_name_;

	const char* input_blob_name_;
	const char* input_param_name_;

	bool read_cache_{true};

	void* device_input_{nullptr};
	void* device_param_{nullptr};

	std::vector<char> calibration_cache_;
};

template<typename TBatchStream>
class Int8EntropyCalibrator2: public nvinfer1::IInt8EntropyCalibrator2 {
   public:
	Int8EntropyCalibrator2(TBatchStream stream, int first_batch, const char* network_name, const char* input_blob_name, const char* input_param_name, bool read_cache = true)
	    : impl_(stream, first_batch, network_name, input_blob_name, input_param_name, read_cache) {
	}

	int getBatchSize() const noexcept override {
		return impl_.getBatchSize();
	}

	bool getBatch(void* bindings[], const char* names[], int nb_bindings) noexcept override {
		return impl_.getBatch(bindings, names, nb_bindings);
	}

	const void* readCalibrationCache(size_t& length) noexcept override {
		return impl_.readCalibrationCache(length);
	}

	void writeCalibrationCache(const void* cache, size_t length) noexcept override {
		impl_.writeCalibrationCache(cache, length);
	}

   private:
	EntropyCalibratorImpl<TBatchStream> impl_;
};

struct InferDeleter {
	template<typename T>
	void operator()(T* obj) const {
		delete obj;
	}
};

template<typename T>
using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;
