#include <algorithm>
#include <cmath>

#include "iou3d.h"
#include "parallel_merge_sort.h"
#include "trt_fastbev.h"

// #define VIS

TRTFastBEV::TRTFastBEV(YAML::Node& param_node, YAML::Node& calib_node) {
	init_ = false;

	stream_.resize(6);

	for(int i = 0; i < stream_.size(); ++i) {
		cudaStreamCreate(&stream_[i]);
	}

	// Parameter
	model_height_ = param_node["model_height"].as<int>();
	model_width_  = param_node["model_width"].as<int>();
	img_channel_  = param_node["img_channel"].as<int>();
	n_times_      = param_node["n_times"].as<int>();
	scale_factor_ = param_node["scale_factor"].as<int>();
	canvas_size_  = param_node["canvas_size"].as<int>();
	show_range_   = param_node["show_range"].as<int>();

	vis_thred_ = param_node["vis_thred"].as<float>();

	resize_size_ = param_node["resize_size"].as<std::vector<int>>();
	crop_size_   = param_node["crop_size"].as<std::vector<int>>();

	mean_ = param_node["mean"].as<std::vector<float>>();
	std_  = param_node["std"].as<std::vector<float>>();

	draw_boxes_indexes_bev_ = param_node["draw_boxes_indexes_bev"].as<std::vector<std::vector<int>>>();
	color_map_              = param_node["color_map"].as<std::vector<std::vector<int>>>();

	// Calibration
	std::vector<float> tmp1 = calib_node["ego2global_rot"].as<std::vector<float>>();

	ego2global_rot_ = Eigen::Quaternionf(tmp1[0], tmp1[1], tmp1[2], tmp1[3]);

	tmp1 = calib_node["lidar2ego_rot"].as<std::vector<float>>();

	lidar2ego_rot_ = Eigen::Quaternionf(tmp1[0], tmp1[1], tmp1[2], tmp1[3]);

	tmp1 = calib_node["ego2global_trans"].as<std::vector<float>>();

	ego2global_trans_ << tmp1[0], tmp1[1], tmp1[2];

	tmp1 = calib_node["lidar2ego_trans"].as<std::vector<float>>();

	lidar2ego_trans_ << tmp1[0], tmp1[1], tmp1[2];

	std::vector<std::vector<float>> tmp2 = calib_node["extrinsic"].as<std::vector<std::vector<float>>>();

	for(int i = 0; i < tmp2.size(); ++i) {
		for(int j = 0; j < tmp2[i].size(); ++j) {
			extrinsic_.emplace_back(tmp2[i][j]);
		}
	}

	tmp2 = calib_node["sensor2lidar_rot"].as<std::vector<std::vector<float>>>();

	for(int i = 0; i < tmp2.size(); ++i) {
		Eigen::Matrix3f mat3;

		for(int j = 0; j < 3; ++j) {
			for(int k = 0; k < 3; ++k) {
				mat3(j, k) = tmp2[i][j * 3 + k];
			}
		}

		sensor2lidar_rot_.emplace_back(mat3);
	}

	tmp2 = calib_node["cam_intrinsic"].as<std::vector<std::vector<float>>>();

	for(int i = 0; i < tmp2.size(); ++i) {
		Eigen::Matrix3f mat3;

		for(int j = 0; j < 3; ++j) {
			for(int k = 0; k < 3; ++k) {
				mat3(j, k) = tmp2[i][j * 3 + k];
			}
		}

		cam_intrinsic_.emplace_back(mat3);
	}

	tmp2 = calib_node["sensor2ego_rot"].as<std::vector<std::vector<float>>>();

	for(int i = 0; i < tmp2.size(); ++i) {
		sensor2ego_rot_.emplace_back(Eigen::Quaternionf(tmp2[i][0], tmp2[i][1], tmp2[i][2], tmp2[i][3]));
	}

	tmp2 = calib_node["sensor2lidar_trans"].as<std::vector<std::vector<float>>>();

	for(int i = 0; i < tmp2.size(); ++i) {
		sensor2lidar_trans_.emplace_back(Eigen::Vector3f(tmp2[i][0], tmp2[i][1], tmp2[i][2]));
	}

	tmp2 = calib_node["sensor2ego_trans"].as<std::vector<std::vector<float>>>();

	for(int i = 0; i < tmp2.size(); ++i) {
		sensor2ego_trans_.emplace_back(Eigen::Vector3f(tmp2[i][0], tmp2[i][1], tmp2[i][2]));
	}

	Eigen::Matrix4f lidar2ego   = Eigen::Matrix4f::Identity();
	lidar2ego.block<3, 3>(0, 0) = lidar2ego_rot_.toRotationMatrix();
	lidar2ego.block<3, 1>(0, 3) = lidar2ego_trans_;

	Eigen::Matrix4f ego2global   = Eigen::Matrix4f::Identity();
	ego2global.block<3, 3>(0, 0) = ego2global_rot_.toRotationMatrix();
	ego2global.block<3, 1>(0, 3) = ego2global_trans_;

	l2g_ = ego2global * lidar2ego;

	preprc_ptr_ = std::make_unique<Preprocessor>(param_node, calib_node);

	num_cam_ = param_node["num_cam"].as<int>();

	// Optional headless BEV dump directory (added for the Fast-BEV deploy port).
	if(param_node["save_dir"]) {
		save_dir_ = param_node["save_dir"].as<std::string>();
	}
}

// Top-down BEV render of the decoded lidar-frame boxes. box3d is laid out as
// [x, y, z, dx, dy, dz, yaw, vx, vy] per box (box_code_size_ = 9). The pixel
// mapping matches the in-tree VIS canvas: (coord + show_range)/show_range/2*canvas.
void TRTFastBEV::saveBev(const std::vector<float>& box3d, const std::vector<float>& scores, const std::vector<float>& labels) {
	cv::Mat canvas = cv::Mat::zeros(canvas_size_, canvas_size_, CV_8UC3);
	cv::circle(canvas, cv::Point(canvas_size_ / 2, canvas_size_ / 2), 4, cv::Scalar(80, 80, 80), -1);

	int drawn = 0;
	for(size_t i = 0; i < labels.size(); ++i) {
		if(scores[i] < vis_thred_) {
			continue;
		}
		++drawn;

		const float cx  = box3d[i * box_code_size_ + 0];
		const float cy  = box3d[i * box_code_size_ + 1];
		const float dx  = box3d[i * box_code_size_ + 3];
		const float dy  = box3d[i * box_code_size_ + 4];
		const float yaw = box3d[i * box_code_size_ + 6];

		const float c = std::cos(yaw);
		const float s = std::sin(yaw);

		// 4 BEV corners (box frame -> lidar frame). y is negated to match the VIS view.
		const float ox[4] = {dx * 0.5f, dx * 0.5f, -dx * 0.5f, -dx * 0.5f};
		const float oy[4] = {dy * 0.5f, -dy * 0.5f, -dy * 0.5f, dy * 0.5f};

		cv::Point pts[4];
		for(int k = 0; k < 4; ++k) {
			const float x = cx + ox[k] * c - oy[k] * s;
			const float y = cy + ox[k] * s + oy[k] * c;
			const int   px = int((x + show_range_) / show_range_ / 2.0f * canvas_size_);
			const int   py = int((-y + show_range_) / show_range_ / 2.0f * canvas_size_);
			pts[k]         = cv::Point(px, py);
		}

		const std::vector<int> color = color_map_[int(labels[i]) % color_map_.size()];
		for(int k = 0; k < 4; ++k) {
			cv::line(canvas, pts[k], pts[(k + 1) % 4], CV_RGB(color[0], color[1], color[2]), 1);
		}
	}

	char name[256];
	std::snprintf(name, sizeof(name), "%s/bev_%04d.jpg", save_dir_.c_str(), frame_idx_);
	cv::imwrite(name, canvas);
	std::cout << "[frame " << frame_idx_ << "] detections(score>" << vis_thred_ << ")=" << drawn
	          << "  -> " << name << std::endl;
}

TRTFastBEV::~TRTFastBEV() {
	for(int i = 0; i < stream_.size(); ++i) {
		cudaStreamDestroy(stream_[i]);
	}
}

void        TRTFastBEV::tensorFromImg(const std::vector<cv::Mat>& imgs) {
#pragma omp parallel for num_threads(openmp_num_threads_)
	for(int i = 0; i < imgs.size(); ++i) {
		cv::Mat dst_img;

		cv::resize(imgs[i], dst_img, cv::Size(resize_size_[0], resize_size_[1]));
		dst_img = dst_img(cv::Rect(crop_size_[0], crop_size_[1], (crop_size_[2] - crop_size_[0]), (crop_size_[3] - crop_size_[1])));
		dst_img.convertTo(dst_img, CV_32FC3);

		const long idx0 = i * dim_in_[0].d[2] * dim_in_[0].d[3] * dim_in_[0].d[4];

		for(int c = 0; c < dim_in_[0].d[2]; ++c) {
			const long idx1 = idx0 + c * dim_in_[0].d[3] * dim_in_[0].d[4];

			for(int h = 0; h < dim_in_[0].d[3]; ++h) {
				const long idx2 = idx1 + h * dim_in_[0].d[4];

				for(int w = 0; w < dim_in_[0].d[4]; ++w) {
					const long idx3 = idx2 + w;

					// tensor_in_[idx3] = ((float)imgs[i].at<cv::Vec3f>(h, w)[c] - mean_[c]) / std_[c];
					tensor_in_[idx3] = ((float)dst_img.at<cv::Vec3f>(h, w)[c] - mean_[c]) / std_[c];
				}
			}
		}
	}
}

void TRTFastBEV::preprocess(const std::vector<cv::Mat>& imgs) {
	if(!init_) {
		init_ = true;

		int dim_size = 1;

		for(int i = 0; i < dim_in_[0].nbDims; ++i) {
			dim_size *= dim_in_[0].d[i];
		}

		tensor_in_.resize(dim_size);

		for(int i = 0; i < dim_out_.size(); ++i) {
			dim_size = 1;

			for(int j = 0; j < dim_out_[i].nbDims; ++j) {
				dim_size *= dim_out_[i].d[j];
			}

			std::vector<float> tmp(dim_size);

			tensor_out_.emplace_back(tmp);
		}

		const std::vector<int> featmap_sizes = {dim_out_[2].d[2], dim_out_[2].d[3]};

		gridAnchors(featmap_sizes, mlvl_anchors_);
	}

	tensorFromImg(imgs);
}

std::vector<std::vector<float>> TRTFastBEV::anchorsSingleRange(const std::vector<int>& feature_size, const std::vector<float>& anchor_range, int scale, const std::vector<float>& sizes, const std::vector<float>& rotations) {
	const std::vector<float> z_centers = {anchor_range[2], anchor_range[5]};

	// Linspace
	std::vector<float> y_centers;

	for(int i = 0; i < feature_size[0] + 1; ++i) {
		y_centers.emplace_back(anchor_range[1] + (anchor_range[4] - anchor_range[1]) / feature_size[0] * i);
	}

	std::vector<float> x_centers;

	for(int i = 0; i < feature_size[1] + 1; ++i) {
		x_centers.emplace_back(anchor_range[0] + (anchor_range[3] - anchor_range[0]) / feature_size[1] * i);
	}

	const float y_shift = (y_centers[1] - y_centers[0]) / 2.0f;
	const float x_shift = (x_centers[1] - x_centers[0]) / 2.0f;

	for(int i = 0; i < y_centers.size(); ++i) {
		y_centers[i] += y_shift;
	}

	for(int i = 0; i < x_centers.size(); ++i) {
		x_centers[i] += x_shift;
	}

	// Meshgrid
	std::vector<std::vector<float>> rets;

	rets.resize(9);

	for(int m = 0; m < 1; ++m) {
		for(int j = 0; j < feature_size[0]; ++j) {
			for(int i = 0; i < feature_size[1]; ++i) {
				for(int n = 0; n < rotations.size(); ++n) {
					rets[0].emplace_back(x_centers[i]);
					rets[1].emplace_back(y_centers[j]);
					rets[2].emplace_back(z_centers[m]);
					rets[3].emplace_back(sizes[0]);
					rets[4].emplace_back(sizes[1]);
					rets[5].emplace_back(sizes[2]);
					rets[6].emplace_back(rotations[n]);
					rets[7].emplace_back(0.0f);
					rets[8].emplace_back(0.0f);
				}
			}
		}
	}

	return rets;
}

void TRTFastBEV::singleLevelGridAnchors(const std::vector<int>& featmap_size, const int scale, std::vector<float>& anchors) {
	std::vector<std::vector<std::vector<float>>> mr_anchors;

	// sizes: 4
	for(int i = 0; i < sizes_.size(); ++i) {
		// 1 x 100 x 100 x 1 x 2 x 9
		mr_anchors.emplace_back(anchorsSingleRange(featmap_size, ranges_, scale, sizes_[i], rotations_));
	}

	// Concatenate
	// 1 x 100 x 100 x 4 x 2 x 9
	for(int i = 0; i < 1; ++i) {
		for(int j = 0; j < featmap_size[0]; ++j) {
			const int idx1 = j * featmap_size[1] * rotations_.size();

			for(int k = 0; k < featmap_size[1]; ++k) {
				const int idx2 = idx1 + k * rotations_.size();

				for(int l = 0; l < mr_anchors.size(); ++l) {
					for(int m = 0; m < rotations_.size(); ++m) {
						const int idx3 = idx2 + m;

						for(int n = 0; n < mr_anchors[l].size(); ++n) {
							anchors.emplace_back(mr_anchors[l][n][idx3]);
						}
					}
				}
			}
		}
	}
}

void TRTFastBEV::gridAnchors(const std::vector<int>& featmap_sizes, std::vector<float>& mlvl_anchors) {
	singleLevelGridAnchors(featmap_sizes, scales_, mlvl_anchors);
}

bool cmp(std::pair<int, float>& a, std::pair<int, float>& b) {
	if(a.second == b.second) {
		return a.first > b.first;
	}

	return a.second > b.second;
}

void        TRTFastBEV::decode(const std::vector<float>& anchors, const std::vector<float>& deltas, std::vector<float>& bboxes, std::vector<float>& bboxes_nms) {
#pragma omp parallel for num_threads(openmp_num_threads_)
	for(int idx = 0; idx < anchors.size(); idx += box_code_size_) {
		float       za = anchors[idx + 2];
		const float ha = anchors[idx + 5];

		za = za + ha / 2.0f;

		const float la = anchors[idx + 4];
		const float wa = anchors[idx + 3];

		const float diagonal = std::sqrt(la * la + wa * wa);

		const float xa = anchors[idx];
		const float xt = deltas[idx];
		const float xg = xt * diagonal + xa;

		const float ya = anchors[idx + 1];
		const float yt = deltas[idx + 1];
		const float yg = yt * diagonal + ya;

		const float zt = deltas[idx + 2];
		float       zg = zt * ha + za;

		const float lt = deltas[idx + 4];
		const float lg = std::exp(lt) * la;

		const float wt = deltas[idx + 3];
		const float wg = std::exp(wt) * wa;

		const float ht = deltas[idx + 5];
		const float hg = std::exp(ht) * ha;

		const float rg = anchors[idx + 6] + deltas[idx + 6];

		zg = zg - hg / 2.0f;

		bboxes[idx]     = xg;
		bboxes[idx + 1] = yg;
		bboxes[idx + 2] = zg;
		bboxes[idx + 3] = wg;
		bboxes[idx + 4] = lg;
		bboxes[idx + 5] = hg;
		bboxes[idx + 6] = rg;
		bboxes[idx + 7] = anchors[idx + 7] + deltas[idx + 7];
		bboxes[idx + 8] = anchors[idx + 8] + deltas[idx + 8];

		bboxes_nms[idx / box_code_size_ * bev_code_size_]     = bboxes[idx];
		bboxes_nms[idx / box_code_size_ * bev_code_size_ + 1] = bboxes[idx + 1];
		bboxes_nms[idx / box_code_size_ * bev_code_size_ + 2] = bboxes[idx + 3];
		bboxes_nms[idx / box_code_size_ * bev_code_size_ + 3] = bboxes[idx + 4];
		bboxes_nms[idx / box_code_size_ * bev_code_size_ + 4] = bboxes[idx + 6];
	}
}

void        TRTFastBEV::xywhr2xyxyr(const std::vector<float>& boxes_xywhr, std::vector<float>& boxes) {
#pragma omp parallel for num_threads(openmp_num_threads_)
	for(int i = 0; i < boxes_xywhr.size(); i += bev_code_size_) {
		float half_w = boxes_xywhr[i + 2] / 2.0f;
		float half_h = boxes_xywhr[i + 3] / 2.0f;

		boxes[i]     = boxes_xywhr[i] - half_w;
		boxes[i + 1] = boxes_xywhr[i + 1] - half_h;
		boxes[i + 2] = boxes_xywhr[i] + half_w;
		boxes[i + 3] = boxes_xywhr[i + 1] + half_h;
		boxes[i + 4] = boxes_xywhr[i + 4];
	}
}

void TRTFastBEV::nmsGpu(const std::vector<float>& boxes, std::vector<std::pair<int, float>> scores, float thresh, std::vector<int>& keep) {
	std::vector<std::pair<int, float>> dst_scores(scores.size());
	std::copy(scores.begin(), scores.end(), dst_scores.begin());
	mergeSortParallel(scores, dst_scores, 0, scores.size() - 1);

	std::vector<float> order_boxes(boxes.size());

#pragma omp parallel for num_threads(openmp_num_threads_)
	for(int i = 0; i < boxes.size(); i += bev_code_size_) {
		const int score_idx = scores[i / bev_code_size_].first;

		order_boxes[i]     = boxes[score_idx * bev_code_size_];
		order_boxes[i + 1] = boxes[score_idx * bev_code_size_ + 1];
		order_boxes[i + 2] = boxes[score_idx * bev_code_size_ + 2];
		order_boxes[i + 3] = boxes[score_idx * bev_code_size_ + 3];
		order_boxes[i + 4] = boxes[score_idx * bev_code_size_ + 4];
	}

	std::vector<unsigned long long> tmp_keep(scores.size(), 0);

	const int num_out = nmsNormalGpu(order_boxes, tmp_keep, thresh);

	keep.resize(num_out);

#pragma omp parallel for num_threads(openmp_num_threads_)
	for(int i = 0; i < num_out; ++i) {
		keep[i] = scores[tmp_keep[i]].first;
	}
}

void TRTFastBEV::circleNms(const std::vector<float>& dets, std::vector<std::pair<int, float>> scores, float thresh, std::vector<int>& keep) {
	std::vector<std::pair<int, float>> dst_scores(scores.size());
	std::copy(scores.begin(), scores.end(), dst_scores.begin());
	mergeSortParallel(scores, dst_scores, 0, scores.size() - 1);

	std::vector<int> suppresses(scores.size(), 0);

	for(int i = 0; i < scores.size(); ++i) {
		const int tmp_i = scores[i].first;

		if(suppresses[tmp_i] == 1) {
			continue;
		}

		if(keep.size() > post_max_size_) {
			break;
		}

		keep.emplace_back(tmp_i);

		for(int j = i + 1; j < scores.size(); ++j) {
			const int tmp_j = scores[j].first;

			if(suppresses[tmp_j] == 1) {
				continue;
			}

			const float dist = (dets[tmp_i * 2] - dets[tmp_j * 2]) * (dets[tmp_i * 2] - dets[tmp_j * 2]) + (dets[tmp_i * 2 + 1] - dets[tmp_j * 2 + 1]) * (dets[tmp_i * 2 + 1] - dets[tmp_j * 2 + 1]);

			if(dist <= thresh) {
				suppresses[tmp_j] = 1;
			}
		}
	}
}

void TRTFastBEV::box3dMulticlassScaleNms(const std::vector<float>& mlvl_bboxes, const std::vector<float>& mlvl_bboxes_for_nms, const std::vector<float>& mlvl_scores, const float score_thr, const int max_num, const std::vector<float>& mlvl_dir_scores, std::vector<float>& bboxes, std::vector<float>& scores, std::vector<float>& labels, std::vector<float>& dir_scores) {
	for(int i = 0; i < num_classes_; ++i) {
		std::vector<std::pair<int, float>> tmp_scores;
		std::vector<float>                 tmp_bboxes_for_nms;
		std::vector<float>                 tmp_mlvl_bboxes;
		std::vector<float>                 tmp_mlvl_dir_scores;

		const float nms_thre        = nms_thr_list_[i];
		const float nms_radius_thre = nms_radius_thr_list_[i];
		const float nms_rescale     = nms_rescale_factor_[i];

		int thr_idx = 0;

		for(int j = 0; j < mlvl_scores.size(); j += (num_classes_ + 1)) {
			if(mlvl_scores[j + i] > score_thr) {
				const int idx = j / (num_classes_ + 1);

				tmp_scores.emplace_back(std::make_pair(thr_idx++, mlvl_scores[j + i]));
				tmp_mlvl_dir_scores.emplace_back(mlvl_dir_scores[idx]);

				for(int k = 0; k < bev_code_size_; ++k) {
					float val = mlvl_bboxes_for_nms[idx * bev_code_size_ + k];

					if(k == 2 || k == 3) {
						val = val * nms_rescale;
					}

					tmp_bboxes_for_nms.emplace_back(val);
				}

				for(int k = 0; k < box_code_size_; ++k) {
					tmp_mlvl_bboxes.emplace_back(mlvl_bboxes[idx * box_code_size_ + k]);
				}
			}
		}

		if(tmp_scores.size() == 0) {
			continue;
		}

		// Rotate
		std::vector<int> selected;

		if(nms_type_list_[i] == 0) {
			std::vector<float> xyxyr_boxes(tmp_bboxes_for_nms.size());

			xywhr2xyxyr(tmp_bboxes_for_nms, xyxyr_boxes);
			nmsGpu(xyxyr_boxes, tmp_scores, nms_thre, selected);
		} else {
			// Circle
			std::vector<float> tmp_centers;

			for(int j = 0; j < tmp_scores.size(); ++j) {
				tmp_centers.emplace_back(tmp_bboxes_for_nms[j * bev_code_size_]);
				tmp_centers.emplace_back(tmp_bboxes_for_nms[j * bev_code_size_ + 1]);
			}

			circleNms(tmp_centers, tmp_scores, nms_radius_thre, selected);
		}

		for(int j = 0; j < selected.size(); ++j) {
			for(int k = 0; k < box_code_size_; ++k) {
				bboxes.emplace_back(tmp_mlvl_bboxes[selected[j] * box_code_size_ + k]);
			}

			scores.emplace_back(tmp_scores[selected[j]].second);

			labels.emplace_back(i);

			dir_scores.emplace_back(tmp_mlvl_dir_scores[selected[j]]);
		}
	}
}

void        TRTFastBEV::limitPeriod(const std::vector<float>& boxes, const float dir_offset, const float offset, const float period, std::vector<float>& dir_rot) {
#pragma omp parallel for num_threads(openmp_num_threads_)
	for(int i = 0; i < boxes.size(); i += box_code_size_) {
		const int idx = i / box_code_size_;
		dir_rot[idx]  = boxes[i + 6] - dir_offset - std::floor((boxes[i + 6] - dir_offset) / period + offset) * period;
	}
}

void TRTFastBEV::getBboxesSingle(const std::vector<float>& mlvl_anchors, std::vector<float>& bboxes, std::vector<float>& scores, std::vector<float>& labels) {
	// Permute(1, 2, 0)
	std::vector<float> dir_cls_pred(dim_out_[1].d[1] * dim_out_[1].d[2] * dim_out_[1].d[3]);
	std::vector<int>   dir_cls_score(dir_cls_pred.size() / 2);

#pragma omp parallel for num_threads(openmp_num_threads_)
	for(int i = 0; i < dim_out_[1].d[2]; ++i) {
		const int idx1     = i * dim_out_[1].d[3];
		const int dst_idx1 = i * dim_out_[1].d[3] * dim_out_[1].d[1];

		for(int j = 0; j < dim_out_[1].d[3]; ++j) {
			const int idx2     = idx1 + j;
			const int dst_idx2 = dst_idx1 + j * dim_out_[1].d[1];

			for(int k = 0; k < dim_out_[1].d[1]; k += 2) {
				const int idx3     = idx2 + k * dim_out_[1].d[2] * dim_out_[1].d[3];
				const int dst_idx3 = dst_idx2 + k;

				dir_cls_pred[dst_idx3]     = tensor_out_[1][idx3];
				dir_cls_pred[dst_idx3 + 1] = tensor_out_[1][idx3 + 1];

				if(dir_cls_pred[dst_idx3] > dir_cls_pred[dst_idx3 + 1]) {
					dir_cls_score[dst_idx3 / 2] = 0;
				} else {
					dir_cls_score[dst_idx3 / 2] = 1;
				}
			}
		}
	}

	// Max Idx

	// Permute(1, 2, 0)
	std::vector<float> cls_score(dim_out_[2].d[1] * dim_out_[2].d[2] * dim_out_[2].d[3]);

#pragma omp parallel for num_threads(openmp_num_threads_)
	for(int i = 0; i < dim_out_[2].d[2]; ++i) {
		const int idx1     = i * dim_out_[2].d[3];
		const int dst_idx1 = i * dim_out_[2].d[3] * dim_out_[2].d[1];

		for(int j = 0; j < dim_out_[2].d[3]; ++j) {
			const int idx2     = idx1 + j;
			const int dst_idx2 = dst_idx1 + j * dim_out_[2].d[1];

			for(int k = 0; k < dim_out_[2].d[1]; k++) {
				const int idx3     = idx2 + k * dim_out_[2].d[2] * dim_out_[2].d[3];
				const int dst_idx3 = dst_idx2 + k;

				cls_score[dst_idx3] = tensor_out_[2][idx3];
			}
		}
	}

	// Permute(1, 2, 0)
	std::vector<float> bbox_pred(dim_out_[0].d[1] * dim_out_[0].d[2] * dim_out_[0].d[3]);

#pragma omp parallel for num_threads(openmp_num_threads_)
	for(int i = 0; i < dim_out_[0].d[2]; ++i) {
		const int idx1     = i * dim_out_[0].d[3];
		const int dst_idx1 = i * dim_out_[0].d[3] * dim_out_[0].d[1];

		for(int j = 0; j < dim_out_[0].d[3]; ++j) {
			const int idx2     = idx1 + j;
			const int dst_idx2 = dst_idx1 + j * dim_out_[0].d[1];

			for(int k = 0; k < dim_out_[0].d[1]; ++k) {
				const int idx3     = idx2 + k * dim_out_[0].d[2] * dim_out_[0].d[3];
				const int dst_idx3 = dst_idx2 + k;

				bbox_pred[dst_idx3] = tensor_out_[0][idx3];
			}
		}
	}

	// Max Dim1(Num Class)
	std::vector<std::pair<int, float>> max_scores(cls_score.size() / num_classes_);

#pragma omp parallel for num_threads(openmp_num_threads_)
	for(int idx = 0; idx < cls_score.size(); idx += num_classes_) {
		float max_score = cls_score[idx];

		for(int i = 1; i < num_classes_; ++i) {
			if(max_score < cls_score[idx + i]) {
				max_score = cls_score[idx + i];
			}
		}

		max_scores[idx / num_classes_] = std::make_pair(idx / num_classes_, max_score);
	}

	// Topk
	std::vector<std::pair<int, float>> dst_scores(max_scores.size());
	std::copy(max_scores.begin(), max_scores.end(), dst_scores.begin());
	mergeSortParallel(max_scores, dst_scores, 0, max_scores.size() - 1);

	std::vector<float> topk_anchors(nms_pre_ * box_code_size_);
	std::vector<float> topk_bbox_pred(nms_pre_ * box_code_size_);
	std::vector<float> mlvl_scores(nms_pre_ * (num_classes_ + 1));
	std::vector<float> mlvl_dir_cls_score(nms_pre_);

#pragma omp parallel for num_threads(openmp_num_threads_)
	for(int i = 0; i < nms_pre_; ++i) {
		const int box_idx = max_scores[i].first * box_code_size_;
		const int cls_idx = max_scores[i].first * num_classes_;

		for(int j = 0; j < box_code_size_; ++j) {
			topk_anchors[i * box_code_size_ + j]   = mlvl_anchors[box_idx + j];
			topk_bbox_pred[i * box_code_size_ + j] = bbox_pred[box_idx + j];
		}

		for(int j = 0; j < num_classes_; ++j) {
			mlvl_scores[i * (num_classes_ + 1) + j] = cls_score[cls_idx + j];
		}

		// Zero Concatenate
		mlvl_scores[i * (num_classes_ + 1) + num_classes_] = 0.0f;

		mlvl_dir_cls_score[i] = dir_cls_score[max_scores[i].first];
	}

	std::vector<float> mlvl_bboxes(topk_anchors.size());
	std::vector<float> mlvl_bboxes_for_nms(mlvl_bboxes.size() / box_code_size_ * bev_code_size_);

	decode(topk_anchors, topk_bbox_pred, mlvl_bboxes, mlvl_bboxes_for_nms);

	// BEV

	std::vector<float> dir_scores;

	box3dMulticlassScaleNms(mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores, score_thr_, max_num_, mlvl_dir_cls_score, bboxes, scores, labels, dir_scores);

	std::vector<float> dir_rot(bboxes.size() / box_code_size_);

	limitPeriod(bboxes, dir_offset_, dir_limit_offset_, M_PI, dir_rot);

#pragma omp parallel for num_threads(openmp_num_threads_)
	for(int i = 0; i < dir_rot.size(); ++i) {
		bboxes[i * box_code_size_ + 6] = dir_rot[i] + dir_offset_ + M_PI * dir_scores[i];
	}
}

void TRTFastBEV::getBboxes(std::vector<float>& box3d, std::vector<float>& scores, std::vector<float>& labels) {
	getBboxesSingle(mlvl_anchors_, box3d, scores, labels);
}

void TRTFastBEV::getDetObjInfos(std::vector<float>& box3d, std::vector<float>& scores, std::vector<float>& labels) {
	det_objinfos_.clear();

	for(int i = 0; i < scores.size(); ++i) {
		ObjInfo obj_info;

		obj_info.class_       = static_cast<int>(labels[i]);
		obj_info.cam_idx_     = 0;
		obj_info.score_       = scores[i];
		obj_info.position_[0] = box3d[i * box_code_size_ + 0];
		obj_info.position_[1] = box3d[i * box_code_size_ + 1];
		obj_info.position_[2] = box3d[i * box_code_size_ + 2];
		obj_info.position_[3] = box3d[i * box_code_size_ + 3];
		obj_info.position_[4] = box3d[i * box_code_size_ + 4];
		obj_info.position_[5] = box3d[i * box_code_size_ + 5];
		obj_info.position_[6] = box3d[i * box_code_size_ + 6];
		obj_info.position_[7] = box3d[i * box_code_size_ + 7];
		obj_info.position_[8] = box3d[i * box_code_size_ + 8];

		det_objinfos_.emplace_back(obj_info);
	}
}

std::vector<std::vector<std::array<float, 9>>> TRTFastBEV::getDetInfo(std::vector<float>& box3d, std::vector<float>& scores, std::vector<float>& labels) {
	std::vector<std::vector<std::array<float, 9>>> det_infos(7);

	// Car, Truck, Construction Vehicle, Bus, Trailer, Barrier, Motorcycle, Bicycle, Pedestrian, Traffic Cone
	std::vector<int> cls_map = {0, 2, -1, 4, 3, -1, 5, 6, 1, -1};

	for(int i = 0; i < labels.size(); ++i) {
		int track_cls = cls_map[int(labels[i])];

		if(track_cls == -1) {
			continue;
		}

		std::array<float, 9> det_info;

		det_info[0] = box3d[i * box_code_size_ + 5];
		det_info[1] = box3d[i * box_code_size_ + 3];
		det_info[2] = box3d[i * box_code_size_ + 4];
		det_info[3] = box3d[i * box_code_size_ + 0];
		det_info[4] = box3d[i * box_code_size_ + 1];
		det_info[5] = box3d[i * box_code_size_ + 2] + box3d[i * box_code_size_ + 5] * 0.5f;
		det_info[6] = -box3d[i * box_code_size_ + 6] - M_PI / 2.0f;
		det_info[7] = scores[i];
		det_info[8] = track_cls;

		det_infos[track_cls].emplace_back(det_info);
	}

	return det_infos;
}

void TRTFastBEV::postprocess(std::vector<cv::Mat>& imgs) {
	std::vector<float> box3d;
	std::vector<float> scores;
	std::vector<float> labels;

	getBboxes(box3d, scores, labels);

	if(!save_dir_.empty()) {
		saveBev(box3d, scores, labels);
	}
	++frame_idx_;

#ifdef VIS
	// Visualize
	cv::Mat canvas   = cv::Mat::zeros(canvas_size_, canvas_size_, CV_8UC3);
	cv::Mat show_img = cv::Mat::zeros(imgs[0].rows * 2 + canvas_size_ * scale_factor_, imgs[0].cols * 3, CV_8UC3);

#pragma omp parallel for num_threads(openmp_num_threads_)
	for(int i = 0; i < labels.size(); ++i) {
		const std::vector<float> box_gravity_center = {box3d[i * box_code_size_], box3d[i * box_code_size_ + 1], box3d[i * box_code_size_ + 2] + box3d[i * box_code_size_ + 5] * 0.5f};
		const std::vector<float> box_dims           = {box3d[i * box_code_size_ + 3], box3d[i * box_code_size_ + 4], box3d[i * box_code_size_ + 5]};

		float box_yaw = box3d[i * box_code_size_ + 6];

		box_yaw = -box_yaw - M_PI / 2.0f;

		const Eigen::Quaternionf quat(std::cos(box_yaw / 2.0f), 0.0f, 0.0f, std::sin(box_yaw / 2.0f));

		const std::vector<float> velocity_all = {box3d[i * box_code_size_ + 7], box3d[i * box_code_size_ + 8], 0.0f};

		if(scores[i] < vis_thred_) {
			continue;
		}

		// Lidar2Ego Rotation & Translation
		Eigen::Vector3f    lidar_center = lidar2ego_rot_.toRotationMatrix() * Eigen::Vector3f(box_gravity_center[0], box_gravity_center[1], box_gravity_center[2]);
		Eigen::Quaternionf lidar_quat   = lidar2ego_rot_ * quat;
		lidar_center                    = lidar_center + lidar2ego_trans_;

		// Ego2Global Rotation & Translation
		lidar_center = ego2global_rot_.toRotationMatrix() * lidar_center;
		lidar_quat   = ego2global_rot_ * lidar_quat;
		lidar_center = lidar_center + ego2global_trans_;

		Eigen::Vector3f lidar_wlh(box_dims[0], box_dims[1], box_dims[2]);

		// Yaw
		float yaw = std::atan2(2 * (lidar_quat.w() * lidar_quat.z() - lidar_quat.x() * lidar_quat.y()), 1 - 2 * (lidar_quat.y() * lidar_quat.y() + lidar_quat.z() * lidar_quat.z()));
		yaw       = -yaw - M_PI / 2.0f;

		const float rot_sin = std::sin(yaw);
		const float rot_cos = std::cos(yaw);

		const Eigen::Matrix3f rot_mat = (Eigen::Matrix3f() << rot_cos, -rot_sin, 0.0f, rot_sin, rot_cos, 0.0f, 0.0f, 0.0f, 1.0f).finished();

		// Box Corners
		Eigen::MatrixXf corners_global(8, 4);

		for(int j = 0; j < corners_norm_.size(); ++j) {
			Eigen::Vector3f corners = lidar_wlh.cwiseProduct(corners_norm_[j]);

			// Rotation 3D in Axis
			corners = corners.transpose() * rot_mat;

			corners = corners + lidar_center;

			corners_global(j, 0) = corners(0);
			corners_global(j, 1) = corners(1);
			corners_global(j, 2) = corners(2);
			corners_global(j, 3) = 1.0f;
		}

		Eigen::MatrixXf corners_lidar(8, 4);

		corners_lidar                   = corners_global * l2g_.inverse().transpose();
		corners_lidar.block<8, 1>(0, 1) = -corners_lidar.block<8, 1>(0, 1);

		std::vector<std::vector<int>> bottom_corners_bev;
		std::vector<float>            center_bev(2, 0.0f);
		std::vector<float>            head_bev(2, 0.0f);

		for(int j = 0; j < bev_idx_.size(); ++j) {
			const int x = int(float(corners_lidar(bev_idx_[j], 0) + show_range_) / show_range_ / 2.0f * canvas_size_);
			const int y = int(float(corners_lidar(bev_idx_[j], 1) + show_range_) / show_range_ / 2.0f * canvas_size_);

			const std::vector<int> bottom_corner = {x, y};

			bottom_corners_bev.emplace_back(bottom_corner);

			center_bev[0] += corners_lidar(bev_idx_[j], 0);
			center_bev[1] += corners_lidar(bev_idx_[j], 1);

			if(j == 0 || j == (bev_idx_.size() - 1)) {
				head_bev[0] += corners_lidar(bev_idx_[j], 0);
				head_bev[1] += corners_lidar(bev_idx_[j], 1);
			}
		}

		center_bev[0] = (center_bev[0] / bev_idx_.size() + show_range_) / show_range_ / 2.0f * canvas_size_;
		center_bev[1] = (center_bev[1] / bev_idx_.size() + show_range_) / show_range_ / 2.0f * canvas_size_;

		head_bev[0] = (head_bev[0] / 2.0f + show_range_) / show_range_ / 2.0f * canvas_size_;
		head_bev[1] = (head_bev[1] / 2.0f + show_range_) / show_range_ / 2.0f * canvas_size_;

		const std::vector<int> color = color_map_[int(labels[i])];

		for(int j = 0; j < draw_boxes_indexes_bev_.size(); ++j) {
			const int x1 = bottom_corners_bev[draw_boxes_indexes_bev_[j][0]][0];
			const int y1 = bottom_corners_bev[draw_boxes_indexes_bev_[j][0]][1];

			const int x2 = bottom_corners_bev[draw_boxes_indexes_bev_[j][1]][0];
			const int y2 = bottom_corners_bev[draw_boxes_indexes_bev_[j][1]][1];

			cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2), CV_RGB(color[2], color[1], color[0]), 1);
		}

		cv::line(canvas, cv::Point(int(center_bev[0]), int(center_bev[1])), cv::Point(int(head_bev[0]), int(head_bev[1])), CV_RGB(color[2], color[1], color[0]), 1);

		for(int j = 0; j < imgs.size(); ++j) {
			Eigen::Vector3f camera_center = lidar_center - ego2global_trans_;
			camera_center                 = ego2global_rot_.toRotationMatrix().inverse() * camera_center;

			Eigen::Quaternionf camera_quat = ego2global_rot_.inverse() * lidar_quat;

			camera_center = camera_center - sensor2ego_trans_[cam_dir_idx_[j]];
			camera_center = sensor2ego_rot_[cam_dir_idx_[j]].toRotationMatrix().inverse() * camera_center;

			camera_quat = sensor2ego_rot_[cam_dir_idx_[j]].inverse() * camera_quat;

			const Eigen::Vector3f camera_wlh = lidar_wlh;

			// Corners
			Eigen::MatrixXf corners_camera(3, 8);

			for(int k = 0; k < 8; ++k) {
				corners_camera(0, k) = camera_wlh(1) / 2.0f * nusc_corners_(0, k);
				corners_camera(1, k) = camera_wlh(0) / 2.0f * nusc_corners_(1, k);
				corners_camera(2, k) = camera_wlh(2) / 2.0f * nusc_corners_(2, k);
			}

			corners_camera = camera_quat.toRotationMatrix() * corners_camera;

			for(int k = 0; k < 8; ++k) {
				corners_camera(0, k) += camera_center(0);
				corners_camera(1, k) += camera_center(1);
				corners_camera(2, k) += camera_center(2);
			}

			Eigen::Matrix4f viewpad = Eigen::Matrix4f::Identity();

			viewpad.block<3, 3>(0, 0) = cam_intrinsic_[cam_dir_idx_[j]];

			Eigen::MatrixXf corners_img(4, 8);

			corners_img.block<3, 8>(0, 0) = corners_camera;

			for(int k = 0; k < 8; ++k) {
				corners_img(3, k) = 1.0f;
			}

			corners_img = viewpad * corners_img;

			int visible  = 0;
			int in_front = 0;

			for(int k = 0; k < 8; ++k) {
				corners_img(0, k) /= corners_img(2, k);
				corners_img(1, k) /= corners_img(2, k);

				if(corners_img(0, k) > 0.0f && corners_img(0, k) < imgs[0].cols && corners_img(1, k) > 0.0f && corners_img(1, k) < imgs[0].rows && corners_camera(2, k) > 1.0f) {
					visible++;
				}

				if(corners_camera(2, k) > 0.1f) {
					in_front++;
				}
			}

			if(visible < 1 || in_front != 8) {
				continue;
			}

			int prev0_x = int(corners_img(0, 3));
			int prev0_y = int(corners_img(1, 3));

			int prev1_x = int(corners_img(0, 7));
			int prev1_y = int(corners_img(1, 7));

			for(int k = 0; k < 4; ++k) {
				cv::line(imgs[j], cv::Point(int(corners_img(0, k)), int(corners_img(1, k))), cv::Point(int(corners_img(0, k + 4)), int(corners_img(1, k + 4))), CV_RGB(color[2], color[1], color[0]), scale_factor_);
				cv::line(imgs[j], cv::Point(prev0_x, prev0_y), cv::Point(int(corners_img(0, k)), int(corners_img(1, k))), CV_RGB(color[2], color[1], color[0]), scale_factor_);
				cv::line(imgs[j], cv::Point(prev1_x, prev1_y), cv::Point(int(corners_img(0, k + 4)), int(corners_img(1, k + 4))), CV_RGB(color[2], color[1], color[0]), scale_factor_);

				prev0_x = int(corners_img(0, k));
				prev0_y = int(corners_img(1, k));

				prev1_x = int(corners_img(0, k + 4));
				prev1_y = int(corners_img(1, k + 4));
			}
		}
	}

#pragma omp parallel for num_threads(openmp_num_threads_)
	for(int i = 0; i < cam_dir_idx_.size(); ++i) {
		const int h = cam_dir_idx_[i] / 3;
		const int w = cam_dir_idx_[i] % 3;

		if(h > 0) {
			cv::Mat tmp_mat;
			cv::flip(imgs[i], tmp_mat, 1);

			tmp_mat.copyTo(show_img.rowRange(imgs[0].rows * h + canvas_size_ * scale_factor_, imgs[0].rows * (h + 1) + canvas_size_ * scale_factor_).colRange(imgs[0].cols * w, imgs[0].cols * (w + 1)));
		} else {
			imgs[i].copyTo(show_img.rowRange(imgs[0].rows * h, imgs[0].rows * (h + 1)).colRange(imgs[0].cols * w, imgs[0].cols * (w + 1)));
		}
	}

	cv::resize(show_img, show_img, cv::Size(int(imgs[0].cols / scale_factor_ * 3), int(imgs[0].rows / scale_factor_ * 2 + canvas_size_)));

	const int w_begin = int(float(imgs[0].cols * 3 / scale_factor_ - canvas_size_) / 2.0f);

	canvas.copyTo(show_img.rowRange(int(imgs[0].rows / scale_factor_), int(imgs[0].rows / scale_factor_) + canvas_size_).colRange(w_begin, w_begin + canvas_size_));

	cv::imshow("result", show_img);
	cv::waitKey(1);
#endif
}

void TRTFastBEV::inference(std::vector<cv::Mat>& imgs) {
	if(!init_) {
		init_ = true;

		int dim_size = 1;

		for(int i = 0; i < dim_out_.size(); ++i) {
			dim_size = 1;

			for(int j = 0; j < dim_out_[i].nbDims; ++j) {
				dim_size *= dim_out_[i].d[j];
			}

			std::vector<float> tmp(dim_size);

			tensor_out_.emplace_back(tmp);
		}

		const std::vector<int> featmap_sizes = {dim_out_[2].d[2], dim_out_[2].d[3]};

		gridAnchors(featmap_sizes, mlvl_anchors_);

		Eigen::Matrix<float, 4, 4> tf_vec = Eigen::Matrix<float, 4, 4>::Identity();

		cudaMemcpyAsync(cuda_in_[1], (void*)extrinsic_.data(), size_in_[1], cudaMemcpyHostToDevice, stream_[1]);
		cudaMemcpyAsync(cuda_in_[2], (void*)tf_vec.data(), size_in_[2], cudaMemcpyHostToDevice, stream_[2]);

		for(int i = 0; i < cuda_in_.size(); ++i) {
			cuda_buff_[i] = cuda_in_[i];
		}

		for(int i = 0; i < cuda_out_.size(); ++i) {
			cuda_buff_[cuda_in_.size() + i] = cuda_out_[i];
		}
	}

	preprc_ptr_->preprcMain(imgs, (float*)cuda_in_[0], stream_);

	context_->enqueueV2(cuda_buff_.data(), stream_[0], nullptr);

	cudaStreamSynchronize(stream_[0]);

	cudaMemcpyAsync(tensor_out_[0].data(), (float*)cuda_out_[0], size_out_[0], cudaMemcpyDeviceToHost, stream_[0]);
	cudaMemcpyAsync(tensor_out_[1].data(), (float*)cuda_out_[1], size_out_[1], cudaMemcpyDeviceToHost, stream_[1]);
	cudaMemcpyAsync(tensor_out_[2].data(), (float*)cuda_out_[2], size_out_[2], cudaMemcpyDeviceToHost, stream_[2]);

	cudaStreamSynchronize(stream_[0]);
	cudaStreamSynchronize(stream_[1]);
	cudaStreamSynchronize(stream_[2]);

	postprocess(imgs);
}
