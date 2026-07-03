#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>

#include "ab3dmot.h"
#include "qhull_cus.h"

KF::KF(std::array<float, 9>& box3d, int track_id)
    : kf_(KalmanFilterCus(10, 7)) {
	initial_pos_       = box3d;
	time_since_update_ = 0;
	track_id_          = track_id;
	hits_              = 1;

	kf_.f_       = Eigen::MatrixXf(10, 10).setIdentity();
	kf_.f_(0, 7) = 1.0f;
	kf_.f_(1, 8) = 1.0f;
	kf_.f_(2, 9) = 1.0f;

	kf_.h_ = Eigen::MatrixXf(7, 10).setIdentity();

	kf_.p_.block<3, 3>(7, 7) *= 1000.0f;
	kf_.p_ *= 10.0f;

	kf_.q_.block<3, 3>(7, 7) *= 0.01f;

	for(int i = 0; i < 7; ++i) {
		kf_.x_(i) = initial_pos_[i];
	}
}

AB3DMOT::AB3DMOT(int id_init) {
	id_count_ = id_init;

	cat_      = {"Car", "Pedestrian", "Truck", "Trailer", "Bus", "Motorcycle", "Bicycle"};
	algm_     = {"greedy", "greedy", "greedy", "greedy", "greedy", "greedy", "greedy"};
	metric_   = {"giou_3d", "dist_3d", "giou_3d", "giou_3d", "giou_3d", "giou_3d", "giou_3d"};
	thres_    = {-0.5f, -2.0f, -0.2f, -0.2f, -0.2f, -0.8f, -0.6f};
	min_hits_ = {3, 3, 3, 3, 3, 3, 3};
	max_sim_  = {1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	min_sim_  = {-1.0f, -100.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
	max_age_  = {2, 2, 2, 2, 2, 2, 2};

	trackers_.resize(7);
}

AB3DMOT::~AB3DMOT() {
}

float AB3DMOT::withinRange(float theta) {
	if(theta >= M_PI) {
		theta -= M_PI * 2;
	}

	if(theta < -M_PI) {
		theta += M_PI * 2;
	}

	return theta;
}

std::vector<Eigen::VectorXf> AB3DMOT::prediction(std::vector<KF>& trk) {
	std::vector<Eigen::VectorXf> trks;

	for(int t = 0; t < trk.size(); ++t) {
		// KF kf_tmp = trk[t];

		trk[t].kf_.predict();

		trk[t].kf_.x_(3) = withinRange(trk[t].kf_.x_(3));

		trk[t].time_since_update_ += 1;

		Eigen::VectorXf tmp_vec(7);
		tmp_vec << trk[t].kf_.x_(6), trk[t].kf_.x_(5), trk[t].kf_.x_(4), trk[t].kf_.x_(0), trk[t].kf_.x_(1), trk[t].kf_.x_(2), trk[t].kf_.x_(3);

		trks.emplace_back(tmp_vec);
	}

	return trks;
}

void AB3DMOT::orientationCorrection(float& theta_pre, float& theta_obs) {
	theta_pre = withinRange(theta_pre);
	theta_obs = withinRange(theta_obs);

	if(std::abs(theta_obs - theta_pre) > M_PI / 2.0f && std::abs(theta_obs - theta_pre) < M_PI * 3.0f / 2.0f) {
		theta_pre += M_PI;
		theta_pre = withinRange(theta_pre);
	}

	if(std::abs(theta_obs - theta_pre) >= M_PI * 3.0f / 2.0f) {
		if(theta_obs > 0.0f) {
			theta_pre += M_PI * 2.0f;
		} else {
			theta_pre -= M_PI * 2.0f;
		}
	}
}

void AB3DMOT::update(std::vector<std::pair<int, int>>& matched, std::vector<int>& unmatched_trks, std::vector<std::array<float, 9>> dets, std::vector<KF>& trks) {
	for(int t = 0; t < trks.size(); ++t) {
		bool unmatched = false;

		for(int i = 0; i < unmatched_trks.size(); ++i) {
			if(t == unmatched_trks[i]) {
				unmatched = true;

				break;
			}
		}

		if(unmatched) {
			continue;
		}

		for(int i = 0; i < matched.size(); ++i) {
			if(t == matched[i].second) {
				int d = matched[i].first;

				trks[t].time_since_update_ = 0;
				trks[t].hits_++;

				orientationCorrection(trks[t].kf_.x_(3), dets[d][6]);

				Eigen::VectorXf bbox3d(7);

				bbox3d << dets[d][3], dets[d][4], dets[d][5], dets[d][6], dets[d][2], dets[d][1], dets[d][0];

				trks[t].kf_.update(bbox3d);
				trks[t].kf_.x_(3) = withinRange(trks[t].kf_.x_(3));
			}
		}
	}
}

std::vector<int> AB3DMOT::birth(std::vector<std::array<float, 9>>& dets, std::vector<int>& unmatched_dets, std::vector<KF>& trks) {
	std::vector<int> new_id_list;

	for(int i = 0; i < unmatched_dets.size(); ++i) {
		std::array<float, 9> tmp_det = {dets[i][3], dets[i][4], dets[i][5], dets[i][6], dets[i][2], dets[i][1], dets[i][0], dets[i][7], dets[i][8]};

		KF trk = KF(tmp_det, id_count_);

		trks.emplace_back(trk);

		new_id_list.emplace_back(trk.track_id_);

		id_count_++;
	}

	return new_id_list;
}

std::vector<std::array<float, 10>> AB3DMOT::output(std::vector<KF>& trks) {
	std::vector<std::array<float, 10>> results;

	int i = 0;

	while(i < trks.size()) {
		int cls = int(trks[i].initial_pos_[8]);

		if(trks[i].time_since_update_ < max_age_[cls] && trks[i].hits_ >= min_hits_[cls]) {
			std::array<float, 10> result = {trks[i].kf_.x_(0), trks[i].kf_.x_(1), trks[i].kf_.x_(2), trks[i].kf_.x_(5), trks[i].kf_.x_(4), trks[i].kf_.x_(6), trks[i].kf_.x_(3), trks[i].initial_pos_[7], trks[i].initial_pos_[8], (float)trks[i].track_id_};
			results.emplace_back(result);
		}

		if(trks[i].time_since_update_ >= max_age_[cls]) {
			trks.erase(trks.begin() + i);
		} else {
			i++;
		}
	}

	return results;
}

static Eigen::Matrix3f roty(float t) {
	float c = std::cos(t);
	float s = std::sin(t);

	Eigen::Matrix3f r;
	r << c, 0, s, 0, 1, 0, -s, 0, c;

	return r;
}

static Eigen::MatrixXf box2corners3dCamcoord(std::vector<float>& bbox) {
	Eigen::Matrix3f r = roty(bbox[6]);

	float l = bbox[2];
	float w = bbox[1];
	float h = bbox[0];

	Eigen::MatrixXf corners(3, 8);
	corners << l / 2.0f, l / 2.0f, -l / 2.0f, -l / 2.0f, l / 2.0f, l / 2.0f, -l / 2.0f, -l / 2.0f,
	    0.0f, 0.0f, 0.0f, 0.0f, -h, -h, -h, -h,
	    w / 2.0f, -w / 2.0f, -w / 2.0f, w / 2.0f, w / 2.0f, -w / 2.0f, -w / 2.0f, w / 2.0f;

	corners = r * corners;

	for(int i = 0; i < 3; ++i) {
		for(int j = 0; j < 8; ++j) {
			corners(i, j) = corners(i, j) + bbox[i + 3];
		}
	}

	Eigen::MatrixXf corners_t = corners.transpose();

	return corners_t;
}

float AB3DMOT::dist3d(std::array<float, 9>& bbox1, Eigen::VectorXf& bbox2) {
	// Box2Corners3D Camcoord
	std::vector<float> bbox1_vec = {bbox1[0], bbox1[1], bbox1[2], bbox1[3], bbox1[4], bbox1[5], bbox1[6]};
	std::vector<float> bbox2_vec = {bbox2(0), bbox2(1), bbox2(2), bbox2(3), bbox2(4), bbox2(5), bbox2(6)};

	Eigen::MatrixXf corners1 = box2corners3dCamcoord(bbox1_vec);
	Eigen::MatrixXf corners2 = box2corners3dCamcoord(bbox2_vec);

	std::array<float, 3> c1 = {0.0f, 0.0f, 0.0f};
	std::array<float, 3> c2 = {0.0f, 0.0f, 0.0f};

	float dist = 0.0f;

	for(int i = 0; i < 3; ++i) {
		for(int j = 0; j < 8; ++j) {
			c1[i] += corners1(j, i);
			c2[i] += corners2(j, i);
		}

		c1[i] /= 8.0f;
		c2[i] /= 8.0f;

		dist += (c1[i] - c2[i]) * (c1[i] - c2[i]);
	}

	dist = std::sqrt(dist);

	return dist;
}

void AB3DMOT::computeBottom(std::vector<float>& box_a, std::vector<float>& box_b, std::vector<std::array<float, 2>>& boxa_bot, std::vector<std::array<float, 2>>& boxb_bot) {
	Eigen::MatrixXf corners1 = box2corners3dCamcoord(box_a);
	Eigen::MatrixXf corners2 = box2corners3dCamcoord(box_b);

	for(int i = 0; i < 4; ++i) {
		std::array<float, 2> bot_a;
		std::array<float, 2> bot_b;

		bot_a[0] = corners1(7 - i, 0);
		bot_a[1] = corners1(7 - i, 2);

		bot_b[0] = corners2(7 - i, 0);
		bot_b[1] = corners2(7 - i, 2);

		boxa_bot.emplace_back(bot_a);
		boxb_bot.emplace_back(bot_b);
	}
}

static bool inside(std::array<float, 2>& cp1, std::array<float, 2>& cp2, std::array<float, 2>& p) {
	return ((cp2[0] - cp1[0]) * (p[1] - cp1[1])) > ((cp2[1] - cp1[1]) * (p[0] - cp1[0]));
}

static std::array<float, 2> computeIntersection(std::array<float, 2>& cp1, std::array<float, 2>& cp2, std::array<float, 2>& s, std::array<float, 2>& e) {
	std::array<float, 2> dc = {cp1[0] - cp2[0], cp1[1] - cp2[1]};
	std::array<float, 2> dp = {s[0] - e[0], s[1] - e[1]};
	float                n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0];
	float                n2 = s[0] * e[1] - s[1] * e[0];

	float n3;
	if((dc[0] * dp[1] - dc[1] * dp[0]) == 0) {
		n3 = 0.0f;
	} else {
		n3 = 1.0f / (dc[0] * dp[1] - dc[1] * dp[0]);
	}

	return {(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3};
}

std::vector<double> AB3DMOT::polygonClip(std::vector<std::array<float, 2>>& subject_polygon, std::vector<std::array<float, 2>>& clip_polygon) {
	std::vector<std::array<float, 2>> output_list = subject_polygon;
	std::array<float, 2>              cp1         = clip_polygon[clip_polygon.size() - 1];

	std::vector<double> output_vec;

	for(int c = 0; c < clip_polygon.size(); ++c) {
		std::array<float, 2>              cp2        = clip_polygon[c];
		std::vector<std::array<float, 2>> input_list = output_list;

		output_list.clear();

		std::array<float, 2> s = input_list[input_list.size() - 1];

		for(int i = 0; i < input_list.size(); ++i) {
			std::array<float, 2> e = input_list[i];

			if(inside(cp1, cp2, e)) {
				if(!inside(cp1, cp2, s)) {
					output_list.emplace_back(computeIntersection(cp1, cp2, s, e));
				}

				output_list.emplace_back(e);
			} else if(inside(cp1, cp2, s)) {
				output_list.emplace_back(computeIntersection(cp1, cp2, s, e));
			}

			s = e;
		}

		cp1 = cp2;

		if(output_list.size() == 0) {
			return output_vec;
		}
	}

	for(int i = 0; i < output_list.size(); ++i) {
		output_vec.emplace_back((double)output_list[i][0]);
		output_vec.emplace_back((double)output_list[i][1]);
	}

	return output_vec;
}

float AB3DMOT::computeInter2d(std::vector<std::array<float, 2>>& boxa_bottom, std::vector<std::array<float, 2>>& boxb_bottom) {
	if(boxa_bottom.size() == 0 || boxb_bottom.size() == 0) {
		return 0.0f;
	}

	std::vector<double> inter_p = polygonClip(boxa_bottom, boxb_bottom);

	if(inter_p.size() == 0) {
		return 0.0f;
	} else {
		QhullCus hull_inter("i", inter_p, "", "QT", false);

		float vol = (float)hull_inter.va_[0];

		return vol;
	}
}

float AB3DMOT::polyArea2d(std::vector<std::array<float, 2>>& pts) {
	if(pts.size() == 0) {
		return 0.0f;
	}

	float area = pts[0][0] * pts[pts.size() - 1][1] - pts[0][1] * pts[pts.size() - 1][0];

	for(int i = 1; i < pts.size(); ++i) {
		area += pts[i][0] * pts[i - 1][1] - pts[i][1] * pts[i - 1][0];
	}

	return std::abs(area) * 0.5f;
}

float AB3DMOT::convexArea(std::vector<std::array<float, 2>>& boxa_bottom, std::vector<std::array<float, 2>>& boxb_bottom) {
	std::vector<double> all_corners;

	for(int i = 0; i < boxa_bottom.size(); ++i) {
		all_corners.emplace_back(boxa_bottom[i][0]);
		all_corners.emplace_back(boxa_bottom[i][1]);
	}

	for(int i = 0; i < boxb_bottom.size(); ++i) {
		all_corners.emplace_back(boxb_bottom[i][0]);
		all_corners.emplace_back(boxb_bottom[i][1]);
	}

	QhullCus c("i", all_corners, "", "Qt", false);

	std::vector<std::array<float, 2>> convex_corners;

	std::vector<int> vertices = c.vertices_;

	for(int i = 0; i < vertices.size(); ++i) {
		std::array<float, 2> corner = {(float)all_corners[vertices[i] * 2], (float)all_corners[vertices[i] * 2 + 1]};

		convex_corners.emplace_back(corner);
	}

	if(convex_corners.size() == 0) {
		return 0.0f;
	}

	return polyArea2d(convex_corners);
}

float AB3DMOT::computeHeight(std::vector<float>& box_a, std::vector<float>& box_b, bool inter) {
	Eigen::MatrixXf corners1 = box2corners3dCamcoord(box_a);
	Eigen::MatrixXf corners2 = box2corners3dCamcoord(box_b);

	float height;

	if(inter) {
		float ymax = std::min(corners1(0, 1), corners2(0, 1));
		float ymin = std::max(corners1(4, 1), corners2(4, 1));
		height     = std::max(0.0f, ymax - ymin);
	} else {
		float ymax = std::max(corners1(0, 1), corners2(0, 1));
		float ymin = std::min(corners1(4, 1), corners2(4, 1));
		height     = std::max(0.0f, ymax - ymin);
	}

	return height;
}

float AB3DMOT::iou(std::array<float, 9>& bbox1, Eigen::VectorXf& bbox2) {
	// Box2Corners3D Camcoord
	std::vector<float> bbox1_vec = {bbox1[0], bbox1[1], bbox1[2], bbox1[3], bbox1[4], bbox1[5], bbox1[6]};
	std::vector<float> bbox2_vec = {bbox2(0), bbox2(1), bbox2(2), bbox2(3), bbox2(4), bbox2(5), bbox2(6)};

	std::vector<std::array<float, 2>> boxa_bot;
	std::vector<std::array<float, 2>> boxb_bot;

	computeBottom(bbox1_vec, bbox2_vec, boxa_bot, boxb_bot);

	float i_2d           = computeInter2d(boxa_bot, boxb_bot);
	float c_2d           = convexArea(boxa_bot, boxb_bot);
	float overlap_height = computeHeight(bbox1_vec, bbox2_vec);
	float i_3d           = i_2d * overlap_height;
	float u_3d           = bbox1_vec[0] * bbox1_vec[1] * bbox1_vec[2] + bbox2_vec[0] * bbox2_vec[1] * bbox2_vec[2] - i_3d;
	float union_height   = computeHeight(bbox1_vec, bbox2_vec, false);
	float c_3d           = c_2d * union_height;

	if(u_3d == 0 || c_3d == 0) {
		return 0.0f;
	}

	return i_3d / u_3d - (c_3d - u_3d) / c_3d;
}

Eigen::MatrixXf AB3DMOT::computeAffinity(std::vector<std::array<float, 9>>& dets, std::vector<Eigen::VectorXf>& trks, std::string metric) {
	Eigen::MatrixXf aff_matrix = Eigen::MatrixXf(dets.size(), trks.size());

	for(int d = 0; d < dets.size(); ++d) {
		for(int t = 0; t < trks.size(); ++t) {
			float dist_now;

			if(metric == "dist_3d") {
				dist_now = -dist3d(dets[d], trks[t]);
			} else {
				dist_now = iou(dets[d], trks[t]);
			}

			aff_matrix(d, t) = dist_now;
		}
	}

	return aff_matrix;
}

static bool cmp(std::pair<int, float>& a, std::pair<int, float>& b) {
	if(a.second == b.second) {
		return a.first > b.first;
	}

	return a.second < b.second;
}

std::vector<std::pair<int, int>> AB3DMOT::greedyMatching(Eigen::MatrixXf& cost_matrix) {
	int num_dets = cost_matrix.rows();
	int num_trks = cost_matrix.cols();

	std::vector<std::pair<int, float>> distance_1d;

	for(int i = 0; i < num_dets; ++i) {
		for(int j = 0; j < num_trks; ++j) {
			distance_1d.emplace_back(std::make_pair(i * num_trks + j, -cost_matrix(i, j)));
		}
	}

	std::sort(distance_1d.begin(), distance_1d.end(), cmp);

	std::vector<int> det_matches_to_trk(num_dets, -1);
	std::vector<int> trk_matches_to_det(num_trks, -1);

	std::vector<std::pair<int, int>> matched_indices;

	for(int i = 0; i < distance_1d.size(); ++i) {
		int det_id = distance_1d[i].first / num_trks;
		int trk_id = distance_1d[i].first % num_trks;

		if(trk_matches_to_det[trk_id] == -1 && det_matches_to_trk[det_id] == -1) {
			trk_matches_to_det[trk_id] = det_id;
			det_matches_to_trk[det_id] = trk_id;

			matched_indices.emplace_back(std::make_pair(det_id, trk_id));
		}
	}

	return matched_indices;
}

bool AB3DMOT::dataAssociation(std::vector<std::array<float, 9>>& dets, std::vector<Eigen::VectorXf>& trks, std::string metric, float threshold, std::vector<std::pair<int, int>>& matches, std::vector<int>& unmatched_dets, std::vector<int>& unmatched_trks) {
	if(dets.size() == 0 && trks.size() == 0) {
		return false;
	} else if(dets.size() == 0) {
		unmatched_trks.resize(trks.size());

		std::iota(unmatched_trks.begin(), unmatched_trks.end(), 1);

		return false;
	} else if(trks.size() == 0) {
		unmatched_dets.resize(dets.size());

		std::iota(unmatched_dets.begin(), unmatched_dets.end(), 1);

		return false;
	}

	Eigen::MatrixXf aff_matrix;
	aff_matrix = computeAffinity(dets, trks, metric);

	std::vector<std::pair<int, int>> matched_indices;
	matched_indices = greedyMatching(aff_matrix);

	for(int i = 0; i < dets.size(); ++i) {
		bool matched = false;

		for(int j = 0; j < matched_indices.size(); ++j) {
			if(i == matched_indices[j].first) {
				matched = true;

				break;
			}
		}

		if(matched) {
			continue;
		}

		unmatched_dets.emplace_back(i);
	}

	for(int i = 0; i < trks.size(); ++i) {
		bool matched = false;

		for(int j = 0; j < matched_indices.size(); ++j) {
			if(i == matched_indices[j].second) {
				matched = true;

				break;
			}
		}

		if(matched) {
			continue;
		}

		unmatched_trks.emplace_back(i);
	}

	for(int i = 0; i < matched_indices.size(); ++i) {
		if(aff_matrix(matched_indices[i].first, matched_indices[i].second) < threshold) {
			unmatched_dets.emplace_back(matched_indices[i].first);
			unmatched_trks.emplace_back(matched_indices[i].second);
		} else {
			matches.emplace_back(matched_indices[i]);
		}
	}

	return true;
}

std::vector<std::array<float, 10>> AB3DMOT::track(std::vector<std::vector<std::array<float, 9>>>& dets) {
	std::vector<std::array<float, 10>> results;

	for(int idx = 0; idx < cat_.size(); ++idx) {
		std::vector<Eigen::VectorXf> trks = prediction(trackers_[idx]);

		std::vector<std::pair<int, int>> matched;
		std::vector<int>                 unmatched_dets;
		std::vector<int>                 unmatched_trks;

		dataAssociation(dets[idx], trks, metric_[idx], thres_[idx], matched, unmatched_dets, unmatched_trks);

		update(matched, unmatched_trks, dets[idx], trackers_[idx]);

		std::vector<int> new_id_list = birth(dets[idx], unmatched_dets, trackers_[idx]);

		std::vector<std::array<float, 10>> idx_result = output(trackers_[idx]);

		for(int i = 0; i < idx_result.size(); ++i) {
			results.emplace_back(idx_result[i]);
		}
	}

	return results;
}
