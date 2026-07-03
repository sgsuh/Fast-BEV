#pragma once

#include <array>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "kalman_filter_cus.h"

class KF {
   public:
	KF(std::array<float, 9>& box3d, int track_id);

	int time_since_update_;
	int hits_;
	int track_id_;

	KalmanFilterCus kf_;

	std::array<float, 9> initial_pos_;
};

class AB3DMOT {
   public:
	AB3DMOT(int id_init = 0);
	virtual ~AB3DMOT();

	std::vector<std::array<float, 10>> track(std::vector<std::vector<std::array<float, 9>>>& dets);

   protected:
	std::vector<Eigen::VectorXf> prediction(std::vector<KF>& trk);

	void orientationCorrection(float& theta_pre, float& theta_obs);
	void update(std::vector<std::pair<int, int>>& matched, std::vector<int>& unmatched_trks, std::vector<std::array<float, 9>> dets, std::vector<KF>& trks);

	std::vector<int>                   birth(std::vector<std::array<float, 9>>& dets, std::vector<int>& unmatched_dets, std::vector<KF>& trks);
	std::vector<std::array<float, 10>> output(std::vector<KF>& trks);

	bool dataAssociation(std::vector<std::array<float, 9>>& dets, std::vector<Eigen::VectorXf>& trks, std::string metric, float threshold, std::vector<std::pair<int, int>>& matches, std::vector<int>& unmatched_dets, std::vector<int>& unmatched_trks);

	Eigen::MatrixXf computeAffinity(std::vector<std::array<float, 9>>& dets, std::vector<Eigen::VectorXf>& trks, std::string metric);

	float dist3d(std::array<float, 9>& bbox1, Eigen::VectorXf& bbox2);
	float iou(std::array<float, 9>& bbox1, Eigen::VectorXf& bbox2);

	void computeBottom(std::vector<float>& boxA, std::vector<float>& boxB, std::vector<std::array<float, 2>>& boxaBot, std::vector<std::array<float, 2>>& boxbBot);

	float withinRange(float theta);

	float computeInter2d(std::vector<std::array<float, 2>>& boxa_bottom, std::vector<std::array<float, 2>>& boxb_bottom);

	std::vector<double> polygonClip(std::vector<std::array<float, 2>>& subject_polygon, std::vector<std::array<float, 2>>& clip_polygon);

	float convexArea(std::vector<std::array<float, 2>>& boxa_bottom, std::vector<std::array<float, 2>>& boxb_bottom);
	float polyArea2d(std::vector<std::array<float, 2>>& pts);
	float computeHeight(std::vector<float>& box_a, std::vector<float>& box_b, bool inter = true);

	std::vector<std::pair<int, int>> greedyMatching(Eigen::MatrixXf& cost_matrix);

	int id_count_;

	std::vector<std::string> cat_;
	std::vector<std::string> algm_;
	std::vector<std::string> metric_;

	std::vector<float> thres_;
	std::vector<int>   min_hits_;
	std::vector<float> max_sim_;
	std::vector<float> min_sim_;
	std::vector<int>   max_age_;
	std::vector<int>   id_past_;

	std::vector<std::vector<KF>> trackers_;
};
