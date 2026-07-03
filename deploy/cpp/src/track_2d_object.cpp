#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

#include "track_2d_object.h"

Track2D::Track2D() {
	valid_flag_            = 0;
	count_                 = 0;
	life_time_             = 0;
	lateral_distance_      = 0.0f;
	longitudinal_distance_ = 0.0f;
	object_flag_           = 0;
	track_flag_            = 0;
}

Track2D::~Track2D() {
}

TrackManagement::TrackManagement()
    : z_(2, 4) {
	au_ << 1.0f, 0.0f, 0.03f, 0.0f,
	    0.0f, 1.0f, 0.0f, 0.03f,
	    0.0f, 0.0f, 1.0f, 0.0f,
	    0.0f, 0.0f, 0.0f, 1.0f;

	av_ << 1.0f, 0.0f, 0.03f, 0.0f,
	    0.0f, 1.0f, 0.0f, 0.03f,
	    0.0f, 0.0f, 1.0f, 0.0f,
	    0.0f, 0.0f, 0.0f, 1.0f;

	aw_ << 1.0f, 0.0f, 0.03f, 0.0f,
	    0.0f, 1.0f, 0.0f, 0.03f,
	    0.0f, 0.0f, 1.0f, 0.0f,
	    0.0f, 0.0f, 0.0f, 1.0f;

	ah_ << 1.0f, 0.0f, 0.03f, 0.0f,
	    0.0f, 1.0f, 0.0f, 0.03f,
	    0.0f, 0.0f, 1.0f, 0.0f,
	    0.0f, 0.0f, 0.0f, 1.0f;

	at_ << 1.0f, 0.0f, 0.03f, 0.0f,
	    0.0f, 1.0f, 0.0f, 0.03f,
	    0.0f, 0.0f, 1.0f, 0.0f,
	    0.0f, 0.0f, 0.0f, 1.0f;

	z_ << 1.0f, 0.0f, 0.0f, 0.0f,
	    0.0f, 1.0f, 0.0f, 0.0f;

	p_ << 1.0f, 0.0f, 0.0f, 0.0f,
	    0.0f, 1.0f, 0.0f, 0.0f,
	    0.0f, 0.0f, 1.0f, 0.0f,
	    0.0f, 0.0f, 0.0f, 1.0f;

	q_ << std::pow(0.03f, 4) / 4.0f, 0.0f, std::pow(0.03f, 3) / 2, 0.0f,
	    0.0f, std::pow(0.03f, 4) / 4.0f, 0.0f, std::pow(0.03f, 3) / 2,
	    std::pow(0.03f, 3) / 2, 0.0f, std::pow(0.03f, 2), 0.0f,
	    0.0f, std::pow(0.03f, 3) / 2, 0.0f, std::pow(0.03f, 2);

	ru_ << measurement_noise_init_, 0.0f,
	    0.0f, measurement_noise_init_;

	rv_ << measurement_noise_init_, 0.0f,
	    0.0f, measurement_noise_init_;

	rw_ << 10.0f, 0.0f,
	    0.0f, 10.0f;

	rh_ << 10.0f, 0.0f,
	    0.0f, 10.0f;

	rt_ << measurement_noise_init_, 0.0f,
	    0.0f, measurement_noise_init_;
}

TrackManagement::~TrackManagement() {
}

float TrackManagement::computeDistanceRatio(ObjInfo& cand, Track2D& track) {
	float diff_area = std::fabs((cand.position_[3] * cand.position_[4]) - (track.predicted_width_ * track.predicted_height_));
	float diff_x    = std::fabs(cand.position_[0] - track.predicted_center_u_);
	float diff_y    = std::fabs(cand.position_[1] - track.predicted_center_v_);
	float distance  = (diff_area * diff_area * costfunc_size_weight_) + (diff_x * diff_x * costfunc_dx_weight_) + (diff_y * diff_y * costfunc_dy_weight_);

	return distance;
}

void TrackManagement::invalidTrack(Track2D& track) {
	track.valid_flag_  = 0;
	track.object_flag_ = 0;
	track.count_       = 0;
	track.life_time_   = 0;
}

void TrackManagement::compareCountWidth(Track2D& track1, Track2D& track2) {
	if(track1.score_ < track2.score_) {
		invalidTrack(track1);

		track2.penalty_ = std::min(track2.penalty_ + 1, max_penalty_);
	} else if(track1.score_ > track2.score_) {
		invalidTrack(track2);

		track1.penalty_ = std::min(track1.penalty_ + 1, max_penalty_);
	} else {
		if(track1.count_ > track2.count_) {
			invalidTrack(track2);

			track1.penalty_ = std::min(track1.penalty_ + 1, max_penalty_);
		} else {
			invalidTrack(track1);

			track2.penalty_ = std::min(track2.penalty_ + 1, max_penalty_);
		}
	}
}

void TrackManagement::removeOverlapTrack() {
	std::array<int, 2>   left;
	std::array<int, 2>   right;
	std::array<int, 2>   top;
	std::array<int, 2>   bottom;
	std::array<int, 2>   area;
	std::array<float, 2> overlap_ratio;

	for(int i = 0; i < track2d_.size(); ++i) {
		if(track2d_[i].valid_flag_ > 0) {
			left[0]   = track2d_[i].ileft_;
			right[0]  = track2d_[i].ileft_ + track2d_[i].iwidth_;
			top[0]    = track2d_[i].ibottom_ - track2d_[i].iheight_;
			bottom[0] = track2d_[i].ibottom_;
			area[0]   = track2d_[i].iwidth_ * track2d_[i].iheight_;

			for(int j = i + 1; j < track2d_.size(); ++j) {
				if(track2d_[i].id_ != track2d_[j].id_) {
					continue;
				}

				left[1]   = track2d_[j].ileft_;
				right[1]  = track2d_[j].ileft_ + track2d_[j].iwidth_;
				top[1]    = track2d_[j].ibottom_ - track2d_[j].iheight_;
				bottom[1] = track2d_[j].ibottom_;
				area[1]   = track2d_[j].iwidth_ * track2d_[j].iheight_;

				int overlap_left   = std::max(left[0], left[1]);
				int overlap_right  = std::min(right[0], right[1]);
				int overlap_top    = std::max(top[0], top[1]);
				int overlap_bottom = std::min(bottom[0], bottom[1]);
				int overlap_area   = (overlap_right - overlap_left) * (overlap_bottom - overlap_top);

				if(overlap_right < overlap_left || overlap_bottom < overlap_top) {
					overlap_area = 0;
				}

				overlap_ratio[0] = (float)overlap_area / area[0];
				overlap_ratio[1] = (float)overlap_area / area[1];

				// Jth track is included in Ith track
				if(overlap_ratio[0] > remove_th_ && overlap_ratio[1] < remove_th_) {
					compareCountWidth(track2d_[i], track2d_[j]);
					// Ith track is included in Jth track
				} else if(overlap_ratio[1] > remove_th_ && overlap_ratio[0] < remove_th_) {
					compareCountWidth(track2d_[i], track2d_[j]);
					// If Ith track and Jth track can merge, then track who has more life-time and count will be alive
				} else if(overlap_ratio[0] > merge_th_ && overlap_ratio[1] > merge_th_) {
					if(track2d_[i].life_time_ > track2d_[j].life_time_) {
						invalidTrack(track2d_[j]);

						track2d_[i].penalty_ = std::min(track2d_[i].penalty_++, max_penalty_);
					} else if(track2d_[i].life_time_ == track2d_[j].life_time_) {
						compareCountWidth(track2d_[i], track2d_[j]);
					} else {
						invalidTrack(track2d_[i]);

						track2d_[j].penalty_ = std::min(track2d_[j].penalty_++, max_penalty_);
					}
					// If a certain track already includes the other track
				} else if(overlap_ratio[0] == 1.0f || overlap_ratio[1] == 1.0f) {
					if(track2d_[i].life_time_ > track2d_[j].life_time_) {
						invalidTrack(track2d_[j]);

						track2d_[i].penalty_ = std::min(track2d_[i].penalty_++, max_penalty_);
					} else if(track2d_[i].life_time_ == track2d_[j].life_time_) {
						compareCountWidth(track2d_[i], track2d_[j]);
					} else {
						invalidTrack(track2d_[i]);

						track2d_[j].penalty_ = std::min(track2d_[j].penalty_++, max_penalty_);
					}
				}
			}
		}
	}

	int idx = 0;

	while(idx < track2d_.size()) {
		if(track2d_[idx].valid_flag_ == 0) {
			track2d_.erase(track2d_.begin() + idx);

			continue;
		} else {
			idx++;
		}
	}
}

void TrackManagement::predictTrack() {
	for(int i = 0; i < track2d_.size(); ++i) {
		// Check current track invalid
		if(track2d_[i].valid_flag_ == 0) {
			continue;
		}

		// Prediction using kalman filter
		track2d_[i].kalman_position_u_.predict();
		track2d_[i].kalman_position_v_.predict();
		track2d_[i].kalman_size_w_.predict();
		track2d_[i].kalman_size_h_.predict();
		track2d_[i].kalman_position_theta_.predict();

		track2d_[i].predicted_center_u_ = track2d_[i].kalman_position_u_.xp_(0);
		track2d_[i].predicted_center_v_ = track2d_[i].kalman_position_v_.xp_(0);
		track2d_[i].predicted_width_    = track2d_[i].kalman_size_w_.xp_(0);
		track2d_[i].predicted_height_   = track2d_[i].kalman_size_h_.xp_(0);
		track2d_[i].predicted_theta_    = track2d_[i].kalman_position_theta_.xp_(0);

		track2d_[i].predicted_left_   = track2d_[i].predicted_center_u_ - track2d_[i].predicted_width_ / 2.0f;
		track2d_[i].predicted_bottom_ = track2d_[i].predicted_center_v_ + track2d_[i].predicted_height_ / 2.0f;
	}
}

void TrackManagement::setTrackAreaLimit(Track2D& track) {
	if(track.object_flag_ == 0) {
		track.gate_width_  = std::fabs(track.kalman_position_u_.xp_(1) + track.kalman_position_u_.xp_(2) * 0.5f) + std::sqrt(track.kalman_position_u_.p_(0, 0)) * track.predicted_width_ / 2.0f + 10.0f;
		track.gate_height_ = std::sqrt(track.kalman_position_v_.p_(0, 0)) * 2.0f + track.predicted_height_ / 2.0f + 10.0f;
	} else {
		track.gate_width_  = std::fabs(track.kalman_position_u_.xp_(1) + track.kalman_position_u_.xp_(2) * 0.5f) + track.predicted_width_ / 6.0f + 10.0f;
		track.gate_height_ = track.predicted_height_ / 3.0f + 10.0f;
		track.gate_width_ += std::min(track.predicted_width_ / 5.0f, track.predicted_width_ / 6.0f * track.penalty_);
		track.gate_height_ += std::min(track.predicted_height_ / 6.0f, track.predicted_height_ / 8.0f * track.penalty_);
	}

	track.gate_left_top_u_     = track.predicted_center_u_ - track.gate_width_ / 2.0f;
	track.gate_left_top_v_     = track.predicted_center_v_ - track.gate_height_ / 2.0f;
	track.gate_right_bottom_u_ = track.predicted_center_u_ + track.gate_width_ / 2.0f;
	track.gate_right_bottom_v_ = track.predicted_center_v_ + track.gate_height_ / 2.0f;
}

bool TrackManagement::isPointInRect(ObjInfo& cand, Track2D& track) {
	float rect_left_top_u     = std::min(track.gate_left_top_u_, track.gate_right_bottom_u_);
	float rect_left_top_v     = std::min(track.gate_left_top_v_, track.gate_right_bottom_v_);
	float rect_right_bottom_u = std::max(track.gate_left_top_u_, track.gate_right_bottom_u_);
	float rect_right_bottom_v = std::max(track.gate_left_top_v_, track.gate_right_bottom_v_);

	if(rect_left_top_u <= cand.position_[0] && cand.position_[0] <= rect_right_bottom_u && rect_left_top_v <= cand.position_[1] && cand.position_[1] <= rect_right_bottom_v) {
		return true;
	} else {
		return false;
	}
}

bool TrackManagement::isOverlap(ObjInfo& cand, Track2D& track) {
	float cand_left   = cand.position_[0] - cand.position_[3] / 2.0f;
	float cand_bottom = cand.position_[1] + cand.position_[4] / 2.0f;
	float cand_width  = cand.position_[3];
	float cand_height = cand.position_[4];

	float max_left_u   = std::max(track.predicted_left_, cand_left);
	float min_right_u  = std::min(track.predicted_left_ + track.predicted_width_, cand_left + cand_width);
	float min_bottom_v = std::min(track.predicted_bottom_, cand_bottom);
	// TODO predicted width -> predicted height
	float max_top_v = std::max(track.predicted_bottom_ - track.predicted_height_, cand_bottom - cand_height);

	float overlap_width  = min_right_u - max_left_u;
	float overlap_height = min_bottom_v - max_top_v;

	if(max_top_v < min_bottom_v && max_left_u < min_right_u && overlap_width > 0.0f && overlap_height > 0.0f) {
		// TODO predicted width -> predicted height
		float min_size = std::min(track.predicted_width_ * track.predicted_height_, cand_width * cand_height);
		float max_size = std::max(track.predicted_width_ * track.predicted_height_, cand_width * cand_height);

		float overlap_size  = overlap_width * overlap_height;
		float overlap_ratio = overlap_size / min_size;
		float size_ratio    = overlap_size / max_size;

		if(overlap_ratio > min_overlap_ratio_ && size_ratio > min_size_ratio_) {
			return true;
		}
	}

	return false;
}

void TrackManagement::track(std::vector<ObjInfo>& cands) {
	std::vector<bool> association_flag(cands.size());

	// Search for a object to be associated
	for(int i = 0; i < track2d_.size(); ++i) {
		// Check current track invalid
		if(track2d_[i].valid_flag_ == 0) {
			continue;
		}

		// Save the previous position of track
		track2d_[i].prev_center_u_ = track2d_[i].center_u_;
		track2d_[i].prev_center_v_ = track2d_[i].center_v_;

		int min_idx = -1;

		track2d_[i].assoc_index_ = -1;

		float min_overlap = std::numeric_limits<float>::max();

		for(int j = 0; j < cands.size(); ++j) {
			if(!isOverlap(cands[j], track2d_[i])) {
				continue;
			}

			if(track2d_[i].id_ != cands[j].class_) {
				continue;
			}

			float overlap = computeDistanceRatio(cands[j], track2d_[i]);

			if(min_overlap > overlap) {
				min_overlap = overlap;
				min_idx     = j;
			}
		}

		// If object is associated with the current track
		if(min_idx >= 0) {
			track2d_[i].life_time_ = std::min(track2d_[i].life_time_ + 1, max_life_);

			float center_u = cands[min_idx].position_[0];
			float center_v = cands[min_idx].position_[1];
			float width    = cands[min_idx].position_[3];
			float height   = cands[min_idx].position_[4];
			float theta    = cands[min_idx].position_[6];

			track2d_[i].assoc_index_ = min_idx;

			std::array<float, 2> z;

			z[0] = center_u;
			z[1] = 0.0f;

			track2d_[i].kalman_position_u_.correct(z);

			z[0] = center_v;

			track2d_[i].kalman_position_v_.correct(z);

			z[0] = width;

			track2d_[i].kalman_size_w_.r_(0, 0) = std::pow(track2d_[i].kalman_size_w_.xp_(0) * measurement_noise_w_, 2);

			track2d_[i].kalman_size_w_.correct(z);

			z[0] = height;

			track2d_[i].kalman_size_h_.r_(0, 0) = std::pow(track2d_[i].kalman_size_h_.xp_(0) * measurement_noise_w_, 2);

			track2d_[i].kalman_size_h_.correct(z);

			if(fabs(track2d_[i].kalman_position_theta_.xp_(0) - theta) >= M_PI / 6.0f) {
				z[0] = track2d_[i].kalman_position_theta_.xp_(0);
			} else {
				z[0]                                        = theta;
				track2d_[i].kalman_position_theta_.r_(0, 0) = std::pow(track2d_[i].kalman_position_theta_.xp_(0) * measurement_noise_w_, 2);
				track2d_[i].kalman_position_theta_.correct(z);
			}

			track2d_[i].center_u_ = track2d_[i].kalman_position_u_.x_(0);
			track2d_[i].center_v_ = track2d_[i].kalman_position_v_.x_(0);
			track2d_[i].fwidth_   = track2d_[i].kalman_size_w_.x_(0);
			track2d_[i].fheight_  = track2d_[i].kalman_size_h_.x_(0);
			track2d_[i].ftheta_   = track2d_[i].kalman_position_theta_.x_(0);

			track2d_[i].fleft_       = track2d_[i].center_u_ - track2d_[i].fwidth_ / 2.0f;
			track2d_[i].fbottom_     = track2d_[i].center_v_ + track2d_[i].fheight_ / 2.0f;
			track2d_[i].ileft_       = int(track2d_[i].fleft_ + 0.5f);
			track2d_[i].ibottom_     = int(track2d_[i].fbottom_ + 0.5f);
			track2d_[i].iwidth_      = int(track2d_[i].fwidth_ + 0.5f);
			track2d_[i].iheight_     = int(track2d_[i].fheight_ + 0.5f);
			track2d_[i].penalty_     = std::max(track2d_[i].penalty_ - 1, 0);
			track2d_[i].track_flag_  = 0;
			track2d_[i].track_debug_ = 2;

			track2d_[i].center_z_ = cands[min_idx].position_[2];
			track2d_[i].flength_  = cands[min_idx].position_[5];

			// If a track's life time >= 3, It's a real object
			if(track2d_[i].life_time_ >= min_life_) {
				track2d_[i].object_flag_ = 1;
			}

			track2d_[i].count_ = std::min(track2d_[i].count_ + 1, max_count_);

			// Association mark
			association_flag[min_idx] = true;
		} else if(track2d_[i].life_time_ < min_life_) {  // Tentative state

			track2d_[i].life_time_   = std::max(track2d_[i].life_time_ - 1, 0);
			track2d_[i].count_       = std::max(track2d_[i].count_ - 1, 0);
			track2d_[i].track_debug_ = 3;

			// Kill current track
			if(track2d_[i].life_time_ <= 0) {
				invalidTrack(track2d_[i]);
			}
		} else if(track2d_[i].life_time_ >= min_life_) {  // Confirm state
			track2d_[i].life_time_ = std::max(track2d_[i].life_time_ - 1, 0);
			track2d_[i].count_     = std::max(track2d_[i].count_ - 1, 0);
			track2d_[i].max_overlap_ratio_ += overlap_step_;
			track2d_[i].max_overlap_ratio_ = std::min(track2d_[i].max_overlap_ratio_, max_overlap_);
			track2d_[i].track_debug_       = 3;

			// If the life time is 0, kill the current track
			// Terminate track
			if(track2d_[i].life_time_ < min_life_) {
				invalidTrack(track2d_[i]);
			} else {
				std::array<float, 2> z;

				z[0] = track2d_[i].kalman_position_u_.xp_(0);
				z[1] = 0.0f;

				track2d_[i].kalman_position_u_.correct(z);

				z[0] = track2d_[i].kalman_position_v_.xp_(0);

				track2d_[i].kalman_position_v_.correct(z);

				z[0] = track2d_[i].kalman_size_w_.x_(0);

				track2d_[i].kalman_size_w_.r_(0, 0) = std::pow(track2d_[i].kalman_size_w_.xp_(0) * measurement_noise_w_, 2);

				track2d_[i].kalman_size_w_.correct(z);

				z[0] = track2d_[i].kalman_size_h_.x_(0);

				track2d_[i].kalman_size_h_.r_(0, 0) = std::pow(track2d_[i].kalman_size_h_.xp_(0) * measurement_noise_w_, 2);

				track2d_[i].kalman_size_h_.correct(z);

				z[0] = track2d_[i].kalman_position_theta_.xp_(0);

				track2d_[i].kalman_position_theta_.correct(z);

				track2d_[i].track_flag_ = 1;
			}

			track2d_[i].center_u_ = track2d_[i].kalman_position_u_.x_(0);
			track2d_[i].center_v_ = track2d_[i].kalman_position_v_.x_(0);
			track2d_[i].fwidth_   = track2d_[i].kalman_size_w_.x_(0);
			track2d_[i].fheight_  = track2d_[i].kalman_size_h_.x_(0);
			track2d_[i].ftheta_   = track2d_[i].kalman_position_theta_.x_(0);

			track2d_[i].fleft_   = track2d_[i].center_u_ - track2d_[i].fwidth_ / 2.0f;
			track2d_[i].fbottom_ = track2d_[i].center_v_ + track2d_[i].fheight_ / 2.0f;
			track2d_[i].ileft_   = (int)(track2d_[i].fleft_ + 0.5f);
			track2d_[i].ibottom_ = (int)(track2d_[i].fbottom_ + 0.5f);
			track2d_[i].iwidth_  = (int)(track2d_[i].fwidth_ + 0.5f);
			track2d_[i].iheight_ = (int)(track2d_[i].fheight_ + 0.5f);
		}
	}

	int idx = 0;

	while(idx < track2d_.size()) {
		if(track2d_[idx].valid_flag_ == 0) {
			track2d_.erase(track2d_.begin() + idx);
			continue;
		} else {
			idx++;
		}
	}

	// For object that are not associated yet, create new track
	for(int i = 0; i < cands.size(); ++i) {
		if(association_flag[i] == 0) {
			Track2D new_track;

			new_track.object_flag_        = 0;
			new_track.valid_flag_         = 1;
			new_track.life_time_          = 1;
			new_track.count_              = 1;
			new_track.ileft_              = (int)(cands[i].position_[0] - cands[i].position_[3] / 2.0f);
			new_track.ibottom_            = (int)(cands[i].position_[1] + cands[i].position_[4] / 2.0f);
			new_track.iwidth_             = (int)(cands[i].position_[3]);
			new_track.iheight_            = (int)(cands[i].position_[4]);
			new_track.track_flag_         = 0;
			new_track.max_overlap_ratio_  = default_overlap_;
			new_track.score_              = cands[i].score_;
			new_track.center_u_           = cands[i].position_[0];
			new_track.center_v_           = cands[i].position_[1];
			new_track.predicted_center_u_ = cands[i].position_[0];
			new_track.predicted_center_v_ = cands[i].position_[1];
			new_track.fwidth_             = cands[i].position_[3];
			new_track.fheight_            = cands[i].position_[4];
			new_track.predicted_width_    = cands[i].position_[3];
			new_track.predicted_height_   = cands[i].position_[4];
			new_track.gate_width_         = 0;
			new_track.gate_height_        = 0;
			new_track.penalty_            = 0;
			new_track.track_debug_        = 1;
			new_track.assoc_index_        = -1;
			new_track.id_                 = cands[i].class_;

			new_track.center_z_ = cands[i].position_[2];
			new_track.flength_  = cands[i].position_[5];
			new_track.ftheta_   = cands[i].position_[6];

			Eigen::Vector4f x;

			x(0) = new_track.center_u_;
			x(1) = 0.0f;
			x(2) = 0.0f;
			x(3) = 0.0f;

			new_track.kalman_position_u_.init(au_, z_, p_, q_, ru_, x);

			x(0) = new_track.center_v_;

			new_track.kalman_position_v_.init(av_, z_, p_, q_, rv_, x);

			x(0) = new_track.fwidth_;

			new_track.kalman_size_w_.init(aw_, z_, p_, q_, rw_, x);

			x(0) = new_track.fheight_;

			new_track.kalman_size_h_.init(ah_, z_, p_, q_, rh_, x);

			x(0) = new_track.ftheta_;

			new_track.kalman_position_theta_.init(at_, z_, p_, q_, rt_, x);

			association_flag[i] = 1;

			track2d_.emplace_back(new_track);
		}
	}
}
