#pragma once

#include <vector>

int nmsNormalGpu(std::vector<float>& boxes, std::vector<unsigned long long>& keep, float nms_overlap_thresh);
