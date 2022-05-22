#pragma once

#include "handlandmarks.h"

#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

constexpr char kHandLandmarks[] = "hand_landmarks";

class HandlandmarksDetectorGPU : public HandlandmarksDetector
{
private:
	mediapipe::GlCalculatorHelper gpu_helper;
	absl::Status RunMPPGraph(std::string &calculator_graph_config_file) override;

public:
	explicit HandlandmarksDetectorGPU(std::string calculator_graph_config_file);
	cv::Mat DetectLandmarks(cv::Mat image) override;
};
