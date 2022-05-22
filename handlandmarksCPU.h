#pragma once

#include "handlandmarks.h"

constexpr char kHandLandmarks[] = "landmarks";

class HandlandmarksDetectorCPU : public HandlandmarksDetector
{
private:
    absl::Status RunMPPGraph(std::string &calculator_graph_config_file) override;

public:
    cv::Mat DetectLandmarks(cv::Mat camera_frame) override;

    explicit HandlandmarksDetectorCPU(std::string calculator_graph_config_file);
};