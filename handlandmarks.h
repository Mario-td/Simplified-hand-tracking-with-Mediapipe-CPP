#include <fstream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"

#define NUM_LANDMARKS 21

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kHandLandmarks[] = "hand_landmarks";
constexpr char kWindowName[] = "MediaPipe";

class HandlandmarksDetector
{

private:
	mediapipe::CalculatorGraph graph;
	std::unique_ptr<mediapipe::OutputStreamPoller> poller;
	mediapipe::GlCalculatorHelper gpu_helper;

	absl::Status RunMPPGraph(std::string &calculator_graph_config_file);

public:
	HandlandmarksDetector(std::string calculator_graph_config_file);
	~HandlandmarksDetector();

	float coordinates[NUM_LANDMARKS * 2] = {0};
	cv::Mat DetectLandmarks(cv::Mat image);
};
