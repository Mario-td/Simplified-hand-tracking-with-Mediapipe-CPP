#include <cstdlib>
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

#include "boost/interprocess/shared_memory_object.hpp"
#include "boost/interprocess/mapped_region.hpp"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";
constexpr char kHandLandmarks[] = "hand_landmarks";

class HandlandmarksDetector 
{

private:
	mediapipe::CalculatorGraph graph;

	std::unique_ptr<mediapipe::OutputStreamPoller> poller;
	std::unique_ptr<mediapipe::OutputStreamPoller> poller_handlandmarks;
	
	mediapipe::GlCalculatorHelper gpu_helper;

public:
	absl::Status RunMPPGraph(std::string& calculator_graph_config_file) {

		std::ifstream input_file(calculator_graph_config_file);
  		std::string calculator_graph_config_contents = std::string((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());	
		
  		mediapipe::CalculatorGraphConfig config =
  		    mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
  		        calculator_graph_config_contents);

  		MP_RETURN_IF_ERROR(this->graph.Initialize(config));

  		ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  		MP_RETURN_IF_ERROR(this->graph.SetGpuResources(std::move(gpu_resources)));
  		this->gpu_helper.InitializeForTest(this->graph.GetGpuResources().get());

		ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
		                 this->graph.AddOutputStreamPoller(kOutputStream));
//		ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_handlandmarks,
//		                 this->graph.AddOutputStreamPoller(kHandLandmarks));
///
//mediapipe::CalculatorGraph& graph = this->graph;
MP_RETURN_IF_ERROR(
    this->graph.ObserveOutputStream(kHandLandmarks,
                              [](const mediapipe::Packet& packet) -> ::mediapipe::Status {
    auto landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
		        for (const ::mediapipe::NormalizedLandmarkList &normalizedlandmarkList : landmarks)
		        {
					std::cout << normalizedlandmarkList.DebugString();
		        }
    return mediapipe::OkStatus();
  }));
///
		MP_RETURN_IF_ERROR(this->graph.StartRun({}));
		
		this->poller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller));
		//this->poller_handlandmarks = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller_handlandmarks));

  		return mediapipe::OkStatus();
	}

	cv::Mat DetectLandmarks(cv::Mat camera_frame)
	{
		// Wrap Mat into an ImageFrame.
		auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
		    mediapipe::ImageFormat::SRGBA, camera_frame.cols, camera_frame.rows,
		    mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
		cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
		camera_frame.copyTo(input_frame_mat);
		
		// Prepare and add graph input packet.
		size_t frame_timestamp_us =
		    (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
		
		mediapipe::CalculatorGraph& graph = this->graph;
  		
		mediapipe::GlCalculatorHelper &gpu_helper = this->gpu_helper;
	
		gpu_helper.RunInGlContext([&input_frame, &frame_timestamp_us, &graph,
		                           &gpu_helper]() -> absl::Status {
		  // Convert ImageFrame to GpuBuffer.
		  auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
		  auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
		  glFlush();
		  texture.Release();
		  // Send GPU image packet into the graph.
		  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
		      kInputStream, mediapipe::Adopt(gpu_frame.release())
		                        .At(mediapipe::Timestamp(frame_timestamp_us))));
		  return absl::OkStatus();
		});
		
		// Get the graph result landmarks.
	/*	mediapipe::Packet packet_handlandmarks;
		if(this->poller_handlandmarks->QueueSize() > 0)
		{
		    if (this->poller_handlandmarks->Next(&packet_handlandmarks))
		    {
		        auto& output_handlandmarks = packet_handlandmarks.Get<std::vector<::mediapipe::NormalizedLandmarkList>>();
		        for (const ::mediapipe::NormalizedLandmarkList &normalizedlandmarkList : output_handlandmarks)
		        {
		            std::cout << normalizedlandmarkList.DebugString();
		        }
		    }
		
		}
	*/
		// Get the graph result packet, or stop if that fails.
		mediapipe::Packet packet;
		this->poller->Next(&packet);
		std::unique_ptr<mediapipe::ImageFrame> output_frame;
			
		// Convert GpuBuffer to ImageFrame.
		gpu_helper.RunInGlContext(
		  [&packet, &output_frame, &gpu_helper]() -> absl::Status {
		    auto& gpu_frame = packet.Get<mediapipe::GpuBuffer>();
		    auto texture = gpu_helper.CreateSourceTexture(gpu_frame);
		    output_frame = absl::make_unique<mediapipe::ImageFrame>(
		        mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
		        gpu_frame.width(), gpu_frame.height(),
		        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
		    gpu_helper.BindFramebuffer(texture);
		    const auto info = mediapipe::GlTextureInfoForGpuBufferFormat(
		        gpu_frame.format(), 0, gpu_helper.GetGlVersion());
		    glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
		                 info.gl_type, output_frame->MutablePixelData());
		    glFlush();
		    texture.Release();
		    return absl::OkStatus();
		  });
		
		// Convert back to opencv for display or saving.
		cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());
		if (output_frame_mat.channels() == 4)
		  cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGBA2BGR);
		else
		  cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
		return output_frame_mat;
	}

	HandlandmarksDetector(std::string calculator_graph_config_file)
	{
  		this->RunMPPGraph(calculator_graph_config_file);
	}

	~HandlandmarksDetector()
	{
  		this->graph.CloseInputStream(kInputStream);
  		this->graph.WaitUntilDone();
	}
};

int main(int argc, char** argv) {
	
	using namespace boost::interprocess;
	if (argc == 1)
	{ // Parent process
		cv::Mat frame = cv::imread("Untitled.png", cv::IMREAD_COLOR);
		size_t sizeInBytes = frame.step[0] * frame.rows;

		// Remove shared memory on construction and destruction
		struct shm_remove
		{
			shm_remove() { shared_memory_object::remove("MySharedMemory"); }
			~shm_remove() { shared_memory_object::remove("MySharedMemory"); }
		} remover;
		// Create a shared memory object.
		shared_memory_object shm(create_only, "MySharedMemory", read_write);
		// Set size
		shm.truncate(sizeInBytes);

		// Map the whole shared memory in this process
		mapped_region region(shm, read_write);

		// Write all the memory to 1
		uchar *buff = static_cast<uchar *>(region.get_address());
		memcpy(buff, frame.data, sizeInBytes);

		cv::imshow("Display window", frame);
		cv::waitKey(0); // Wait for a keystroke in the window

		// Launch child process
		std::string s(argv[0]);
		s += " child";
		if (0 != std::system(s.c_str()))
			return 1;
	}
	else
	{
		HandlandmarksDetector handlandmarksDetector("hand_tracking_desktop_live_gpu.pbtxt");
		// Open already created shared memory object.
		shared_memory_object shm(open_only, "MySharedMemory", read_only);

		// Map the whole shared memory in this process
		mapped_region region(shm, read_only);

		uchar *buff = static_cast<uchar *>(region.get_address());
		cv::Mat image(cv::Size(603, 621), CV_8UC3, buff, cv::Mat::AUTO_STEP);
  	  	cv::Mat output_frame_mat;

  	  	cv::cvtColor(image, output_frame_mat, cv::COLOR_BGR2RGBA);
		output_frame_mat = handlandmarksDetector.DetectLandmarks(output_frame_mat);
		
		cv::imshow("Display window child output", output_frame_mat);
		cv::waitKey(0); // Wait for a keystroke in the window
	}
	return 0;
/*
	HandlandmarksDetector handlandmarksDetector(argv[1]);

  	cv::VideoCapture capture;
  	capture.open(0);

  	cv::namedWindow(kWindowName, 1);
  	#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
  	    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  	    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  	    capture.set(cv::CAP_PROP_FPS, 30);
  	#endif

  	bool grab_frames = true;
  	while (grab_frames) {
  	  // Capture opencv camera or video frame.
  	  cv::Mat camera_frame_raw;
  	  capture >> camera_frame_raw;
  	  if (camera_frame_raw.empty()) {
  	    continue;
  	  }
  	  cv::Mat camera_frame;
  	  cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGBA);
  	  cv::flip(camera_frame, camera_frame, 1);

  	  auto output_frame_mat = handlandmarksDetector.DetectLandmarks(camera_frame);

	  cv::imshow(kWindowName, output_frame_mat);
	  // Press any key to exit.
	  const int pressed_key = cv::waitKey(5);
	  if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
	}

	return EXIT_SUCCESS;
*/
}
