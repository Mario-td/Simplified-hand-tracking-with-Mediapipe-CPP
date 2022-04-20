#include "handlandmarks.h"

absl::Status HandlandmarksDetector::RunMPPGraph(std::string &calculator_graph_config_file)
{
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
	// Print the coordinates of the landmarks asynchronously
	MP_RETURN_IF_ERROR(
		this->graph.ObserveOutputStream(kHandLandmarks,
			[this](const mediapipe::Packet &packet) -> ::mediapipe::Status
			{
				auto landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
				for (const ::mediapipe::NormalizedLandmarkList &normalizedlandmarkList : landmarks)
				{
					for (int i = 0; i < normalizedlandmarkList.landmark_size(); ++i)
					{
						this->coordinates[i] = normalizedlandmarkList.landmark(i).x();
						this->coordinates[i + 1] = normalizedlandmarkList.landmark(i).y();
					}
					// std::cout << normalizedlandmarkList.DebugString();
				}
				return mediapipe::OkStatus();
			}));

	MP_RETURN_IF_ERROR(this->graph.StartRun({}));

	this->poller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller));

	return mediapipe::OkStatus();
}

cv::Mat HandlandmarksDetector::DetectLandmarks(cv::Mat image)
{
	// Wrap Mat into an ImageFrame.
	auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
		mediapipe::ImageFormat::SRGBA, image.cols, image.rows,
		mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
	cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
	image.copyTo(input_frame_mat);

	// Prepare and add graph input packet.
	size_t frame_timestamp_us =
		(double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

	gpu_helper.RunInGlContext([&input_frame, &frame_timestamp_us, this]() -> absl::Status
							  {
	  // Convert ImageFrame to GpuBuffer.
	  auto texture = this->gpu_helper.CreateSourceTexture(*input_frame.get());
	  auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
	  glFlush();
	  texture.Release();
	  // Send GPU image packet into the graph.
	  MP_RETURN_IF_ERROR(this->graph.AddPacketToInputStream(
	      kInputStream, mediapipe::Adopt(gpu_frame.release())
	                        .At(mediapipe::Timestamp(frame_timestamp_us))));
	  return absl::OkStatus(); });

	// Get the graph result packet, or stop if that fails.
	mediapipe::Packet packet;
	this->poller->Next(&packet);
	std::unique_ptr<mediapipe::ImageFrame> output_frame;

	// Convert GpuBuffer to ImageFrame.
	this->gpu_helper.RunInGlContext(
		[&packet, &output_frame, this]() -> absl::Status
		{
			auto &gpu_frame = packet.Get<mediapipe::GpuBuffer>();
			auto texture = this->gpu_helper.CreateSourceTexture(gpu_frame);
			output_frame = absl::make_unique<mediapipe::ImageFrame>(
				mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
				gpu_frame.width(), gpu_frame.height(),
				mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
			this->gpu_helper.BindFramebuffer(texture);
			const auto info = mediapipe::GlTextureInfoForGpuBufferFormat(
				gpu_frame.format(), 0, this->gpu_helper.GetGlVersion());
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

HandlandmarksDetector::HandlandmarksDetector(std::string calculator_graph_config_file)
{
	this->RunMPPGraph(calculator_graph_config_file);
}

HandlandmarksDetector::~HandlandmarksDetector()
{
	this->graph.CloseInputStream(kInputStream);
	this->graph.WaitUntilDone();
}
