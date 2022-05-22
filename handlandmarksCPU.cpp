#include "handlandmarksCPU.h"

absl::Status HandlandmarksDetectorCPU::RunMPPGraph(std::string &calculator_graph_config_file)
{

    std::ifstream input_file(calculator_graph_config_file);
    std::string calculator_graph_config_contents = std::string((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

    MP_RETURN_IF_ERROR(this->graph.Initialize(config));

    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                     this->graph.AddOutputStreamPoller(kOutputStream));

    MP_RETURN_IF_ERROR(
        this->graph.ObserveOutputStream(kHandLandmarks,
                                        [this](const mediapipe::Packet &packet) -> ::mediapipe::Status
                                        {
                                            auto landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
                                            for (const ::mediapipe::NormalizedLandmarkList &normalizedlandmarkList : landmarks)
                                            {
                                                for (int i = 0; i < normalizedlandmarkList.landmark_size(); ++i)
                                                {
                                                    this->coordinates[i * 2] = normalizedlandmarkList.landmark(i).x();
                                                    this->coordinates[i * 2 + 1] = normalizedlandmarkList.landmark(i).y();
                                                }
                                                // std::cout << normalizedlandmarkList.DebugString();
                                            }
                                            return mediapipe::OkStatus();
                                        }));

    MP_RETURN_IF_ERROR(this->graph.StartRun({}));

    this->poller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller));

    return mediapipe::OkStatus();
}

cv::Mat HandlandmarksDetectorCPU::DetectLandmarks(cv::Mat camera_frame)
{
    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGBA, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    this->graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us)));

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    this->poller->Next(&packet);
    auto &output_frame = packet.Get<mediapipe::ImageFrame>();

    mediapipe::Packet packet_handlandmarks;
    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

    return output_frame_mat;
}

HandlandmarksDetectorCPU::HandlandmarksDetectorCPU(std::string calculator_graph_config_file)
{
    this->RunMPPGraph(calculator_graph_config_file);
}
