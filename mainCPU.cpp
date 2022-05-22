// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include "handlandmarks.h"

#include <cstdlib>

#include "boost/interprocess/shared_memory_object.hpp"
#include "boost/interprocess/mapped_region.hpp"

constexpr char kHandLandmarks[] = "landmarks";

class HandlandmarksDetectorCPU : public HandlandmarksDetector
{

public:
    absl::Status RunMPPGraph(std::string &calculator_graph_config_file)
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
                                                        std::cout << "\nLandmark " << i << ":" << std::endl;
                                                        std::cout << "\tx:" << normalizedlandmarkList.landmark(i).x() << std::endl;
                                                        std::cout << "\ty:" << normalizedlandmarkList.landmark(i).y() << std::endl;
                                                        std::cout << "\tz:" << normalizedlandmarkList.landmark(i).z() << std::endl;
                                                    }
                                                    // std::cout << normalizedlandmarkList.DebugString();
                                                }
                                                return mediapipe::OkStatus();
                                            }));

        MP_RETURN_IF_ERROR(this->graph.StartRun({}));

        this->poller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller));

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

    HandlandmarksDetectorCPU(std::string calculator_graph_config_file)
    {
        this->RunMPPGraph(calculator_graph_config_file);
    }
};

int main(int argc, char **argv)
{
    /*    HandlandmarksDetectorCPU handlandmarksDetector(argv[1]);
        cv::Mat output_image = cv::imread("Untitled.png", cv::IMREAD_COLOR);


        cv::cvtColor(output_image, output_image, cv::COLOR_BGR2RGBA);
        output_image = handlandmarksDetector.DetectLandmarks(output_image);

        cv::imshow("Child display window", output_image);
        cv::waitKey(0);

        return EXIT_SUCCESS;
    */
    using namespace boost::interprocess;

    size_t image_dimension_bytes = 2 * sizeof(int);                        // One int for the cols and another for the rows
    size_t landmark_coordinates_bytes = NUM_LANDMARKS * 2 * sizeof(float); // Size for all the landmarks x's and y's

    if (argc == 2)
    {
        // Parent process
        cv::Mat image = cv::imread("Untitled.png", cv::IMREAD_COLOR);

        // Remove shared memory on construction and destruction
        struct shm_remove
        {
            shm_remove() { shared_memory_object::remove("MySharedMemory"); }
            ~shm_remove() { shared_memory_object::remove("MySharedMemory"); }
        } remover;

        // Create a shared memory object.
        shared_memory_object shm(create_only, "MySharedMemory", read_write);

        // Set size
        size_t image_size_bytes = image.step[0] * image.rows;
        shm.truncate(landmark_coordinates_bytes + image_dimension_bytes + image_size_bytes);

        // Map the whole shared memory in this process
        mapped_region region(shm, read_write);

        void *region_address = region.get_address();
        std::memset(region_address, 0, region.get_size());

        // Write the image data
        uchar *image_buff = static_cast<uchar *>(region_address + landmark_coordinates_bytes + image_dimension_bytes);
        memcpy(image_buff, image.data, image_size_bytes);

        // Write the image size
        int *dimension_buff = static_cast<int *>(region_address + landmark_coordinates_bytes);
        dimension_buff[0] = image.cols;
        dimension_buff[1] = image.rows;

        cv::imshow("Parent display window", image);
        cv::waitKey(0);

        // Launch child process
        std::string s(argv[0] + std::string(" ") + argv[1]);
        s += " child";
        if (0 != std::system(s.c_str()))
            return 1;

        // Read the result data
        float *coordinates_buff = static_cast<float *>(region_address);
        for (int i = 0; i < NUM_LANDMARKS; ++i)
        {
            std::cout << "\nLandmark " << i << ":" << std::endl;
            std::cout << "\tx:" << coordinates_buff[i * 2] << std::endl;
            std::cout << "\ty:" << coordinates_buff[i * 2 + 1] << std::endl;
        }
    }
    else
    {
        // Child process

        // Open already created shared memory object.
        shared_memory_object shm(open_only, "MySharedMemory", read_write);

        // Map the whole shared memory in this process
        mapped_region region(shm, read_write);

        void *region_address = region.get_address();

        // Get the buffer locations
        uchar *image_buff = static_cast<uchar *>(region_address + landmark_coordinates_bytes + image_dimension_bytes);
        int *dimension_buff = static_cast<int *>(region_address + landmark_coordinates_bytes);
        float *coordinates_buff = static_cast<float *>(region_address);

        cv::Mat image(cv::Size(dimension_buff[0], dimension_buff[1]), CV_8UC3, image_buff, cv::Mat::AUTO_STEP);

        cv::Mat output_image;
        cv::cvtColor(image, output_image, cv::COLOR_BGR2RGBA);

        // Detect the landmarks in the image
        HandlandmarksDetectorCPU handlandmarksDetector(argv[1]);
        output_image = handlandmarksDetector.DetectLandmarks(output_image);

        // Write the landmark coordinates
        memcpy(coordinates_buff, handlandmarksDetector.coordinates, landmark_coordinates_bytes);
        handlandmarksDetector.resetCoordinates();

        cv::imshow("Child display window", output_image);
        cv::waitKey(0);
    }

    return 0;
}
