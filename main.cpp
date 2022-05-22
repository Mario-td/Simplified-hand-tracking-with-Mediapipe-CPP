#include <cstdlib>

#include "boost/interprocess/shared_memory_object.hpp"
#include "boost/interprocess/mapped_region.hpp"

#ifdef MEDIAPIPE_DISABLE_GPU
#include "handlandmarksCPU.h"
#else
#include "handlandmarksGPU.h"
#endif

int main(int argc, char **argv)
{

	using namespace boost::interprocess;

	size_t image_dimension_bytes = 2 * sizeof(int);						   // One int for the cols and another for the rows
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

#ifdef MEDIAPIPE_DISABLE_GPU
		HandlandmarksDetectorCPU handlandmarksDetector(argv[1]);
#else
		HandlandmarksDetectorGPU handlandmarksDetector(argv[1]);
#endif

		// Detect the landmarks in the image
		output_image = handlandmarksDetector.DetectLandmarks(output_image);

		// Write the landmark coordinates
		memcpy(coordinates_buff, handlandmarksDetector.coordinates, landmark_coordinates_bytes);
		handlandmarksDetector.resetCoordinates();

		cv::imshow("Child display window", output_image);
		cv::waitKey(0);
	}

	return 0;
}
