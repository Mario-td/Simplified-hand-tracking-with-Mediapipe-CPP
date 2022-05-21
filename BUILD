# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

# Linux only
cc_binary(
    name = "hand_tracking_cpu",
    deps = [ 
        "main_cpu",
        "//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
		"@boost//:interprocess",
    ],  
)

cc_library(
    name = "main_cpu",
    srcs = ["mainCPU.cpp"], 
    deps = [ 
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/framework/formats:landmark_cc_proto",
    ],  
)


cc_binary(
    name = "hand_tracking_gpu",
    deps = [
        "main_gpu",
        "//mediapipe/graphs/hand_tracking:mobile_calculators",
		"@boost//:interprocess",
    ],
)

# Linux only.
# Must have a GPU with EGL support:
# ex: sudo apt-get install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev
# (or similar nvidia/amd equivalent)
cc_library(
    name = "main_gpu",
    srcs = ["main.cpp", "handlandmarks.cpp"],
    hdrs = ["handlandmarks.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_shared_data_internal",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
		"//mediapipe/calculators/util:landmarks_to_render_data_calculator",
		"//mediapipe/framework/formats:landmark_cc_proto",
    ],
)

