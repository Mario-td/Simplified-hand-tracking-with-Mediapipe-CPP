# Simplified hand tracking with Mediapipe C++

## Description

A multiprocessing program in which a parent process writes an image in shared memory, the child process reads it, detects hand landmarks, writes the coordinates in the shared memory, and finally, the parent reads and prints the results.

The class HandlandmarksDetector encapsulates all the mediapipe functionality. It is constructed with the path of the GraphConfig file, and it is ready to detect the hand landmarks in images using the GPU.

---

### Technologies

- Boost.Interprocess
- Mediapipe
- OpenCV

---

## How to use

### Installation

- Install mediapipe and bazel in your system, [link](https://google.github.io/mediapipe/getting_started/install.html).
- Change your current directory to the mediapipe one.
- Open the WORKSPACE file, and add the boost library set-up, to be able to use it with bazel, [link](https://github.com/nelhage/rules_boost).
- Clone this repository in the mediapipe directory:

```console
git clone https://github.com/Mario-td/Simplified-hand-tracking-with-Mediapipe-CPP
```

### Build and run

```console
bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 hand_tracking_gpu

GLOG_logtostderr=1 ../bazel-bin/hand_tracking/hand_tracking_gpu hand_tracking_desktop_live_gpu.pbtxt
```
