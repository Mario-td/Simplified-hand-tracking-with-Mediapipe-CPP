# Simplified hand tracking with Mediapipe C++

## Description

A multiprocessing program in which a parent process writes an image in shared memory, the child process reads it, detects hand landmarks, writes the coordinates in the shared memory, and finally, the parent reads and prints the results.

The class **HandlandmarksDetector** encapsulates all the mediapipe functionality. It is constructed with the path of the GraphConfig file, and it is ready to detect the hand landmarks in images using the GPU.

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
- Open the WORKSPACE file, and add the boost library set-up, to be able to use it with bazel, [link](https://github.com/nelhage/rules_boost), also available in the boost-part-WORKSPACE.txt file, although it could be outdated.
- Clone this repository in the mediapipe directory:

```console
git clone https://github.com/Mario-td/Simplified-hand-tracking-with-Mediapipe-CPP
```

### Build and run

Access the cloned directory, build and run the program:
```console
cd Simplified-hand-tracking-with-Mediapipe-CPP
```

#### With CPU
```console
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 hand_tracking_cpu

GLOG_logtostderr=1 ../bazel-bin/Simplified-hand-tracking-with-Mediapipe-CPP/hand_tracking_cpu hand_tracking_desktop_live.pbtxt
```

#### With GPU
```console
bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 hand_tracking_gpu

GLOG_logtostderr=1 ../bazel-bin/Simplified-hand-tracking-with-Mediapipe-CPP/hand_tracking_gpu hand_tracking_desktop_live_gpu.pbtxt
```
