# Minimal-Visual-Odometry
Simple Visual Odometry Framework for RGB-D cameras Resources

This repository is intended purely for learning. It is not robust, not production-ready, and should not be used in any real-world application. The code is incomplete, experimental, and exists only as a way to explore the basic concepts of visual odometry.

## Prerequisites

* **CMake** >= 3.15
* **C++17 compiler** (e.g., g++, clang++)
* **OpenCV** (built with C++ support)
* **Eigen3** >= 3.3

## Build Instructions

Clone the repository and build from source:

```bash
git clone https://github.com/george-yuanji-wang/Minimal-Visual-Odometry.git
cd Minimal-Visual-Odometry
mkdir build
cd build
cmake ..
make
```

## Running the Tests

Before running the test executable `tum_vo_only`, update the dataset path in
`test/tum_vo_only.cpp`:

```cpp
const std::string DATA_DIR   = "/path/to/rgbd_dataset_freiburg1_xyz";
const std::string rgb_list   = DATA_DIR + "/rgb.txt";
const std::string depth_list = DATA_DIR + "/depth.txt";
const std::string gt_path    = DATA_DIR + "/groundtruth.txt";
```

Replace `"rgbd_dataset_freiburg1_xyz"` with the actual location of your dataset.

### Run the Test

After editing the path, rebuild and run:

```bash
cd build
./tum_vo_only
```

## Other Executables

The repository also provides several test and utility executables:

* `test_feature`
* `cam_feature`
* `test_matcher`
* `calibrate`
* `parameters_tune`

Each can be built in the same way and run from the `build` directory.

---

## Notes

* Dataset download: [TUM RGB-D Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset)
* Ensure that OpenCV and Eigen3 are correctly installed and discoverable by CMake.
