#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/quaternion.hpp>
#include <vo_core/feature_extractor.hpp>
#include <vo_core/feature_matcher.hpp>
#include <vo_core/pose_estimator.hpp>
#include <memory>
#include <vector>
#include <cstdint>
#include "geometry/geometry.hpp"

namespace vo_core {

enum class CameraType : std::uint8_t { MONO = 0, RGBD = 1 };

struct PipelineConfig {
    CameraType camera{CameraType::RGBD};
    cv::Mat K;
    cv::Mat dist;
    std::vector<double> extra_cfg;
    Pose initial_pose{};
};

class VOPipeline {
public:
    explicit VOPipeline(const PipelineConfig& cfg);
    VOPipeline(const VOPipeline&) = delete;
    VOPipeline& operator=(const VOPipeline&) = delete;
    VOPipeline(VOPipeline&&) noexcept = default;
    VOPipeline& operator=(VOPipeline&&) noexcept = default;
    ~VOPipeline();

    void reset(const Pose& initial_pose);
    void setIntrinsics(const cv::Mat& K);
    void setDistortion(const cv::Mat& dist);
    void setCameraType(CameraType type);

    const Pose& currentPose() const noexcept { return pose_world_; }
    const cv::Mat& intrinsics() const noexcept { return K_; }
    const cv::Mat& distortion() const noexcept { return dist_; }
    CameraType cameraType() const noexcept { return camera_; }
    bool isInitialized() const noexcept { return initialized_; }

    void initialize(const cv::Mat& rgb, const cv::Mat& depth);
    PoseResult processFrame(const cv::Mat& rgb, const cv::Mat& depth);

private:
    CameraType camera_{CameraType::RGBD};
    cv::Mat K_;
    cv::Mat dist_; 
    Pose pose_world_;
    std::vector<double> extra_cfg_;
    bool initialized_{false};

    struct Impl;
    std::unique_ptr<Impl> impl_;

    void initMono(const cv::Mat& rgb);
    void initRGBD(const cv::Mat& rgb, const cv::Mat& depth);
};

void composePoseIncrement(Pose& pose, const cv::Mat& delta_R, const cv::Mat& delta_t);

} 
