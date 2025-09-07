#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/quaternion.hpp>
#include <memory>
#include <vector>
#include <cstdint>

#include "geometry/geometry.hpp"
#include "vo_core/feature_extractor.hpp"
#include "vo_core/feature_matcher.hpp"
#include "vo_core/feature_matcher_map.hpp"
#include "vo_core/pose_estimator.hpp"
#include "map/map.hpp"
#include "map/g2o.hpp"


namespace vo_core {

enum class CameraType : std::uint8_t { MONO = 0, RGBD = 1 };

struct SlamParams {
    CameraType camera{CameraType::RGBD};
    bool   enableBA{true};

    double depthScale{1.0};

    // ORB extractor defaults
    int    nFeatures{1400};
    float  scaleFactor{1.2f};
    int    nLevels{2};

    int    kfMinFrames{0};      
    int    kfMaxFrames{1};       
    double kfTransThresh{0.0};   
    double kfRotThreshDeg{0.0}; 
    
    int minMapPtsForCulling{1200};

};

class SlamPipeline {
public:
    SlamPipeline(const SlamParams& prm,
                 const cv::Mat& K,       // 3x3 CV_64F
                 const cv::Mat& dist);   // CV_64F or empty
    ~SlamPipeline();

    void reset();
    void initialize(const cv::Mat& rgb, const cv::Mat& depth, const Pose& initial_Twc);

    // Process one frame (RGB)
    PoseResult Track(const cv::Mat& rgb, const cv::Mat& depth);
    PoseResult TrackRelative(const cv::Mat& rgb, const cv::Mat& depth);
    
    const Pose&   currentPose() const noexcept { return pose_world_; }
    const cv::Mat& intrinsics()  const noexcept { return K_; }
    const cv::Mat& distortion()  const noexcept { return dist_; }
    bool isInitialized()          const noexcept { return initialized_; }
    size_t numMapPoints() const;

private:
    // Init paths
    void initRGBD(const cv::Mat& rgb, const cv::Mat& depth, const Pose& initial_pose);
    void initMono (const cv::Mat& rgb); // placeholder

    // Constant-velocity pose prediction
    cv::Mat predictPoseConstantVelocity() const;


    void insertKeyframe(const std::vector<cv::KeyPoint>& kps,
                        const cv::Mat&                   desc,
                        const cv::Mat&                   depth,
                        const PoseResult&                pr,
                        const std::vector<int>&          mpIdx,
                        const std::vector<int>&          kpIdx);
    

    // Local bundle adjustment
    void runBundleAdjustment();

private:
    SlamParams prm_;
    cv::Mat K_, dist_;
    Pose pose_world_;
    bool initialized_{false};

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vo_core

