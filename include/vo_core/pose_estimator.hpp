#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace vo_core {

struct PoseResult {
    bool success = false;
    cv::Mat rvec;     
    cv::Mat tvec;     
    cv::Mat R;        // 3x3, CV_64F
    cv::Mat T;        // 4x4, CV_64F
    std::vector<int> inliers;
    int inlier_count = 0;
};

class PoseEstimator {
public:
    PoseEstimator(const cv::Mat& K, const cv::Mat& dist = cv::Mat(),
                  double ransac_thresh = 3.0, double ransac_conf = 0.99, int ransac_iters = 500, int min_inliers = 50, double min_inlier_ratio_ = 0.25);

    PoseResult estimate(const std::vector<cv::Point3f>& pts3d,
                        const std::vector<cv::Point2f>& pts2d,
                        bool refine = true);

    const cv::Mat& K() const { return K_; }
    const cv::Mat& dist() const { return dist_; }

private:
    cv::Mat K_;
    cv::Mat dist_;
    double ransac_thresh_;
    double ransac_conf_;
    int ransac_iters_;
    int min_inliers_;
    double min_inlier_ratio_;
};

} 
