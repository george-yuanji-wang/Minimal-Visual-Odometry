#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <list>
#include <algorithm>
#include <cmath>

namespace vo_core {

struct FeatureResult {

    // Keypoints coordinates (x,y) -> (Nx2) (float32) 
    cv::Mat points;     

    // ORB Descriptors (Nx32)
    cv::Mat descriptors;  
};


class FeatureExtractor {
public:
    FeatureExtractor(int n_features = 500, double scale_factor = 1.5, int n_levels = 4);

    void extract(const cv::Mat& img_bgr, const cv::Mat& mask = cv::Mat());
    void extract_prune(const cv::Mat& img_bgr, const cv::Mat& mask = cv::Mat());

    const FeatureResult& result() const { return result_; }

    int nFeatures() const { return n_features_; }
    double scaleFactor() const { return scale_factor_; }
    int nLevels() const { return n_levels_; }

private:
    int n_features_;
    double scale_factor_;
    int n_levels_;

    cv::Ptr<cv::ORB> orb_; 
    FeatureResult result_;

    static constexpr int   kEdgeThreshold = 31;
    static constexpr int   kPatchSize     = 31;
    static constexpr float kCellW         = 35.0f;

    static void collectFASTCells_(const cv::Mat& levelImg,
                                  const cv::Mat& levelMask,
                                  float cellW,
                                  int iniThFAST,
                                  int minThFAST,
                                  int minX, int maxX,
                                  int minY, int maxY,
                                  int maxKpTotal,
                                  std::vector<cv::KeyPoint>& out);
};

}
