#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace vo_core {

struct MatchParams {
    float baseRadius = 40.f;
    float scaleFactor = 1.2f;
    int   maxLevelDiff = 1;
    int   thLow = 50;
    float ratio = 0.8f;
    int   orientationBins = 30;
    int   keepBins = 3;
    bool  symmetric = false;
    int   cellW = 35;
    int   cellH = 35;
};

class FeatureMatcher {
public:
    explicit FeatureMatcher(const MatchParams& prm = MatchParams());
    void setParams(const MatchParams& prm);
    const MatchParams& params() const;

    void match(const std::vector<cv::KeyPoint>& kpsQ, const cv::Mat& descQ, cv::Size sizeQ,
               const std::vector<cv::KeyPoint>& kpsT, const cv::Mat& descT, cv::Size sizeT,
               std::vector<cv::DMatch>& matches) const;

private:
    struct GridIndex {
        int cellW = 35, cellH = 35;
        int cols = 0, rows = 0;
        std::vector<std::vector<int>> buckets;
        void build(const std::vector<cv::KeyPoint>& kps, cv::Size imgSz, int cw, int ch);
        void queryWindow(const cv::Point2f& p, float radius, std::vector<int>& outCellIds) const;
    };

    static int hamming256(const uchar* a, const uchar* b);

    void matchLocal(const std::vector<cv::KeyPoint>& kpsQ, const cv::Mat& descQ,
                    const std::vector<cv::KeyPoint>& kpsT, const cv::Mat& descT,
                    const GridIndex& gridT, std::vector<int>& q2t) const;

    void enforceOrientation(const std::vector<cv::KeyPoint>& kpsQ,
                            const std::vector<cv::KeyPoint>& kpsT,
                            std::vector<int>& q2t) const;

    void symmetricFilter(const std::vector<int>& q2t,
                         const std::vector<int>& t2q,
                         std::vector<int>& q2t_sym) const;

    MatchParams prm_;
};



}
