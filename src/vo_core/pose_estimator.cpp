#include <vo_core/pose_estimator.hpp>

namespace vo_core {

PoseEstimator::PoseEstimator(const cv::Mat& K, const cv::Mat& dist,
                             double ransac_thresh, double ransac_conf, int ransac_iters, int min_inliers, double min_inlier_ratio) {
    K_ = K.clone();
    dist_ = dist.clone();
    ransac_thresh_ = ransac_thresh;
    ransac_conf_ = ransac_conf;
    ransac_iters_ = ransac_iters;
    min_inliers_ = min_inliers;
    min_inlier_ratio_ = min_inlier_ratio;
}

PoseResult PoseEstimator::estimate(const std::vector<cv::Point3f>& pts3d,
                                   const std::vector<cv::Point2f>& pts2d,
                                   bool refine) {
    PoseResult res;
    if (pts3d.size() < 4 || pts2d.size() < 4 || pts3d.size() != pts2d.size()) return res;

    cv::Mat rvec, tvec;
    std::vector<int> inliers;

    bool ok = cv::solvePnPRansac(
        pts3d, pts2d, K_, dist_, rvec, tvec,
        false, ransac_iters_, ransac_thresh_, ransac_conf_, inliers,
        cv::SOLVEPNP_EPNP
    );

    // validity check
    if (!ok || inliers.size() < min_inliers_ ||
    (double)inliers.size() / (double)pts3d.size() < min_inlier_ratio_) {
        return res; 
    }

    if (refine) {
        std::vector<cv::Point3f> P; P.reserve(inliers.size());
        std::vector<cv::Point2f> p; p.reserve(inliers.size());
        for (int idx : inliers) { P.push_back(pts3d[idx]); p.push_back(pts2d[idx]); }
        cv::solvePnPRefineLM(P, p, K_, dist_, rvec, tvec);
    }

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T(cv::Rect(0,0,3,3)));
    T.at<double>(0,3) = tvec.at<double>(0);
    T.at<double>(1,3) = tvec.at<double>(1);
    T.at<double>(2,3) = tvec.at<double>(2);

    res.success = true;
    res.rvec = rvec;
    res.tvec = tvec;
    res.R = R;
    res.T = T;
    res.inliers = std::move(inliers);
    res.inlier_count = static_cast<int>(res.inliers.size());
    return res;
}


} 
