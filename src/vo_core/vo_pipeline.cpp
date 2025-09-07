#include "vo_core/vo_pipeline.hpp"
#include <opencv2/calib3d.hpp>
#include <stdexcept>

namespace vo_core {

void composePoseIncrement(Pose& pose, const cv::Mat& delta_R, const cv::Mat& delta_t) {
    CV_Assert(delta_R.rows == 3 && delta_R.cols == 3 && delta_R.type() == CV_64F);
    CV_Assert(delta_t.total() == 3 && (delta_t.rows == 3 || delta_t.cols == 3) && delta_t.type() == CV_64F);

    const cv::Quatd q_delta = mat3x3ToQuat(delta_R);       // unit quaternion
    const cv::Quatd q_new   = (pose.Q * q_delta.conjugate()).normalize();

    const cv::Vec3d t_delta(
        delta_t.at<double>(0),
        delta_t.at<double>(1),
        delta_t.at<double>(2)
    );

    const cv::Matx33d Rw_new = q_new.toRotMat3x3(cv::QUAT_ASSUME_UNIT);
    const cv::Vec3d t_world  = cv::Vec3d(
        Rw_new(0,0)*t_delta[0] + Rw_new(0,1)*t_delta[1] + Rw_new(0,2)*t_delta[2],
        Rw_new(1,0)*t_delta[0] + Rw_new(1,1)*t_delta[1] + Rw_new(1,2)*t_delta[2],
        Rw_new(2,0)*t_delta[0] + Rw_new(2,1)*t_delta[1] + Rw_new(2,2)*t_delta[2]
    );

    pose.T -= t_world;
    pose.Q  = q_new;
}

struct VOPipeline::Impl {
    cv::Mat prev_rgb;
    cv::Mat prev_depth;
    std::vector<cv::KeyPoint> prev_kpts;
    cv::Mat prev_desc;
    int frame_idx{0};
    double depth_scale{1.0};
    Impl() = default;
};

VOPipeline::VOPipeline(const PipelineConfig& cfg)
{
    camera_ = cfg.camera;
    pose_world_ = cfg.initial_pose;
    impl_ = std::make_unique<Impl>();
    extra_cfg_ = cfg.extra_cfg;

    if (cfg.K.empty()) {
        K_ = cv::Mat::eye(3, 3, CV_64F);
    } else if (cfg.K.type() == CV_64F) {
        K_ = cfg.K.clone();
    } else {
        cfg.K.convertTo(K_, CV_64F);
    }

    if (K_.rows != 3 || K_.cols != 3) {
        throw std::invalid_argument("K must be 3x3.");
    }

    if (cfg.dist.empty()) {
        dist_ = cv::Mat();
    } else if (cfg.dist.type() == CV_64F) {
        dist_ = cfg.dist.clone();
    } else {
        cfg.dist.convertTo(dist_, CV_64F);
    }
}

VOPipeline::~VOPipeline() = default;

void VOPipeline::reset(const Pose& initial_pose) {
    pose_world_ = initial_pose;
    initialized_ = false;
    impl_.reset(new Impl());
}

void VOPipeline::setIntrinsics(const cv::Mat& K) {
    if (K.empty() || K.rows != 3 || K.cols != 3) throw std::invalid_argument("K must be 3x3.");
    if (K.type() == CV_64F) K_ = K.clone(); else K.convertTo(K_, CV_64F);
}

void VOPipeline::setDistortion(const cv::Mat& dist) {
    if (dist.empty()) { dist_.release(); return; }
    if (dist.type() == CV_64F) dist_ = dist.clone(); else dist.convertTo(dist_, CV_64F);
}

void VOPipeline::setCameraType(CameraType type) {
    camera_ = type;
    initialized_ = false;
}

void VOPipeline::initialize(const cv::Mat& rgb, const cv::Mat& depth) {
    if (camera_ == CameraType::MONO) {
        initMono(rgb);
    } else {
        initRGBD(rgb, depth);
    }
    initialized_ = true;
}


void VOPipeline::initMono(const cv::Mat& rgb) {
    (void)rgb;
    impl_->frame_idx = 0;
}


void VOPipeline::initRGBD(const cv::Mat& rgb, const cv::Mat& depth) {
    (void)rgb; (void)depth;
    impl_->frame_idx = 0;
}



PoseResult VOPipeline::processFrame(const cv::Mat& rgb, const cv::Mat& depth) {
    if (!initialized_) initialize(rgb, depth);
    PoseResult pr;
    if (camera_ == CameraType::MONO) return pr;

    vo_core::FeatureExtractor extractor(extra_cfg_[0], extra_cfg_[1], extra_cfg_[2]);
    extractor.extract(rgb);
    const vo_core::FeatureResult& cur = extractor.result();

    std::vector<cv::KeyPoint> kps_curr;
    kps_curr.reserve(cur.points.rows);
    for (int i = 0; i < cur.points.rows; ++i) {
        const cv::Vec2f p = cur.points.at<cv::Vec2f>(i, 0);
        kps_curr.emplace_back(cv::Point2f(p[0], p[1]), 1.f);
    }
    const cv::Mat& desc_curr = cur.descriptors;

    if (impl_->prev_desc.empty()) {
        impl_->prev_rgb   = rgb.clone();
        impl_->prev_depth = depth.clone();
        impl_->prev_kpts  = kps_curr;
        impl_->prev_desc  = desc_curr.clone();
        impl_->frame_idx  = 0;
        return pr;
    }

    std::vector<cv::DMatch> matches;
    vo_core::FeatureMatcher matcher;
    matcher.match(
        kps_curr, desc_curr, rgb.size(),
        impl_->prev_kpts, impl_->prev_desc, impl_->prev_rgb.size(),
        matches
    );

    const double fx = K_.at<double>(0,0), fy = K_.at<double>(1,1);
    const double cx = K_.at<double>(0,2), cy = K_.at<double>(1,2);

    const bool prev_u16 = impl_->prev_depth.type() == CV_16UC1;
    const bool prev_f32 = impl_->prev_depth.type() == CV_32FC1;
    double depth_scale = impl_->depth_scale;
    if (prev_u16 && depth_scale <= 0.0) depth_scale = 1.0 / 1000.0;

    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    pts3d.reserve(matches.size());
    pts2d.reserve(matches.size());

    const int W = impl_->prev_depth.cols, H = impl_->prev_depth.rows;
    for (const auto& m : matches) {
        const cv::Point2f pt_prev = impl_->prev_kpts[m.trainIdx].pt;
        const int u = static_cast<int>(std::lround(pt_prev.x));
        const int v = static_cast<int>(std::lround(pt_prev.y));
        if (u < 0 || v < 0 || u >= W || v >= H) continue;

        double Z = 0.0;
        if (prev_u16) {
            const uint16_t d = impl_->prev_depth.at<uint16_t>(v, u);
            if (d == 0) continue;
            Z = static_cast<double>(d) * depth_scale;
        } else if (prev_f32) {
            const float d = impl_->prev_depth.at<float>(v, u);
            if (!std::isfinite(d) || d <= 0.f) continue;
            Z = static_cast<double>(d);
        } else {
            continue;
        }

        cv::Mat src(1, 1, CV_32FC2);
        src.at<cv::Vec2f>(0,0) = cv::Vec2f(pt_prev.x, pt_prev.y);

        cv::Mat und; 
        cv::undistortPoints(src, und, K_, dist_); 
        const cv::Vec2f n = und.at<cv::Vec2f>(0,0);

        const double X = static_cast<double>(n[0]) * Z;
        const double Y = static_cast<double>(n[1]) * Z;
        pts3d.emplace_back(static_cast<float>(X), static_cast<float>(Y), static_cast<float>(Z));


        const cv::Point2f pt_curr = kps_curr[m.queryIdx].pt;
        pts2d.emplace_back(pt_curr);
    }

    if (pts3d.size() >= 4) {
        vo_core::PoseEstimator estimator(K_, dist_);
        pr = estimator.estimate(pts3d, pts2d, true);
        if (pr.success) {
            if (pr.R.empty() && !pr.rvec.empty()) cv::Rodrigues(pr.rvec, pr.R);
            if (pr.R.type() != CV_64F) pr.R.convertTo(pr.R, CV_64F);
            if (pr.tvec.type() != CV_64F) pr.tvec.convertTo(pr.tvec, CV_64F);
            composePoseIncrement(pose_world_, pr.R, pr.tvec);
        }
    }

    impl_->prev_rgb   = rgb.clone();
    impl_->prev_depth = depth.clone();
    impl_->prev_kpts  = kps_curr;
    impl_->prev_desc  = desc_curr.clone();
    ++impl_->frame_idx;

    return pr;
}


} // namespace vo_core

