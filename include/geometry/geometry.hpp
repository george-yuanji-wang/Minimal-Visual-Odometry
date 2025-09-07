#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/quaternion.hpp>

namespace vo_core {

struct Pose {
    cv::Quatd Q;   
    cv::Vec3d T;  

    Pose() : Q(1,0,0,0), T(0,0,0) {}
    cv::Mat asMatrix4x4() const;
};


cv::Quatd mat3x3ToQuat(const cv::Mat& R);

cv::Mat quatToMat3x3(const cv::Quatd& q);

cv::Vec3d rotateVecUnitQuat(const cv::Quatd& q_unit, const cv::Vec3d& v);

void setPoseFromTcw(Pose& dst, const cv::Mat& Rcw, const cv::Mat& tcw);

} // namespace vo_core
