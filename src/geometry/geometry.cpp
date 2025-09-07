#include "geometry/geometry.hpp"

namespace vo_core {

cv::Mat Pose::asMatrix4x4() const {
    cv::Mat M = cv::Mat(Q.toRotMat4x4(cv::QUAT_ASSUME_UNIT)).clone();
    M.at<double>(0,3) = T[0];
    M.at<double>(1,3) = T[1];
    M.at<double>(2,3) = T[2];
    return M;
}

cv::Quatd mat3x3ToQuat(const cv::Mat& R) {
    CV_Assert(R.rows == 3 && R.cols == 3 && R.type() == CV_64F);
    return cv::Quatd::createFromRotMat(R).normalize();
}

cv::Mat quatToMat3x3(const cv::Quatd& q) {
    return cv::Mat(q.toRotMat3x3(cv::QUAT_ASSUME_UNIT)).clone();
}

cv::Vec3d rotateVecUnitQuat(const cv::Quatd& q_unit, const cv::Vec3d& v) {
    cv::Matx33d R = q_unit.toRotMat3x3(cv::QUAT_ASSUME_UNIT);
    return cv::Vec3d(
        R(0,0)*v[0] + R(0,1)*v[1] + R(0,2)*v[2],
        R(1,0)*v[0] + R(1,1)*v[1] + R(1,2)*v[2],
        R(2,0)*v[0] + R(2,1)*v[1] + R(2,2)*v[2]
    );
}

void setPoseFromTcw(Pose& dst, const cv::Mat& Rcw, const cv::Mat& tcw) {
    CV_Assert(Rcw.rows == 3 && Rcw.cols == 3 && Rcw.type() == CV_64F);
    CV_Assert(tcw.total() == 3 && (tcw.rows == 3 || tcw.cols == 3) && tcw.type() == CV_64F);
    cv::Mat Rwc = Rcw.t();
    cv::Mat twc = -Rwc * tcw;
    dst.Q = mat3x3ToQuat(Rwc);
    dst.T = cv::Vec3d(twc.at<double>(0), twc.at<double>(1), twc.at<double>(2));
}

} // namespace vo_core
