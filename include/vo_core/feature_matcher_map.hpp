#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "vo_core/feature_matcher.hpp"
#include "map/map.hpp" 

namespace vo_core {

struct MapPoint;

void projectMapPoints(const std::vector<std::shared_ptr<MapPoint>>& mapPts,
                      const cv::Mat& Tcw,   
                      const cv::Mat& K,     
                      cv::Size imgSize,
                      std::vector<cv::KeyPoint>& kpsProj,
                      cv::Mat& descProj);

void matchMapProjections(const FeatureMatcher& fm,
                         const std::vector<cv::KeyPoint>& kpsMp,
                         const cv::Mat&                   descMp,
                         cv::Size                         imgSize,
                         const std::vector<cv::KeyPoint>& kpsCurr,
                         const cv::Mat&                   descCurr,
                         std::vector<cv::DMatch>&         matches);

void buildMapCorrespondences(const std::vector<cv::DMatch>&                matches,
                             const std::vector<std::shared_ptr<MapPoint>>& mapPts,
                             const std::vector<cv::KeyPoint>&              kpsCurr,
                             std::vector<cv::Point3f>&                     pts3d,
                             std::vector<cv::Point2f>&                     pts2d,
                             std::vector<int>*                             mpIndices = nullptr,
                             std::vector<int>*                             kpIndices = nullptr);

void buildFrameMapCorrespondences(const FeatureMatcher& fm,
                                  const std::vector<std::shared_ptr<MapPoint>>& mapPts,
                                  const cv::Mat& Tcw,
                                  const cv::Mat& K,
                                  cv::Size imgSize,
                                  const std::vector<cv::KeyPoint>& kpsCurr,
                                  const cv::Mat& descCurr,
                                  std::vector<cv::Point3f>& pts3d,
                                  std::vector<cv::Point2f>& pts2d,
                                  std::vector<int>* mpIndices = nullptr,
                                  std::vector<int>* kpIndices = nullptr);

} // namespace vo_core
