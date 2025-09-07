#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <unordered_map>
#include <vector>
#include <memory>

#include "geometry/geometry.hpp"

namespace vo_core {

using KfId = int;
using MpId  = int;

struct MapPoint {
    MpId id{-1};
    cv::Point3d pos;          // world coordinates
    cv::Mat     desc;         // ORB descriptor
    int visible{0};           // counter of how many times it showed up 
    int matched{0};           // counter of how many times it is an "inlier"
    std::unordered_map<KfId, size_t> obs; //store keyframe id

    int created_at{0};     // frame index when created
    int last_seen{-1};     // last frame it was an inlier
    int last_visible{-1};
};


struct Keyframe {
    KfId id{-1};
    Pose T_w_c;                   // camera pose
    cv::Mat K;                    
    cv::Mat dist;                 
    cv::Size img_size;

    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;                 

    cv::Mat depth;                
    double  depth_scale{1.0};

    // 2D -> 3D association (keypoint -> MapPoint)
    std::unordered_map<size_t, MpId> kp_to_mp;
};


struct BAObservation {
    KfId kf;
    MpId mp;
    size_t kp;
    cv::Point2f uv;
};

struct BAInput {
    std::vector<Pose*>        poses;      
    std::vector<KfId>         pose_ids;   

    std::vector<cv::Point3d*> points;     
    std::vector<MpId>         point_ids;  

    std::vector<BAObservation> obs;      
    cv::Mat K;                           
};

struct Map {


    std::unordered_map<KfId, std::shared_ptr<Keyframe>> keyframes;
    std::unordered_map<MpId, std::shared_ptr<MapPoint>>  points;
    std::vector<KfId> recent_kfs; 

    int next_kf{0};
    int next_mp{0};

    // Parameters
    int    local_window      = 10;    // for Bundle Adjustment
    int    min_obs           = 2;    
    double min_match_ratio   = 0.20; 
    const int grace_new_frames      = 3;  
    const int min_visible_for_ratio = 15; 
    const int max_miss_frames       = 60;
    const int protect_recent_kf     = 5;   

    // Culling
    
    KfId addKeyframe(const Keyframe& kf_in);
    MpId addMapPoint(const cv::Point3d& p_w, const cv::Mat& orb_desc,
                 KfId created_by, size_t kp_idx, int current_frame);

    void addObservation(KfId kf, size_t kp_idx, MpId mp);

    void cull(int current_frame);


    BAInput buildBALocalInput() const;
    BAInput buildBALocalInputN(int n_kfs) const;
    std::vector<KfId> lastNKFs(int n_kfs) const;
};

}