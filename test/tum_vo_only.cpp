#include <opencv2/opencv.hpp>
#include <vo_core/vo_pipeline.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <tuple>
#include <string>
#include <iostream>
#include <tuple>    
#include <set>      
#include <algorithm>

static bool parse_assoc_list(const std::string& list_path,
                             std::vector<std::pair<double,std::string>>& out)
{
    std::ifstream f(list_path);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double ts; std::string rel;
        if (!(ss >> ts >> rel)) continue;
        out.emplace_back(ts, rel);
    }
    return true;
}

static bool parse_groundtruth(const std::string& path,
                              std::vector<std::tuple<double, cv::Vec3d, cv::Vec4d>>& gt)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0]=='#') continue;
        std::istringstream ss(line);
        double t; double tx,ty,tz, qx,qy,qz,qw;
        if (!(ss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) continue;
        gt.emplace_back(t, cv::Vec3d(tx,ty,tz), cv::Vec4d(qw,qx,qy,qz));
    }
    return true;
}

// ---------- NEW: TUM-style RGB–Depth association ----------
struct AssocPair {
    double ts;               // use RGB timestamp as row timestamp
    std::string rgb_rel;
    std::string depth_rel;
};

static std::vector<AssocPair> associate_rgb_depth(
    const std::vector<std::pair<double,std::string>>& rgbs,
    const std::vector<std::pair<double,std::string>>& depths,
    double offset = 0.0,
    double max_diff = 0.02)
{
    std::vector<std::tuple<double,size_t,size_t>> potentials;
    for (size_t i = 0; i < rgbs.size(); ++i) {
        const double ta = rgbs[i].first;
        for (size_t j = 0; j < depths.size(); ++j) {
            const double tb = depths[j].first;
            const double diff = std::abs(ta - (tb + offset));
            if (diff < max_diff) potentials.emplace_back(diff, i, j);
        }
    }
    std::sort(potentials.begin(), potentials.end(),
              [](const auto& a, const auto& b){ return std::get<0>(a) < std::get<0>(b); });
    std::set<size_t> used_a, used_b;
    std::vector<AssocPair> out;
    out.reserve(potentials.size());
    for (const auto& p : potentials) {
        size_t ia = std::get<1>(p), ib = std::get<2>(p);
        if (used_a.count(ia) || used_b.count(ib)) continue;
        used_a.insert(ia); used_b.insert(ib);
        out.push_back({ rgbs[ia].first, rgbs[ia].second, depths[ib].second });
    }
    std::sort(out.begin(), out.end(), [](const AssocPair& x, const AssocPair& y){
        return x.ts < y.ts;
    });
    return out;
}
// ---------------------------------------------------------

int main()
{
    const std::string DATA_DIR  = "/Users/georgeilli/Downloads/rgbd_dataset_freiburg1_xyz";
    const std::string rgb_list  = DATA_DIR + "/rgb.txt";
    const std::string depth_list= DATA_DIR + "/depth.txt";
    const std::string gt_path   = DATA_DIR + "/groundtruth.txt";

    cv::Mat K = (cv::Mat_<double>(3,3) << 
                517.306408, 0.0, 318.643040,
                0.0, 516.469215, 255.313989,
                0.0,   0.0,   1.0);

    cv::Mat dist = (cv::Mat_<double>(1,5) <<
                0.262383, -0.953104, -0.005358, 0.002628, 1.163314);

    vo_core::PipelineConfig cfg;
    cfg.camera = vo_core::CameraType::RGBD;
    cfg.K = K.clone();
    cfg.dist = dist.clone();
    cfg.extra_cfg = {1400.0, 1.2, 2.0};

    std::vector<std::pair<double,std::string>> rgbs, depths;
    if (!parse_assoc_list(rgb_list, rgbs))    { std::cerr << "Failed to read " << rgb_list  << "\n"; return 1; }
    if (!parse_assoc_list(depth_list, depths)){ std::cerr << "Failed to read " << depth_list << "\n"; return 1; }

    // NEW: build associated RGB–Depth pairs
    auto pairs = associate_rgb_depth(rgbs, depths, /*offset=*/0.0, /*max_diff=*/0.02);
    if (pairs.empty()) { std::cerr << "No associated RGB–Depth pairs.\n"; return 1; }

    std::vector<std::tuple<double, cv::Vec3d, cv::Vec4d>> gts; // ts, T, (qw,qx,qy,qz)
    if (!parse_groundtruth(gt_path, gts)) { std::cerr << "Failed to read " << gt_path << "\n"; return 1; }
    if (gts.empty()) { std::cerr << "No GT data.\n"; return 1; }

    auto find_nearest_gt = [&](double ts)->size_t {
        size_t lo = 0, hi = gts.size();
        while (lo + 1 < hi) {
            size_t mid = (lo + hi) / 2;
            if (std::get<0>(gts[mid]) <= ts) lo = mid; else hi = mid;
        }
        if (lo + 1 < gts.size()) {
            double t0 = std::get<0>(gts[lo]);
            double t1 = std::get<0>(gts[lo+1]);
            return (std::abs(t1 - ts) < std::abs(ts - t0)) ? (lo+1) : lo;
        }
        return lo;
    };

    const double ts_rgb0 = pairs.front().ts; // CHANGED
    const size_t gt0_idx = find_nearest_gt(ts_rgb0);
    {
        const auto& [t0, T0, Q0] = gts[gt0_idx]; (void)t0;
        vo_core::Pose init;
        init.T = T0;
        init.Q = cv::Quatd(Q0[0], Q0[1], Q0[2], Q0[3]).normalize(); // (w,x,y,z)
        cfg.initial_pose = init;
    }

    vo_core::VOPipeline vo(cfg);

    std::ofstream fe_csv("../test/data/est_traj_vo.csv"),
                  fg_csv("../test/data/gt_traj_vo.csv");
    if (!fe_csv.is_open() || !fg_csv.is_open()) {
        std::cerr << "Failed to open output CSV files.\n";
        return 1;
    }
    fe_csv << "time,tx,ty,tz,qx,qy,qz,qw\n";
    fg_csv << "time,tx,ty,tz,qx,qy,qz,qw\n";

    const size_t N = pairs.size();      // CHANGED
    const float DEPTH_SCALE = 1.0f / 5000.0f;

    // Write explicit initial row for BOTH CSVs using the same (associated) RGB timestamp
    {
        const auto& [t0, T0, Q0] = gts[gt0_idx]; (void)t0;
        const vo_core::Pose& P0 = vo.currentPose();
        const cv::Quatd qe0 = P0.Q;
        fe_csv << std::fixed << std::setprecision(6)
               << ts_rgb0 << ","
               << P0.T[0] << "," << P0.T[1] << "," << P0.T[2] << ","
               << qe0.x << "," << qe0.y << "," << qe0.z << "," << qe0.w << "\n";
        fg_csv << std::fixed << std::setprecision(6)
               << ts_rgb0 << ","
               << T0[0] << "," << T0[1] << "," << T0[2] << ","
               << Q0[1] << "," << Q0[2] << "," << Q0[3] << "," << Q0[0] << "\n";
    }

    // Prime pipeline with frame 0, then start loop at i=1
    {
        const std::string rgb0_path   = DATA_DIR + "/" + pairs[0].rgb_rel;   // CHANGED
        const std::string depth0_path = DATA_DIR + "/" + pairs[0].depth_rel; // CHANGED
        cv::Mat rgb0 = cv::imread(rgb0_path, cv::IMREAD_COLOR);
        cv::Mat depth0_raw = cv::imread(depth0_path, cv::IMREAD_UNCHANGED);
        if (rgb0.empty() || depth0_raw.empty()) {
            std::cerr << "Failed to load first pair.\n";
            return 1;
        }
        cv::Mat depth0_m;
        if (depth0_raw.type() == CV_16UC1)       depth0_raw.convertTo(depth0_m, CV_32FC1, DEPTH_SCALE);
        else if (depth0_raw.type() == CV_32FC1)  depth0_m = depth0_raw;
        else { std::cerr << "Unsupported depth type at frame 0\n"; return 1; }
        vo.initialize(rgb0, depth0_m);
    }

    double sum_sq_trans_err = 0.0;
    double sum_rot_deg = 0.0;
    size_t err_count = 0;

    for (size_t i = 1; i < N; ++i) {
        const double ts_rgb         = pairs[i].ts;                 // CHANGED
        const std::string rgb_path  = DATA_DIR + "/" + pairs[i].rgb_rel;   // CHANGED
        const std::string depth_path= DATA_DIR + "/" + pairs[i].depth_rel; // CHANGED

        cv::Mat rgb = cv::imread(rgb_path, cv::IMREAD_COLOR);
        cv::Mat depth_raw = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
        if (rgb.empty() || depth_raw.empty()) {
            std::cerr << "Warning: failed to load pair " << i << "\n";
            continue;
        }

        cv::Mat depth_m;
        if (depth_raw.type() == CV_16UC1) {
            depth_raw.convertTo(depth_m, CV_32FC1, DEPTH_SCALE);
        } else if (depth_raw.type() == CV_32FC1) {
            depth_m = depth_raw;
        } else {
            std::cerr << "Unsupported depth type at frame " << i << "\n";
            continue;
        }

        vo_core::PoseResult pr = vo.processFrame(rgb, depth_m);
        (void)pr;

        const vo_core::Pose& est_pose = vo.currentPose();
        const cv::Quatd qe = est_pose.Q; // (w,x,y,z)
        fe_csv << std::fixed << std::setprecision(6)
               << ts_rgb << ","
               << est_pose.T[0] << "," << est_pose.T[1] << "," << est_pose.T[2] << ","
               << qe.x << "," << qe.y << "," << qe.z << "," << qe.w << "\n";

        const size_t j = find_nearest_gt(ts_rgb);
        const auto& [t_gt, T_gt, Q_gt] = gts[j]; (void)t_gt;
        fg_csv << std::fixed << std::setprecision(6)
               << ts_rgb << ","
               << T_gt[0] << "," << T_gt[1] << "," << T_gt[2] << ","
               << Q_gt[1] << "," << Q_gt[2] << "," << Q_gt[3] << "," << Q_gt[0] << "\n";

        // Errors vs nearest GT
        cv::Vec3d dT = est_pose.T - T_gt;
        const double trans_err = std::sqrt(dT.dot(dT));
        sum_sq_trans_err += trans_err * trans_err;

        const cv::Quatd qg(Q_gt[0], Q_gt[1], Q_gt[2], Q_gt[3]); // (w,x,y,z)
        cv::Quatd dq = qg.conjugate() * qe;
        dq = dq.normalize();
        double w = std::abs(dq.w); if (w > 1.0) w = 1.0;
        const double ang_rad = 2.0 * std::acos(w);
        const double ang_deg = ang_rad * 180.0 / 3.14159265358979323846;
        sum_rot_deg += ang_deg;

        ++err_count;
    }

    fe_csv.close();
    fg_csv.close();

    if (err_count > 0) {
        const double rmse_m = std::sqrt(sum_sq_trans_err / static_cast<double>(err_count));
        const double mean_rot_deg = sum_rot_deg / static_cast<double>(err_count);
        std::cout << "Frames evaluated: " << err_count << "\n";
        std::cout << "ATE RMSE (m): " << std::setprecision(6) << rmse_m << "\n";
        std::cout << "Mean rotation error (deg): " << std::setprecision(6) << mean_rot_deg << "\n";
    } else {
        std::cout << "No valid frames evaluated.\n";
    }

    std::cout << "Wrote est_traj.csv and gt_traj.csv\n";
    return 0;
}
