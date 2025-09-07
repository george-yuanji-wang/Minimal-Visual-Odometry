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
#include <chrono>

using Clock = std::chrono::high_resolution_clock;

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

int main()
{
    const std::string DATA_DIR  = "/Users/georgeilli/Downloads/rgbd_dataset_freiburg1_xyz";
    const std::string rgb_list  = DATA_DIR + "/rgb.txt";
    const std::string depth_list= DATA_DIR + "/depth.txt";
    const std::string gt_path   = DATA_DIR + "/groundtruth.txt";

    // --- Calibration ---
    cv::Mat K = (cv::Mat_<double>(3,3) << 
                517.306408, 0.0, 318.643040,
                0.0, 516.469215, 255.313989,
                0.0,   0.0,   1.0);

    cv::Mat dist = (cv::Mat_<double>(1,5) <<
                0.262383, -0.953104, -0.005358, 0.002628, 1.163314);

    std::vector<std::pair<double,std::string>> rgbs, depths;
    if (!parse_assoc_list(rgb_list, rgbs))   { std::cerr << "Failed to read " << rgb_list  << "\n"; return 1; }
    if (!parse_assoc_list(depth_list, depths)){ std::cerr << "Failed to read " << depth_list << "\n"; return 1; }

    std::vector<std::tuple<double, cv::Vec3d, cv::Vec4d>> gts;
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

    // --- Parameter matrix: customize combos here ---
    std::vector<std::array<double,3>> param_combos;
    for (int n_feat = 200; n_feat <= 1600; n_feat += 200) {
        for (double scale = 1.2; scale <= 1.5 + 1e-9; scale += 0.1) {
            for (int n_lvl : {1,2,4,6,8,10}) {
                param_combos.push_back({static_cast<double>(n_feat), scale, static_cast<double>(n_lvl)});
            }
        }
    }
    //200 -> 1600, stp: 200
    //1.2 -> 1.5, stp:0.1
    //1,2,4,6,8,10


    std::ofstream results_csv("../test/data/tune_results.csv");
    results_csv << "index,params,rmse_m,mean_rot_deg,fps\n";

    const size_t N = std::min(rgbs.size(), depths.size());
    const float DEPTH_SCALE = 1.0f / 5000.0f; // TUM fr1 -> meters

    for (size_t combo_idx = 0; combo_idx < param_combos.size(); ++combo_idx) {
        const auto& combo = param_combos[combo_idx];

        // --- Build config ---
        vo_core::PipelineConfig cfg;
        cfg.camera = vo_core::CameraType::RGBD;
        cfg.K = K.clone();
        cfg.dist = dist.clone();
        cfg.extra_cfg = {combo[0], combo[1], combo[2]};

        // --- Initial pose ---
        const double ts_rgb0 = rgbs.front().first;
        const size_t gt0_idx = find_nearest_gt(ts_rgb0);
        const auto& [t0, T0, Q0] = gts[gt0_idx]; (void)t0;
        vo_core::Pose init;
        init.T = T0;
        init.Q = cv::Quatd(Q0[0], Q0[1], Q0[2], Q0[3]).normalize();
        cfg.initial_pose = init;

        vo_core::VOPipeline vo(cfg);

        // --- Prime pipeline with first frame ---
        {
            const std::string rgb0_path   = DATA_DIR + "/" + rgbs[0].second;
            const std::string depth0_path = DATA_DIR + "/" + depths[0].second;
            cv::Mat rgb0 = cv::imread(rgb0_path, cv::IMREAD_COLOR);
            cv::Mat depth0_raw = cv::imread(depth0_path, cv::IMREAD_UNCHANGED);
            if (rgb0.empty() || depth0_raw.empty()) { 
                std::cerr << "Failed to load first pair.\n"; return 1; 
            }
            cv::Mat depth0_m;
            if (depth0_raw.type() == CV_16UC1) depth0_raw.convertTo(depth0_m, CV_32FC1, DEPTH_SCALE);
            else if (depth0_raw.type() == CV_32FC1) depth0_m = depth0_raw;
            else { std::cerr << "Unsupported depth type at frame 0\n"; return 1; }
            vo.initialize(rgb0, depth0_m);
        }

        double sum_sq_trans_err = 0.0;
        double sum_rot_deg = 0.0;
        size_t err_count = 0;

        auto start = Clock::now();

        for (size_t i = 1; i < N; ++i) {
            const double ts_rgb   = rgbs[i].first;
            const std::string rgb_path   = DATA_DIR + "/" + rgbs[i].second;
            const std::string depth_path = DATA_DIR + "/" + depths[i].second;

            cv::Mat rgb = cv::imread(rgb_path, cv::IMREAD_COLOR);
            cv::Mat depth_raw = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
            if (rgb.empty() || depth_raw.empty()) continue;

            cv::Mat depth_m;
            if (depth_raw.type() == CV_16UC1) depth_raw.convertTo(depth_m, CV_32FC1, DEPTH_SCALE);
            else if (depth_raw.type() == CV_32FC1) depth_m = depth_raw;
            else continue;

            vo_core::PoseResult pr = vo.processFrame(rgb, depth_m); (void)pr;

            const vo_core::Pose& est_pose = vo.currentPose();
            const size_t j = find_nearest_gt(ts_rgb);
            const auto& [t_gt, T_gt, Q_gt] = gts[j]; (void)t_gt;

            // Errors
            cv::Vec3d dT = est_pose.T - T_gt;
            double trans_err = std::sqrt(dT.dot(dT));
            sum_sq_trans_err += trans_err * trans_err;

            const cv::Quatd qg(Q_gt[0], Q_gt[1], Q_gt[2], Q_gt[3]);
            cv::Quatd dq = qg.conjugate() * est_pose.Q;
            dq = dq.normalize();
            double w = std::abs(dq.w); if (w > 1.0) w = 1.0;
            double ang_rad = 2.0 * std::acos(w);
            double ang_deg = ang_rad * 180.0 / M_PI;
            sum_rot_deg += ang_deg;

            ++err_count;
        }

        auto end = Clock::now();
        double elapsed_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        double fps = (err_count > 0) ? static_cast<double>(err_count) / elapsed_sec : 0.0;

        // --- Stats ---
        double rmse_m = (err_count > 0) ? std::sqrt(sum_sq_trans_err / err_count) : 0.0;
        double mean_rot_deg = (err_count > 0) ? sum_rot_deg / err_count : 0.0;

        std::ostringstream param_str;
        param_str << "(" << combo[0] << "," << combo[1] << "," << combo[2] << ")";

        results_csv << combo_idx << ","
                    << "\"" << param_str.str() << "\","
                    << std::fixed << std::setprecision(6)
                    << rmse_m << ","
                    << mean_rot_deg << ","
                    << fps << "\n";

        std::cout << "Combo " << combo_idx << " done.\n";
    }

    results_csv.close();
    std::cout << "Wrote tune_results.csv\n";
    return 0;
}
