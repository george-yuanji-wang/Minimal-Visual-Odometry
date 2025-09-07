#include <opencv2/opencv.hpp>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "vo_core/feature_extractor.hpp"
#include "vo_core/feature_matcher.hpp"

using namespace vo_core;

static std::vector<cv::KeyPoint> ptsToKps(const cv::Mat& ptsN2) {
    std::vector<cv::KeyPoint> kps; kps.reserve(ptsN2.rows);
    for (int i=0;i<ptsN2.rows;++i) {
        cv::KeyPoint k; 
        k.pt.x = ptsN2.at<float>(i,0);
        k.pt.y = ptsN2.at<float>(i,1);
        k.octave = 0; k.angle = 0.f; k.size = 31.f; 
        kps.push_back(k);
    }
    return kps;
}

int main() {
    const int W=640, H=480;
    cv::VideoCapture cap(0);
    if(!cap.isOpened()) return 1;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, H);

    FeatureExtractor extractor(1400.0, 1.2, 2);

    MatchParams prm;
    prm.baseRadius = 40.f;
    prm.scaleFactor = 1.2f;
    prm.maxLevelDiff = 1;
    prm.thLow = 50;
    prm.ratio = 0.8f;
    prm.orientationBins = 30;
    prm.keepBins = 3;
    prm.symmetric = true;
    prm.cellW = 35;
    prm.cellH = 35;
    FeatureMatcher matcher(prm);

    cv::Mat prevGray, curGray, frame, leftBGR, rightBGR, canvas, overlay;
    std::vector<cv::KeyPoint> kpsPrev, kpsCur;
    cv::Mat descPrev, descCur;

    if(!cap.read(frame)) return 1;
    cv::cvtColor(frame, prevGray, cv::COLOR_BGR2GRAY);
    cv::resize(prevGray, prevGray, cv::Size(W,H));
    extractor.extract(prevGray);
    const auto& r = extractor.result();
    kpsPrev = ptsToKps(r.points);
    descPrev = r.descriptors.clone();

    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> col(0,255);
    auto t_last = std::chrono::steady_clock::now();
    double ema_fps = 0.0;

    double ms_total_last = 0.0;

    for(;;) {
        auto t0 = std::chrono::steady_clock::now();
        if(!cap.read(frame)) break;
        cv::cvtColor(frame, curGray, cv::COLOR_BGR2GRAY);
        cv::resize(curGray, curGray, cv::Size(W,H));

        auto t_ext0 = std::chrono::steady_clock::now();
        extractor.extract(curGray);
        auto t_ext1 = std::chrono::steady_clock::now();

        const auto& r2 = extractor.result();
        kpsCur  = ptsToKps(r2.points);
        descCur = r2.descriptors.clone();

        std::vector<cv::DMatch> matches;
        auto t_mat0 = std::chrono::steady_clock::now();
        matcher.match(kpsPrev, descPrev, prevGray.size(),
                      kpsCur,  descCur,  curGray.size(), matches);
        auto t_mat1 = std::chrono::steady_clock::now();

        cv::cvtColor(prevGray, leftBGR,  cv::COLOR_GRAY2BGR);
        cv::cvtColor(curGray,  rightBGR, cv::COLOR_GRAY2BGR);
        canvas.create(H, W*2, CV_8UC3);
        leftBGR.copyTo(canvas.colRange(0, W));
        rightBGR.copyTo(canvas.colRange(W, 2*W));
        overlay = canvas.clone();

        rng.seed(0);
        for(const auto& m : matches) {
            cv::Point2f p1 = kpsPrev[m.queryIdx].pt;
            cv::Point2f p2 = kpsCur [m.trainIdx].pt + cv::Point2f((float)W, 0.f);
            cv::Scalar c(col(rng), col(rng), col(rng));
            cv::circle(overlay, p1, 3, c, cv::FILLED, cv::LINE_AA);
            cv::circle(overlay, p2, 3, c, cv::FILLED, cv::LINE_AA);
            cv::line(overlay, p1, p2, c, 1, cv::LINE_AA);
        }
        cv::addWeighted(overlay, 0.5, canvas, 0.5, 0.0, canvas);

        double ms_ext = std::chrono::duration_cast<std::chrono::milliseconds>(t_ext1 - t_ext0).count();
        double ms_mat = std::chrono::duration_cast<std::chrono::milliseconds>(t_mat1 - t_mat0).count();
        ms_total_last = ms_ext + ms_mat;

        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t_last).count();
        t_last = t1;
        double fps = dt > 0.0 ? 1.0/dt : 0.0;
        ema_fps = (ema_fps==0.0) ? fps : (0.9*ema_fps + 0.1*fps);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << ms_total_last << " ms";
        cv::putText(canvas, oss.str(), cv::Point(10,30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,255,0), 2, cv::LINE_AA);

        cv::imshow("VO realtime (t-1 | t)", canvas);
        int k = cv::waitKey(1);
        if(k==27 || k=='q') break;

        prevGray = curGray.clone();
        kpsPrev.swap(kpsCur);
        descPrev = descCur.clone();
    }
    return 0;
}
