#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cctype>

static std::vector<cv::Point3f> makeObjectCorners(int cols, int rows, double s) {
    std::vector<cv::Point3f> obj; obj.reserve(cols*rows);
    for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) obj.emplace_back((float)c*(float)s,(float)r*(float)s,0.f);
    return obj;
}

int main() {
    int cols=0, rows=0, targetViews=0; double squareSize=0.0;
    std::cout << "Checkerboard inner corners (columns): "; std::cin >> cols;
    std::cout << "Checkerboard inner corners (rows):    "; std::cin >> rows;
    std::cout << "Square size (e.g., 0.024 for 24mm):   "; std::cin >> squareSize;
    std::cout << "Number of views to collect (>=8):     "; std::cin >> targetViews;
    if (cols < 2 || rows < 2 || squareSize <= 0.0 || targetViews < 8) { std::cerr << "Invalid inputs.\n"; return 1; }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { std::cerr << "Cannot open camera.\n"; return 1; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640); cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    const cv::Size patternSize(cols, rows);
    const auto obj = makeObjectCorners(cols, rows, squareSize);

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;

    cv::Mat frame, gray, vis;
    int accepted = 0, acceptFlash = 0;

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;
        if (frame.cols != 640 || frame.rows != 480) cv::resize(frame, frame, cv::Size(640,480));
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCornersSB(gray, patternSize, corners, cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);
        if (!found) {
            found = cv::findChessboardCorners(gray, patternSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
            if (found) cv::cornerSubPix(gray, corners, cv::Size(11,11), cv::Size(-1,-1),
                                        cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 30, 0.01));
        }

        cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
        cv::Scalar dotColor = found ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255);
        if (acceptFlash > 0) { dotColor = cv::Scalar(255,255,0); acceptFlash--; }
        for (const auto& pt : corners) cv::circle(vis, pt, 3, dotColor, cv::FILLED, cv::LINE_AA);

        std::string s_found = std::string("FOUND: ") + (found ? "YES" : "NO");
        std::string s_count = "Corners: " + std::to_string((int)corners.size()) + " / " + std::to_string(patternSize.width*patternSize.height);
        std::string s_views = "Views: " + std::to_string(accepted) + " / " + std::to_string(targetViews);
        if (found) cv::putText(vis, "READY: SPACE/ENTER to accept", {10, 96}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,255,255}, 1, cv::LINE_AA);

        cv::putText(vis, s_found, {10, 24},  cv::FONT_HERSHEY_SIMPLEX, 0.7, found ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), 2, cv::LINE_AA);
        cv::putText(vis, s_count, {10, 50},  cv::FONT_HERSHEY_SIMPLEX, 0.55, {255,255,0}, 1, cv::LINE_AA);
        cv::putText(vis, s_views, {10, 72},  cv::FONT_HERSHEY_SIMPLEX, 0.55, {200,255,200}, 1, cv::LINE_AA);
        cv::putText(vis, "Q/ESC=quit", {10, 118}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {255,255,255}, 1, cv::LINE_AA);

        cv::imshow("calibrate_camera", vis);
        int key = cv::waitKey(15) & 0xFF;
        if (key == 255 || key == -1) {
            if (accepted >= targetViews) key = 'c';
        }

        if (key == 'q' || key == 27) { cv::destroyAllWindows(); return 0; }

        if ((key == ' ' || key == '\r' || key == '\n') && found && accepted < targetViews) {
            imagePoints.push_back(corners);
            objectPoints.push_back(obj);
            accepted++;
            acceptFlash = 8;
            if (accepted >= targetViews) key = 'c';
            continue;
        }

        if (key == 'c' && accepted >= targetViews) {
            cv::Mat K, dist; std::vector<cv::Mat> rvecs, tvecs;
            double rms = cv::calibrateCamera(objectPoints, imagePoints, gray.size(), K, dist, rvecs, tvecs,
                                             cv::CALIB_RATIONAL_MODEL | cv::CALIB_FIX_K3,
                                             cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 100, 1e-6));

            std::cout << "\n=== Calibration Results ===\n";
            std::cout << "RMS reprojection error: " << rms << "\n";
            std::cout << "K:\n" << K << "\n";
            std::cout << "distCoeffs:\n" << dist.t() << "\n";

            double totalErr = 0.0; size_t totalPts = 0;
            for (size_t i = 0; i < objectPoints.size(); ++i) {
                std::vector<cv::Point2f> proj;
                cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], K, dist, proj);
                double err = cv::norm(proj, imagePoints[i], cv::NORM_L2);
                totalErr += err*err; totalPts += imagePoints[i].size();
            }
            double meanErr = std::sqrt(totalErr / (double)totalPts);
            std::cout << "Mean reprojection error (px): " << meanErr << "\n";

            cv::FileStorage fs("../camera_calib.yaml", cv::FileStorage::WRITE);
            if (fs.isOpened()) {
                fs << "image_width" << gray.cols;
                fs << "image_height" << gray.rows;
                fs << "board_width" << cols;
                fs << "board_height" << rows;
                fs << "square_size" << squareSize;
                fs << "camera_matrix" << K;
                fs << "distortion_coefficients" << dist;
                fs.release();
                std::cout << "Saved to ../camera_calib.yaml\n";
            } else {
                std::cout << "Failed to save ../camera_calib.yaml\n";
            }

            cv::destroyAllWindows();
            return 0;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
