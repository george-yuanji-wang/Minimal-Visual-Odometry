#include <opencv2/opencv.hpp>
#include <vo_core/feature_extractor.hpp>
#include <string>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return 1;

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    vo_core::FeatureExtractor extractor(1000, 1.2, 8);

    cv::Mat frame, gray, vis;
    const cv::Scalar green(0,255,0);
    const int r = 4;

    double prev_time = static_cast<double>(cv::getTickCount());
    double freq = cv::getTickFrequency();

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        double curr_time = static_cast<double>(cv::getTickCount());
        double elapsed = (curr_time - prev_time) / freq;
        prev_time = curr_time;
        double fps = elapsed > 0.0 ? 1.0 / elapsed : 0.0;

        if (frame.cols != 640 || frame.rows != 480) cv::resize(frame, frame, cv::Size(640,480));
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        extractor.extract(gray);
        const auto& res = extractor.result();

        cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
        for (int i = 0; i < res.points.rows; ++i) {
            const float x = res.points.at<float>(i,0);
            const float y = res.points.at<float>(i,1);
            const cv::Point c(cvRound(x), cvRound(y));
            cv::circle(vis, c, 2, green, cv::FILLED);
            cv::rectangle(vis, {c.x - r, c.y - r}, {c.x + r, c.y + r}, green, 1);
        }

        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(vis, fps_text, {10, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);

        cv::imshow("cam_features (q to quit)", vis);
        const int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }
    return 0;
}
