#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <vo_core/feature_extractor.hpp>

int main() {
    using clock = std::chrono::high_resolution_clock;

    std::string input_path = "/Users/georgeilli/Desktop/Projects/VO/src/vo_core/test/data/img1.png";
    std::string output_path = "/Users/georgeilli/Desktop/Projects/VO/src/vo_core/test/data/feature_img1.png";

    auto t0 = clock::now();
    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    auto t1 = clock::now();
    if (img.empty()) {
        std::cerr << "Failed to read image: " << input_path << std::endl;
        return 1;
    }
    std::cout << "Image size:     " << img.cols << " x " << img.rows << "\n";

    vo_core::FeatureExtractor extractor(1000, 1.5, 4);

    auto t2 = clock::now();
    extractor.extract(img);
    auto t3 = clock::now();

    const vo_core::FeatureResult& res = extractor.result();

    auto t4 = clock::now();
    cv::Mat img_drawn = img.clone();
    for (int i = 0; i < res.points.rows; ++i) {
        float x = res.points.at<float>(i, 0);
        float y = res.points.at<float>(i, 1);
        cv::Point center(cvRound(x), cvRound(y));
        cv::Scalar green(0, 255, 0);
        cv::circle(img_drawn, center, 2, green, cv::FILLED);
        int r = 4;
        cv::rectangle(img_drawn, 
                      cv::Point(center.x - r, center.y - r),
                      cv::Point(center.x + r, center.y + r),
                      green, 1);
    }
    auto t5 = clock::now();

    auto t6 = clock::now();
    if (!cv::imwrite(output_path, img_drawn)) {
        std::cerr << "Failed to write image: " << output_path << std::endl;
        return 1;
    }
    auto t7 = clock::now();

    auto dur = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };

    std::cout << "Read image:     " << dur(t0, t1) << " ms\n";
    std::cout << "Extract:        " << dur(t2, t3) << " ms\n";
    std::cout << "Draw features:  " << dur(t4, t5) << " ms\n";
    std::cout << "Write image:    " << dur(t6, t7) << " ms\n";

    return 0;
}
