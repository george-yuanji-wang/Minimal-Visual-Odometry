#include <vo_core/feature_extractor.hpp>

namespace vo_core {

FeatureExtractor::FeatureExtractor(int n_features, double scale_factor, int n_levels) 
{
    n_features_  = n_features;
    scale_factor_ = scale_factor;
    n_levels_     = n_levels;

    orb_ = cv::ORB::create(
        n_features_,                           // nfeatures
        static_cast<float>(scale_factor_),     // scaleFactor
        n_levels_,                             // nlevels
        31,                                    // edgeThreshold
        0,                                     // firstLevel
        2,                                     // WTA_K
        cv::ORB::FAST_SCORE,                   // scoreType
        31,                                    // patchSize
        40                                     // fastThreshold
    );

    result_.points.create(0, 2, CV_32F);
    result_.descriptors.create(0, 32, CV_8U);
}


void FeatureExtractor::extract_prune(const cv::Mat& img_bgr, const cv::Mat& mask)
{
    // reset result
    result_.points.create(0, 2, CV_32F);
    result_.descriptors.create(0, 32, CV_8U);

    if (img_bgr.empty()) return;

    // convert to gray
    cv::Mat gray;
    if (img_bgr.channels() == 1) gray = img_bgr;
    else cv::cvtColor(img_bgr, gray, cv::COLOR_BGR2GRAY);

    // ORB detect+compute
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    orb_->detectAndCompute(gray, mask, kps, desc);

    if (kps.empty()) return;

    // ---- Grid pruning ----
    const int cellW = 32;   // tune: smaller → more even, slower
    const int cellH = 32;
    const int maxPerCell = 5;

    int cols = (gray.cols + cellW - 1) / cellW;
    int rows = (gray.rows + cellH - 1) / cellH;
    std::vector<std::vector<int>> buckets(cols*rows);

    // assign each kp to a grid bucket
    for (int i = 0; i < (int)kps.size(); ++i) {
        int cx = std::min(cols-1, (int)(kps[i].pt.x / cellW));
        int cy = std::min(rows-1, (int)(kps[i].pt.y / cellH));
        buckets[cy*cols + cx].push_back(i);
    }

    // pick top responses per bucket
    std::vector<cv::KeyPoint> keptKps;
    std::vector<int> keptIdx;
    for (auto& bucket : buckets) {
        if (bucket.empty()) continue;
        std::sort(bucket.begin(), bucket.end(), [&](int a, int b){
            return kps[a].response > kps[b].response;
        });
        int take = std::min((int)bucket.size(), maxPerCell);
        for (int j = 0; j < take; ++j) {
            keptKps.push_back(kps[bucket[j]]);
            keptIdx.push_back(bucket[j]);
        }
    }

    // build pruned descriptors
    cv::Mat prunedDesc((int)keptIdx.size(), desc.cols, desc.type());
    for (int i = 0; i < (int)keptIdx.size(); ++i) {
        desc.row(keptIdx[i]).copyTo(prunedDesc.row(i));
    }

    // write result
    result_.descriptors = std::move(prunedDesc);

    std::vector<cv::Point2f> pts;
    cv::KeyPoint::convert(keptKps, pts);
    result_.points = cv::Mat(pts, true).reshape(1, (int)pts.size());
}

void FeatureExtractor::extract(const cv::Mat& img_bgr, const cv::Mat& mask) 
{
    if (img_bgr.empty()) {
        result_.points.create(0, 2, CV_32F);
        result_.descriptors.create(0, 32, CV_8U);
        return;
    }

    // grayscale
    cv::Mat gray;
    if (img_bgr.channels() == 1) {
        gray = img_bgr;
    } else {
        cv::cvtColor(img_bgr, gray, cv::COLOR_BGR2GRAY);
    }

    // build pyramid
    std::vector<cv::Mat> pyramid(n_levels_);
    pyramid[0] = gray;
    for (int lvl = 1; lvl < n_levels_; ++lvl) {
        float scale = std::pow(scale_factor_, lvl);
        cv::resize(gray, pyramid[lvl],
                   cv::Size(std::round(gray.cols / scale),
                            std::round(gray.rows / scale)),
                   0, 0, cv::INTER_LINEAR);
    }

    std::vector<int> featuresPerLevel(n_levels_);
    float factor = 1.0f / scale_factor_;
    float nDesiredFeaturesPerScale = n_features_ * (1.0f - factor) / (1.0f - std::pow(factor, n_levels_));
    int totalAssigned = 0;

    for (int lvl = 0; lvl < n_levels_ - 1; ++lvl) {
        featuresPerLevel[lvl] = cvRound(nDesiredFeaturesPerScale);
        nDesiredFeaturesPerScale *= factor;
        totalAssigned += featuresPerLevel[lvl];
    }
    featuresPerLevel[n_levels_ - 1] = std::max(n_features_ - totalAssigned, 0);


    std::vector<cv::KeyPoint> kps;
    for (int lvl = 0; lvl < n_levels_; ++lvl) {
        cv::Mat lvlImg = pyramid[lvl];
        cv::Mat lvlMask;
        if (!mask.empty()) {
            cv::resize(mask, lvlMask, lvlImg.size(), 0, 0, cv::INTER_NEAREST);
        }

        //OpenCV
        /* 
        std::vector<cv::KeyPoint> lvlkps; // keypoints
        cv::Mat desc; // descriptors
        orb_->detectAndCompute(gray, lvlmask, lvlkps, desc);
        */
        //

        std::vector<cv::KeyPoint> lvlKps;
        collectFASTCells_(lvlImg, lvlMask,
                          kCellW, 20, 7,
                          0, lvlImg.cols,
                          0, lvlImg.rows,
                          featuresPerLevel[lvl],   
                          lvlKps);

        float scale = std::pow(scale_factor_, lvl);
        for (auto& kp : lvlKps) {
            kp.pt.x *= scale;
            kp.pt.y *= scale;
            kp.size = kPatchSize * scale;  
            kp.octave = lvl;               
        }

        kps.insert(kps.end(), lvlKps.begin(), lvlKps.end());
    }

    cv::Mat desc;
    if (!kps.empty()) {
        orb_->compute(gray, kps, desc);
    }

    const int N = static_cast<int>(kps.size());
    if (N == 0) { 
        result_.points.create(0, 2, CV_32F);
        result_.descriptors.create(0, 32, CV_8U);
        return;
    }
    
    std::vector<cv::Point2f> pts;
    pts.reserve(N);
    cv::KeyPoint::convert(kps, pts);
    cv::Mat ptsMat(pts, true);
    result_.points = ptsMat.reshape(1, N);

    result_.descriptors = std::move(desc);
}



void FeatureExtractor::collectFASTCells_(const cv::Mat& lvlImg,
                                         const cv::Mat& lvlMask,
                                         float cellW,
                                         int iniThFAST,
                                         int minThFAST,
                                         int minX, int maxX,
                                         int minY, int maxY,
                                         int maxKpTotal,   
                                         std::vector<cv::KeyPoint>& out)
{
    out.clear();
    const int nCols = std::ceil((maxX - minX) / cellW);
    const int nRows = std::ceil((maxY - minY) / cellW);
    const int nCells = nCols * nRows;

    int quotaPerCell = std::max(1, maxKpTotal / nCells);

    for (int r = 0; r < nRows; ++r) {
        for (int c = 0; c < nCols; ++c) {
            int x0 = std::round(c * cellW);
            int y0 = std::round(r * cellW);
            int x1 = std::min(x0 + cellW, (float)maxX);
            int y1 = std::min(y0 + cellW, (float)maxY);

            cv::Rect cellRect(x0, y0, x1 - x0, y1 - y0);
            cv::Mat cellROI = lvlImg(cellRect);
            cv::Mat maskROI;
            if (!lvlMask.empty()) maskROI = lvlMask(cellRect);

            std::vector<cv::KeyPoint> kps;
            cv::FAST(cellROI, kps, iniThFAST, true);
            if (kps.empty())
                cv::FAST(cellROI, kps, minThFAST, true);

       
            for (auto& kp : kps) {
                kp.pt.x += x0;
                kp.pt.y += y0;
                kp.size = kPatchSize;
            }

            if (!kps.empty()) {
                std::sort(kps.begin(), kps.end(),
                          [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                              return a.response > b.response;
                          });
                if ((int)kps.size() > quotaPerCell)
                    kps.resize(quotaPerCell);

                out.insert(out.end(), kps.begin(), kps.end());
            }
        }
    }
    if ((int)out.size() > maxKpTotal) {
        std::sort(out.begin(), out.end(),
                  [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                      return a.response > b.response;
                  });
        out.resize(maxKpTotal);
    }
}



} 


