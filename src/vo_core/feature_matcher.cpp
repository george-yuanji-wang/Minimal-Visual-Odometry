#include "vo_core/feature_matcher.hpp"

namespace vo_core {

FeatureMatcher::FeatureMatcher(const MatchParams& prm) : prm_(prm) {}

void FeatureMatcher::setParams(const MatchParams& prm) { prm_ = prm; }

const MatchParams& FeatureMatcher::params() const { return prm_; }

void FeatureMatcher::GridIndex::build(const std::vector<cv::KeyPoint>& kps, cv::Size imgSz, int cw, int ch) {
    cellW = cw; cellH = ch;
    cols = std::max(1, imgSz.width  / cellW);
    rows = std::max(1, imgSz.height / cellH);
    buckets.assign(cols*rows, {});
    for (int i = 0; i < (int)kps.size(); ++i) {
        int cx = std::min(cols-1, std::max(0, int(kps[i].pt.x) / cellW));
        int cy = std::min(rows-1, std::max(0, int(kps[i].pt.y) / cellH));
        buckets[cy*cols + cx].push_back(i);
    }
}

void FeatureMatcher::GridIndex::queryWindow(const cv::Point2f& p, float radius, std::vector<int>& outCellIds) const {
    int x0 = std::max(0, int((p.x - radius) / cellW));
    int x1 = std::min(cols-1, int((p.x + radius) / cellW));
    int y0 = std::max(0, int((p.y - radius) / cellH));
    int y1 = std::min(rows-1, int((p.y + radius) / cellH));
    outCellIds.clear();
    for (int y = y0; y <= y1; ++y) for (int x = x0; x <= x1; ++x) outCellIds.push_back(y*cols + x);
}

#if defined(__GNUC__) || defined(__clang__)
static inline int popcnt64(uint64_t x){ return __builtin_popcountll(x); }
#else
#include <intrin.h>
static inline int popcnt64(uint64_t x){ return (int)__popcnt64(x); }
#endif

int FeatureMatcher::hamming256(const uchar* a, const uchar* b) {
    const uint64_t* A = reinterpret_cast<const uint64_t*>(a);
    const uint64_t* B = reinterpret_cast<const uint64_t*>(b);
    return popcnt64(A[0]^B[0]) + popcnt64(A[1]^B[1]) + popcnt64(A[2]^B[2]) + popcnt64(A[3]^B[3]);
}

void FeatureMatcher::matchLocal(const std::vector<cv::KeyPoint>& kpsQ, const cv::Mat& descQ,
                                const std::vector<cv::KeyPoint>& kpsT, const cv::Mat& descT,
                                const GridIndex& gridT, std::vector<int>& q2t) const
{
    q2t.assign(kpsQ.size(), -1);
    std::vector<int> cells;
    for (int qi = 0; qi < (int)kpsQ.size(); ++qi) {
        const auto& qk = kpsQ[qi];
        const uchar* qd = descQ.ptr<uchar>(qi);
        float lvlScale = std::pow(prm_.scaleFactor, (float)qk.octave);
        float radius = prm_.baseRadius * lvlScale;
        gridT.queryWindow(qk.pt, radius, cells);
        int bestJ = -1, best = 1e9, second = 1e9;
        for (int cid : cells) {
            const auto& bucket = gridT.buckets[cid];
            for (int j : bucket) {
                const auto& tk = kpsT[j];
                if (std::abs(tk.pt.x - qk.pt.x) > radius) continue;
                if (std::abs(tk.pt.y - qk.pt.y) > radius) continue;
                if (std::abs(tk.octave - qk.octave) > prm_.maxLevelDiff) continue;
                int d = hamming256(qd, descT.ptr<uchar>(j));
                if (d < best) { second = best; best = d; bestJ = j; }
                else if (d < second) { second = d; }
            }
        }
        if (bestJ >= 0 && best < prm_.thLow && best < prm_.ratio * second) q2t[qi] = bestJ;
    }
}

void FeatureMatcher::enforceOrientation(const std::vector<cv::KeyPoint>& kpsQ,
                                        const std::vector<cv::KeyPoint>& kpsT,
                                        std::vector<int>& q2t) const
{
    if (prm_.orientationBins <= 0 || prm_.keepBins <= 0) return;
    std::vector<int> hist(prm_.orientationBins, 0), bins(q2t.size(), -1);
    auto binOf = [&](float d){
        float a = d;
        while (a < 0) a += 360.f;
        while (a >= 360.f) a -= 360.f;
        int b = int(std::round(a * prm_.orientationBins / 360.f)) % prm_.orientationBins;
        return b;
    };
    for (int i = 0; i < (int)q2t.size(); ++i) if (q2t[i] >= 0) {
        float d = kpsQ[i].angle - kpsT[q2t[i]].angle;
        int b = binOf(d);
        bins[i] = b; hist[b]++;
    }
    std::vector<int> order(prm_.orientationBins);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){ return hist[a] > hist[b]; });
    std::vector<char> keep(prm_.orientationBins, 0);
    for (int i = 0; i < std::min(prm_.keepBins, prm_.orientationBins); ++i) keep[order[i]] = 1;
    for (int i = 0; i < (int)q2t.size(); ++i) if (q2t[i] >= 0 && !keep[bins[i]]) q2t[i] = -1;
}

void FeatureMatcher::symmetricFilter(const std::vector<int>& q2t,
                                     const std::vector<int>& t2q,
                                     std::vector<int>& q2t_sym) const
{
    q2t_sym = q2t;
    for (int qi = 0; qi < (int)q2t.size(); ++qi) {
        int tj = q2t[qi];
        if (tj < 0) continue;
        if (tj >= (int)t2q.size() || t2q[tj] != qi) q2t_sym[qi] = -1;
    }
}

void FeatureMatcher::match(const std::vector<cv::KeyPoint>& kpsQ, const cv::Mat& descQ, cv::Size sizeQ,
                           const std::vector<cv::KeyPoint>& kpsT, const cv::Mat& descT, cv::Size sizeT,
                           std::vector<cv::DMatch>& matches) const
{
    GridIndex gridT; gridT.build(kpsT, sizeT, prm_.cellW, prm_.cellH);
    std::vector<int> q2t;
    matchLocal(kpsQ, descQ, kpsT, descT, gridT, q2t);
    enforceOrientation(kpsQ, kpsT, q2t);
    if (prm_.symmetric) {
        GridIndex gridQ; gridQ.build(kpsQ, sizeQ, prm_.cellW, prm_.cellH);
        std::vector<int> t2q;
        matchLocal(kpsT, descT, kpsQ, descQ, gridQ, t2q);
        std::vector<int> q2t_sym;
        symmetricFilter(q2t, t2q, q2t_sym);
        q2t.swap(q2t_sym);
    }
    matches.clear();
    for (int qi = 0; qi < (int)q2t.size(); ++qi) if (q2t[qi] >= 0) matches.emplace_back(qi, q2t[qi], 0.f);
}



}
