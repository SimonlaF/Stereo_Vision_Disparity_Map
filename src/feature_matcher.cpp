#include "feature_matcher.hpp"

FeatureMatcher::FeatureMatcher() {
    _orb = cv::ORB::create(5000); 
    _matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
}

void FeatureMatcher::findMatches(const cv::Mat& img1, const cv::Mat& img2, 
                                 std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
                                 std::vector<cv::DMatch>& goodMatches) {
    cv::Mat desc1, desc2;
    
    // Detection and calculation of descriptors
    _orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    _orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    if (desc1.empty() || desc2.empty()) return;

    std::vector<cv::DMatch> matches;
    _matcher->match(desc1, desc2, matches);

    // Filter matches based on distance (keep top 20%)
    std::sort(matches.begin(), matches.end());
    const size_t numGoodMatches = static_cast<size_t>(matches.size() * 0.2f);
    goodMatches.assign(matches.begin(), matches.begin() + numGoodMatches);
}

void FeatureMatcher::drawMatches(const cv::Mat& img1, const cv::Mat& img2,
                                 const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
                                 const std::vector<cv::DMatch>& matches) {
    cv::Mat imgMatches;
    cv::drawMatches(img1, kp1, img2, kp2, matches, imgMatches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    cv::imshow("Correspondances (Matches)", imgMatches);
}