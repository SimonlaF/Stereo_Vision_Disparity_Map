#include "feature_matcher.hpp"

FeatureMatcher::FeatureMatcher() {
    // ORB est gratuit et très performant pour le temps réel
    _orb = cv::ORB::create(500); // On cherche 500 points max
    _matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
}

void FeatureMatcher::findMatches(const cv::Mat& img1, const cv::Mat& img2, 
                                 std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
                                 std::vector<cv::DMatch>& goodMatches) {
    cv::Mat desc1, desc2;
    
    // 1. Détection et calcul des descripteurs
    _orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    _orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    if (desc1.empty() || desc2.empty()) return;

    // 2. Mise en correspondance (Matching)
    std::vector<cv::DMatch> matches;
    _matcher->match(desc1, desc2, matches);

    // 3. Filtrage des mauvais points (on ne garde que les plus proches)
    std::sort(matches.begin(), matches.end());
    const int numGoodMatches = matches.size() * 0.2f; // Garde les 20% meilleurs
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