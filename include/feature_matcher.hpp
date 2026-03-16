#ifndef FEATURE_MATCHER_HPP
#define FEATURE_MATCHER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class FeatureMatcher {
public:
    FeatureMatcher();
    
    // Detect and match features between two images
    void findMatches(const cv::Mat& img1, const cv::Mat& img2, 
                     std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
                     std::vector<cv::DMatch>& goodMatches);

    // Draw matches for visualization
    void drawMatches(const cv::Mat& img1, const cv::Mat& img2,
                     const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
                     const std::vector<cv::DMatch>& matches);

private:
    cv::Ptr<cv::ORB> _orb;
    cv::Ptr<cv::DescriptorMatcher> _matcher;
};

#endif // FEATURE_MATCHER_HPP