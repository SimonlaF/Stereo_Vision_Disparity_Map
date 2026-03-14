#include "epipolar.hpp"

cv::Mat Epipolar::computeFundamental(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, std::vector<uchar>& inliers) {
    return cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.99, inliers);
}

void Epipolar::solveEssential(const cv::Mat& F, const cv::Mat& K1, const cv::Mat& K2, 
                              cv::Mat& R, cv::Mat& t, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2) {
    // Etape 5 : E = K2^T * F * K1
    cv::Mat E = K2.t() * F * K1;

    // Décomposition par SVD pour retrouver R et t
    cv::recoverPose(E, pts1, pts2, K1, R, t); 
}

void Epipolar::drawEpipolarLines(cv::Mat& img1, cv::Mat& img2, const cv::Mat& F, 
                                 const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2,
                                 const std::vector<uchar>& inliers) {
    std::vector<cv::Point3f> lines2;
    cv::computeCorrespondEpilines(pts1, 1, F, lines2);

    for (size_t i = 0; i < lines2.size(); ++i) {
        if (inliers.empty() || inliers[i]) {
            cv::line(img2, cv::Point(0, -lines2[i].z / lines2[i].y),
                     cv::Point(img2.cols, -(lines2[i].z + lines2[i].x * img2.cols) / lines2[i].y),
                     cv::Scalar(0, 255, 0), 1);
            cv::circle(img1, pts1[i], 5, cv::Scalar(0, 0, 255), -1);
        }
    }
}