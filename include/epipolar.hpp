#ifndef EPIPOLAR_HPP
#define EPIPOLAR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class Epipolar {
public:
    // Etape 4 : Matrice Fondamentale
    cv::Mat computeFundamental(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, std::vector<uchar>& inliers);

    // Etape 5 : Matrice Essentielle et Décomposition
    void solveEssential(const cv::Mat& F, const cv::Mat& K1, const cv::Mat& K2, 
                        cv::Mat& R, cv::Mat& t, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2);

    // Etape 6 : Lignes épipolaires
    void drawEpipolarLines(cv::Mat& img1, cv::Mat& img2, const cv::Mat& F, 
                           const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, const std::vector<uchar>& inliers);
};

#endif // EPIPOLAR_HPP