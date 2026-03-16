#ifndef DISPARITY_MAP_HPP
#define DISPARITY_MAP_HPP

#include <opencv2/opencv.hpp>
#include <string>
// Structure simple pour stocker un point et sa profondeur
struct SparsePoint {
    cv::Point2f ptL;    // Position sur l'image gauche
    float disparity;    // Valeur de disparité (xL - xR)
};
class DisparityMap {
public:
    static cv::Mat computeBM(const cv::Mat& rectL, const cv::Mat& rectR, 
                             int numDisparities = 64, int blockSize = 15);
    static cv::Mat getVisualMap(const cv::Mat& disparity);
    static std::vector<SparsePoint> computeSparse(const std::vector<cv::KeyPoint>& kpL, 
                                                  const std::vector<cv::KeyPoint>& kpR,
                                                  const std::vector<cv::DMatch>& matches);
    static void drawSparse(cv::Mat& canvas, const std::vector<SparsePoint>& sparsePoints);
};

#endif