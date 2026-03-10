#ifndef STEREORECONSTRUCTOR_H
#define STEREORECONSTRUCTOR_H

#include <opencv2/opencv.hpp>

class StereoReconstructor {
public:
    StereoReconstructor(const std::string& stereoParamsFile);

    // Étape 7 : Rectification (prépare les maps de transformation)
    void computeRectification(cv::Size imgSize);

    // Étape 10 : Reconstruction Dense (Carte de disparité)
    cv::Mat computeDisparity(const cv::Mat& leftImg, const cv::Mat& rightImg);

    // Étape 8 & 9 : Triangulation (Point 2D -> Point 3D)
    cv::Point3f projectTo3D(int uL, int vL, int disparity);

    // Générer le nuage de points complet
    void getPointCloud(const cv::Mat& disparity, std::vector<cv::Point3f>& points);
    void rectify(const cv::Mat& left, const cv::Mat& right, cv::Mat& leftRect, cv::Mat& rightRect);

private:
    cv::Mat _K1, _D1, _K2, _D2, _R, _T; // Paramètres de calibration
    cv::Mat _R1, _P1, _R2, _P2, _Q;     // Paramètres de rectification
    cv::Mat _mapL1, _mapL2, _mapR1, _mapR2; // Maps pour remap()
};

#endif