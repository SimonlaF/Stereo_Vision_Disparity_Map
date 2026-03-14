#include "stereo_reconstruction.hpp"

StereoReconstructor::StereoReconstructor(const std::string& stereoParamsFile) {
    cv::FileStorage fs(stereoParamsFile, cv::FileStorage::READ);
    fs["K1"] >> _K1; fs["D1"] >> _D1;
    fs["K2"] >> _K2; fs["D2"] >> _D2;
    fs["R"] >> _R;   fs["T"] >> _T;
}

void StereoReconstructor::computeRectification(cv::Size imgSize) {
    // Étape 7 : Calcul des matrices de rectification
    cv::stereoRectify(_K1, _D1, _K2, _D2, imgSize, _R, _T, 
                      _R1, _R2, _P1, _P2, _Q, cv::CALIB_ZERO_DISPARITY, 0);

    // Pré-calcul des maps pour une rectification ultra-rapide (remap)
    cv::initUndistortRectifyMap(_K1, _D1, _R1, _P1, imgSize, CV_16SC2, _mapL1, _mapL2);
    cv::initUndistortRectifyMap(_K2, _D2, _R2, _P2, imgSize, CV_16SC2, _mapR1, _mapR2);
}

cv::Mat StereoReconstructor::computeDisparity(const cv::Mat& leftImg, const cv::Mat& rightImg) {
    cv::Mat rectL, rectR, disp;
    // Appliquer la rectification
    cv::remap(leftImg, rectL, _mapL1, _mapL2, cv::INTER_LINEAR);
    cv::remap(rightImg, rectR, _mapR1, _mapR2, cv::INTER_LINEAR);

    // Étape 10 : Algorithme de correspondance dense (StereoSGBM)
    auto stereo = cv::StereoSGBM::create(0, 64, 11); 
    stereo->compute(rectL, rectR, disp);

    cv::Mat dispVis;
    disp.convertTo(dispVis, CV_8U, 255.0 / (64 * 16.0));
    cv::imshow("Disparity Map", dispVis);
    
    return disp; // Retourne la disparité brute pour les calculs 3D
}

cv::Point3f StereoReconstructor::projectTo3D(int u, int v, int d) {
    // Étape 9 : Utilisation de la matrice Q (qui contient f, B, cx, cy)
    if (d <= 0) return cv::Point3f(0.0f, 0.0f, 0.0f);

    cv::Mat vec = (cv::Mat_<double>(4, 1) << u, v, d, 1.0);
    cv::Mat pos = _Q * vec;
    pos /= pos.at<double>(3, 0); // Normalisation homogène

    return cv::Point3f(static_cast<float>(pos.at<double>(0, 0)), 
                       static_cast<float>(pos.at<double>(1, 0)), 
                       static_cast<float>(pos.at<double>(2, 0)));
}