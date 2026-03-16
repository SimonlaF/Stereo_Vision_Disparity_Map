#ifndef STEREO_CALIBRATION_HPP
#define STEREO_CALIBRATION_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class StereoCalibrator {
public:
    StereoCalibrator(cv::Size patternSize, float squareSize);

    // Charge les matrices K et distorsion calculées à l'étape 1
    bool loadIntrinsics(const std::string& leftFile, const std::string& rightFile);

    // Lance la capture synchro et calcule R et T
    bool runStereoCalibration(int camIdx1, int camIdx2, const std::string& saveFile);
    bool displayRectifiedView(const std::string& leftImgPath, const std::string& rightImgPath, const std::string& stereoParamsPath);
    bool runStereoCalibrationFromFileSets(const std::vector<std::string>& leftImages, const std::vector<std::string>& rightImages, const std::string& saveFile);
    void rectifyImages(const cv::Mat& left, const cv::Mat& right, cv::Mat& rectLeft, cv::Mat& rectRight, const std::string& stereoParamsPath);
private:
    cv::Size _patternSize;
    float _squareSize;
    
    // Matrices intrinsèques
    cv::Mat _K1, _dist1, _K2, _dist2;
    
    void generateObjectPoints(std::vector<cv::Point3f>& objp) const;
};

#endif // STEREO_CALIBRATION_HPP