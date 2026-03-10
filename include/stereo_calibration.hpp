#ifndef STEREOCALIBRATOR_H
#define STEREOCALIBRATOR_H

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
    bool runStereoCalibrationFromTwoFiles(const std::string& leftImgPath,const std::string& rightImgPath, const std::string& saveFile);
    bool runStereoCalibrationFromFileSets(const std::vector<std::string>& leftImages, const std::vector<std::string>& rightImages, const std::string& saveFile);
private:
    cv::Size _patternSize;
    float _squareSize;
    
    // Matrices intrinsèques
    cv::Mat _K1, _dist1, _K2, _dist2;
    
    void generateObjectPoints(std::vector<cv::Point3f>& objp);

};

#endif