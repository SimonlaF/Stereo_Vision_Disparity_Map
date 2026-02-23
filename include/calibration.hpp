#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class CameraCalibrator {
public:
    CameraCalibrator(cv::Size patternSize, float squareSize);
    
    // Execute la capture interactive et sauvegarde le .yml
    bool runCalibration(int camID, const std::string& saveFileName);
    bool runCalibrationFromFiles(const std::vector<std::string>& imagePaths, const std::string& saveFileName); 
private:
    cv::Size _patternSize;
    float _squareSize;
    cv::TermCriteria _criteria;
    
    void generateObjectPoints(std::vector<cv::Point3f>& objp);
};

#endif