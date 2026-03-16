#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include "../include/calibration.hpp"
#include "../include/stereo_calibration.hpp"
#include "../include/feature_matcher.hpp"
#include "../include/disparity_map.hpp"
namespace fs = std::filesystem;

static float fx = 0.0f, fy = 0.0f, cx0 = 0.0f, cx1 = 0.0f, cy = 0.0f, baseline = 0.0f, doffs = 0.0f;
// Function to load calibration parameters from a text file (calib.txt)
void loadCalibrationFromTxt(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return;
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("cam0=") != std::string::npos) sscanf_s(line.c_str(), "cam0=[%f 0 %f; 0 %f %f; 0 0 1]", &fx, &cx0, &fy, &cy);
        if (line.find("cam1=") != std::string::npos) sscanf_s(line.c_str(), "cam1=[%*f 0 %f; 0 %*f %*f; 0 0 1]", &cx1);
        if (line.find("baseline=") != std::string::npos) sscanf_s(line.c_str(), "baseline=%f", &baseline);
        if (line.find("doffs=") != std::string::npos) sscanf_s(line.c_str(), "doffs=%f", &doffs);
    }
}

int main() {
    cv::Size pattern(7, 11);   // Size of the checkerboard pattern (columns, rows)
    float squareSize = 25.0f;
    std::cout << "[STEP 1] Intrisic Calibration..." << std::endl;
    CameraCalibrator mono(pattern, squareSize);
    // Loading images from the left and right camera folders
    std::vector<std::string> leftImages, rightImages;

    for (const auto& entry : fs::directory_iterator("../leftcamera")) {
        leftImages.push_back(entry.path().string());
    }

    for (const auto& entry : fs::directory_iterator("../rightcamera")) {
        rightImages.push_back(entry.path().string());

    }

    std::sort(leftImages.begin(), leftImages.end());
    std::sort(rightImages.begin(), rightImages.end());

    mono.runCalibrationFromFiles(leftImages, "intrinsics_L.yml"); 
    mono.runCalibrationFromFiles(rightImages, "intrinsics_R.yml");
    
    
    std::cout << "[STEP 2] Extrinsic Calibration..." << std::endl;
    StereoCalibrator stereo(pattern, squareSize);
    

    if (stereo.loadIntrinsics("intrinsics_L.yml", "intrinsics_R.yml")) {
        std::cout << "[1/2] Computing extrinsic parameters..." << std::endl;    // Compute R and T from the set of images
        if (stereo.runStereoCalibrationFromFileSets(leftImages, rightImages, "stereo_params.yml")) {
            std::cout << "[2/2] Checking rectified images..." << std::endl;                 // Display rectified images to visually check the epipolar alignment
            stereo.displayRectifiedView(leftImages[0], rightImages[0], "../build/stereo_params.yml");
        }
    }
    cv::destroyAllWindows();

    // MATCHING & DISPARITY (Dataset Middlebury) (No need for calibration parameters here because MiddleBury images are already rectified)
    // 1. Matching ORB
    std::cout << "[1/3] Matching of interest points..." << std::endl;
    loadCalibrationFromTxt("calib.txt");
    cv::Mat frameL = cv::imread("../im0.png");
    cv::Mat frameR = cv::imread("../im1.png");

    if (frameL.empty() || frameR.empty()) {
        std::cerr << "Erreur : im0.png ou im1.png introuvable." << std::endl;
        return -1;
    }
    FeatureMatcher matcher;
    std::vector<cv::KeyPoint> kpL, kpR;
    std::vector<cv::DMatch> goodMatches;
    matcher.findMatches(frameL, frameR, kpL, kpR, goodMatches);
    matcher.drawMatches(frameL, frameR, kpL, kpR, goodMatches);
    cv::waitKey(0);

    // Disparity ( Stereo Block Matching with default parameters)
    std::cout << "[2/3] Computing disparity map..." << std::endl;
    cv::Mat dispRaw = DisparityMap::computeBM(frameL, frameR, 64, 21);
    cv::Mat dispColor = DisparityMap::getVisualColorMap(dispRaw);
    cv::imshow("Disparity", dispColor);

    // Sparse Disparity (only on matched keypoints with ORB)
    std::vector<SparsePoint> sparsePoints = DisparityMap::computeSparse(kpL, kpR, goodMatches);
    cv::Mat visSparse = frameL.clone();
    DisparityMap::drawSparse(visSparse, sparsePoints);

    cv::imshow("Sparse Disparity (ORB)", visSparse);
    cv::waitKey(0);

    return 0;
}