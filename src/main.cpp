#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include "../include/calibration.hpp"
#include "../include/stereo_calibration.hpp"
#include "../include/feature_matcher.hpp"
#include "../include/stereo_reconstruction.hpp"

namespace fs = std::filesystem;

// Variables globales Middlebury
static float fx = 0.0f, fy = 0.0f, cx0 = 0.0f, cx1 = 0.0f, cy = 0.0f, baseline = 0.0f, doffs = 0.0f;

void loadCalibrationFromTxt(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return;
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("cam0=") != std::string::npos) sscanf(line.c_str(), "cam0=[%f 0 %f; 0 %f %f; 0 0 1]", &fx, &cx0, &fy, &cy);
        if (line.find("cam1=") != std::string::npos) sscanf(line.c_str(), "cam1=[%*f 0 %f; 0 %*f %*f; 0 0 1]", &cx1);
        if (line.find("baseline=") != std::string::npos) sscanf(line.c_str(), "baseline=%f", &baseline);
        if (line.find("doffs=") != std::string::npos) sscanf(line.c_str(), "doffs=%f", &doffs);
    }
}

int main() {
    cv::Size pattern(7, 11);
    float squareSize = 25.0f;

    // ========================================================================
    // ACTE I : CALIBRATION STÉRÉO (Dataset Damier)
    // ========================================================================
    std::cout << "--- ACTE I : CALIBRATION SUR DATASET DAMIER ---" << std::endl;
    std::cout << "[STEP 1] Calibration Intrinsèque..." << std::endl;
    CameraCalibrator mono(pattern, squareSize);
    // Charger toutes les images du dossier left
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

    for (const auto& entry : fs::directory_iterator("../leftcamera")) leftImages.push_back(entry.path().string());
    for (const auto& entry : fs::directory_iterator("../rightcamera")) rightImages.push_back(entry.path().string());
    std::sort(leftImages.begin(), leftImages.end());
    std::sort(rightImages.begin(), rightImages.end());

    StereoCalibrator stereo(pattern, squareSize);
    
    if (stereo.loadIntrinsics("intrinsics_L.yml", "intrinsics_R.yml")) {
        std::cout << "[1/2] Calcul des parametres extrinseques..." << std::endl;
        size_t subsetSize = std::min((size_t)15, leftImages.size());
        std::vector<std::string> lSub(leftImages.begin(), leftImages.begin() + subsetSize);
        std::vector<std::string> rSub(rightImages.begin(), rightImages.begin() + subsetSize);

        if (stereo.runStereoCalibrationFromFileSets(lSub, rSub, "stereo_params.yml")) {
            std::cout << "[2/2] Verification visuelle de l'alignement..." << std::endl;
            stereo.runStereoCalibrationFromTwoFiles(leftImages[0], rightImages[0], "stereo_params.yml");
        }
    }
    cv::destroyAllWindows();

    // ========================================================================
    // ACTE II : MATCHING & DISPARITÉ (Dataset Middlebury)
    // ========================================================================
    std::cout << "\n--- ACTE II : RECONSTRUCTION SUR IMAGES REELLES ---" << std::endl;
    
    loadCalibrationFromTxt("calib.txt");
    cv::Mat frameL = cv::imread("../im0.png");
    cv::Mat frameR = cv::imread("../im1.png");

    if (frameL.empty() || frameR.empty()) {
        std::cerr << "Erreur : im0.png ou im1.png introuvable." << std::endl;
        return -1;
    }

    // 1. Matching ORB
    std::cout << "[1/3] Matching de points d'interet..." << std::endl;
    FeatureMatcher matcher;
    std::vector<cv::KeyPoint> kpL, kpR;
    std::vector<cv::DMatch> goodMatches;
    matcher.findMatches(frameL, frameR, kpL, kpR, goodMatches);
    matcher.drawMatches(frameL, frameR, kpL, kpR, goodMatches);
    cv::waitKey(0);

    // 2. Disparité (StereoBM avec les paramètres originaux)
    std::cout << "[2/3] Calcul de la carte de disparite (StereoBM)..." << std::endl;
    cv::Mat grayL, grayR;
    cv::cvtColor(frameL, grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frameR, grayR, cv::COLOR_BGR2GRAY);
    
    cv::Ptr<cv::StereoBM> stereoBM = cv::StereoBM::create(96, 15);
    cv::Mat disparity;
    stereoBM->compute(grayL, grayR, disparity);
    
    cv::Mat disp8;
    cv::normalize(disparity, disp8, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Disparity", disp8);

    // 3. Projection 3D (Distance au centre)
    std::cout << "[3/3] Calcul de la profondeur au centre..." << std::endl;
    int centerX = frameL.cols / 2;
    int centerY = frameL.rows / 2;

    short dValue = disparity.at<short>(centerY, centerX);

    if (dValue > 0)
    {
        float d = dValue / 16.0f;
        // FORMULE AVEC DOFFS
        float Z = (fx * baseline) / (d + doffs);

        std::string text = "Distance: " + std::to_string(Z/10.0f) + " cm";

        cv::putText(frameL, text, cv::Point(50,80), 
                    cv::FONT_HERSHEY_SIMPLEX, 1, 
                    cv::Scalar(0,0,255), 2);

        cv::drawMarker(frameL, cv::Point(centerX, centerY), 
                       cv::Scalar(0,255,0), 
                       cv::MARKER_CROSS, 20, 2);
    }

    cv::waitKey(0);

    return 0;
}