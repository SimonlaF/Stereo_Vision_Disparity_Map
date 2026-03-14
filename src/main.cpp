#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include "../include/calibration.hpp"
#include "../include/stereo_calibration.hpp"
#include "../include/feature_matcher.hpp"
#include "../include/epipolar.hpp"
#include "../include/stereo_reconstruction.hpp"

namespace fs = std::filesystem;

// Variables globales pour la calibration Middlebury (Dataset 2)
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
    // --- CONFIGURATION DAMIER ---
    cv::Size pattern(7, 11);
    float squareSize = 25.0f;

    // ========================================================================
    // ACTE I : CALIBRATION STÉRÉO (Dataset Checkerboard)
    // Objectif : Calculer R et T à partir de paires d'images du damier
    // ========================================================================
    std::cout << "--- ACTE I : CALIBRATION SUR DATASET DAMIER ---" << std::endl;

    std::vector<std::string> leftImages, rightImages;
    for (const auto& entry : fs::directory_iterator("../leftcamera")) leftImages.push_back(entry.path().string());
    for (const auto& entry : fs::directory_iterator("../rightcamera")) rightImages.push_back(entry.path().string());
    std::sort(leftImages.begin(), leftImages.end());
    std::sort(rightImages.begin(), rightImages.end());

    StereoCalibrator stereo(pattern, squareSize);
    
    // On suppose que les intrinsèques (K1, D1, K2, D2) ont été calculés au préalable (Etape 1)
    if (stereo.loadIntrinsics("intrinsics_L.yml", "intrinsics_R.yml")) {
        
        std::cout << "[1/2] Calcul des paramètres extrinsèques (R, T)..." << std::endl;
        // On utilise un subset pour la démo
        size_t subsetSize = std::min((size_t)15, leftImages.size());
        std::vector<std::string> lSub(leftImages.begin(), leftImages.begin() + subsetSize);
        std::vector<std::string> rSub(rightImages.begin(), rightImages.begin() + subsetSize);

        if (stereo.runStereoCalibrationFromFileSets(lSub, rSub, "stereo_params.yml")) {
            std::cout << "[2/2] Vérification visuelle par Rectification..." << std::endl;
            // On affiche le résultat sur la première image du dataset pour valider
            stereo.runStereoCalibrationFromTwoFiles(leftImages[0], rightImages[0], "stereo_params.yml");
        }
    } else {
        std::cerr << "Attention : intrinsics_L.yml/R.yml manquants. Saute l'Acte I." << std::endl;
    }

    cv::destroyAllWindows();

    // ========================================================================
    // ACTE II : MATCHING & DISPARITÉ (Dataset Middlebury)
    // Objectif : Utiliser une calibration existante pour extraire la profondeur
    // ========================================================================
    std::cout << "\n--- ACTE II : RECONSTRUCTION SUR IMAGES RÉELLES ---" << std::endl;
    
    loadCalibrationFromTxt("calib.txt");
    cv::Mat frameL = cv::imread("../im0.png");
    cv::Mat frameR = cv::imread("../im1.png");

    if (frameL.empty() || frameR.empty()) {
        std::cerr << "Erreur : Dataset Middlebury (im0.png / im1.png) introuvable." << std::endl;
        return -1;
    }

    // 1. Matching de points d'intérêt (ORB)
    std::cout << "[1/3] Recherche de correspondances (ORB)..." << std::endl;
    FeatureMatcher matcher;
    std::vector<cv::KeyPoint> kpL, kpR;
    std::vector<cv::DMatch> goodMatches;
    matcher.findMatches(frameL, frameR, kpL, kpR, goodMatches);
    matcher.drawMatches(frameL, frameR, kpL, kpR, goodMatches);
    cv::waitKey(0);

    // 2. Calcul de la carte de disparité
    std::cout << "[2/3] Calcul de la carte de disparité (SGBM)..." << std::endl;
    // On utilise une instance temporaire ou dédiée pour ce dataset spécifique
    cv::Mat grayL, grayR, disparity;
    cv::cvtColor(frameL, grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frameR, grayR, cv::COLOR_BGR2GRAY);
    
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 160, 5, 8*3*5*5, 32*3*5*5, 1, 63, 10, 100, 32);
    sgbm->compute(grayL, grayR, disparity);

    cv::Mat dispVis;
    cv::normalize(disparity, dispVis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Carte de Disparite (Acte II)", dispVis);

    // 3. Projection 3D (Distance au centre)
    std::cout << "[3/3] Estimation de la distance au centre de l'image..." << std::endl;
    int cx = frameL.cols / 2;
    int cy = frameL.rows / 2;
    short dValue = disparity.at<short>(cy, cx);

    if (dValue > 0) {
        float d = dValue / 16.0f; // OpenCV stocke la disparité en 1/16ème de pixel
        float Z = (fx * baseline) / (d + doffs);
        
        std::string result = "Distance centre: " + std::to_string(Z / 10.0f) + " cm";
        cv::putText(frameL, result, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::circle(frameL, cv::Point(cx, cy), 10, cv::Scalar(0, 0, 255), 2);
    }

    std::cout << "Demo terminee. Appuyez sur une touche pour quitter." << std::endl;
    cv::waitKey(0);

    return 0;
}