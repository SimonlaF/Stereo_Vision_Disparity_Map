#include <iostream>
#include "../include/calibration.hpp"
#include "../include/stereo_calibration.hpp"
#include "../include/feature_matcher.hpp"
#include "../include/epipolar.hpp"          // Tes étapes 4, 5, 6
#include "../include/stereo_reconstruction.hpp" // Tes étapes 7, 8, 9, 10
#include <filesystem>
namespace fs = std::filesystem;



int main() {
// --- CONFIGURATION DE BASE ---
cv::Size pattern(7, 11);   // Nombre de coins internes du damier
float squareSize = 25.0f; // Taille d'un carré en mm

// --- FLAGS DE CONTROLE (Active/Désactive ici) ---
bool runMonoCalib   = true; // Étape 1
bool runStereoCalib = true; // Étape 2
bool showMatches    = false;  // Étape 3
bool showEpipolar   = false;  // Étape 4, 5, 6
bool show3D         = false;  // Étape 7, 8, 9, 10

// =========================================================
// ÉTAPE 1 : CALIBRATION INTRINSÈQUE (Individuelle)
// =========================================================
if (runMonoCalib) {
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

    // Tri optionnel pour correspondance
    std::sort(leftImages.begin(), leftImages.end());
    std::sort(rightImages.begin(), rightImages.end());

    mono.runCalibrationFromFiles(leftImages, "intrinsics_L.yml");
    mono.runCalibrationFromFiles(rightImages, "intrinsics_R.yml");
}

// =========================================================
// ÉTAPE 2 : CALIBRATION EXTRINSÈQUE & IMAGE FINALE
// =========================================================
if (runStereoCalib) {
    std::cout << "[STEP 2] Calibration Extrinsèque et lignes épipolaires..." << std::endl;

    StereoCalibrator stereo(pattern, squareSize);

    std::vector<std::string> leftImages, rightImages;
    std::string leftDir  = "../leftcamera";
    std::string rightDir = "../rightcamera";

    for (const auto& entry : fs::directory_iterator(leftDir))
        leftImages.push_back(entry.path().string());
    for (const auto& entry : fs::directory_iterator(rightDir))
        rightImages.push_back(entry.path().string());

    std::sort(leftImages.begin(), leftImages.end());
    std::sort(rightImages.begin(), rightImages.end());

    if (leftImages.size() != rightImages.size() || leftImages.empty()) {
        std::cerr << "Erreur : nombres d'images gauche et droite différents ou vide !" << std::endl;
        return -1;
    }

    if (!stereo.loadIntrinsics("intrinsics_L.yml", "intrinsics_R.yml")) {
        std::cerr << "ERREUR: Fichiers intrinsèques introuvables." << std::endl;
        return -1;
    }

    // Sélection des 20 premières paires
    size_t N = std::min(size_t(20), leftImages.size());
    std::vector<std::string> leftSubset(leftImages.begin(), leftImages.begin() + N);
    std::vector<std::string> rightSubset(rightImages.begin(), rightImages.begin() + N);

    for (size_t i = 0; i < N; i++) {
        std::cout << "Utilisation de la paire: " 
                  << leftSubset[i] << " / " << rightSubset[i] << std::endl;
    }

    // --- Calibration stéréo (avec la version existante) ---
    if (!stereo.runStereoCalibrationFromFileSets(leftSubset, rightSubset, "stereo_params.yml")) {
        std::cerr << "Erreur lors de la calibration stéréo." << std::endl;
        return -1;
    }

    // --- RECTIFICATION ---
    cv::FileStorage fsStereo("stereo_params.yml", cv::FileStorage::READ);
    cv::Mat K1, D1, K2, D2, R, T;
    fsStereo["K1"] >> K1; fsStereo["D1"] >> D1;
    fsStereo["K2"] >> K2; fsStereo["D2"] >> D2;
    fsStereo["R"]  >> R;  fsStereo["T"]  >> T;
    fsStereo.release();

    cv::Mat R1, R2, P1, P2, Q;
    cv::Size imageSize = cv::imread(leftSubset[0]).size();

    cv::stereoRectify(K1, D1, K2, D2, imageSize, R, T, R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, -1, imageSize);

    cv::Mat map1x, map1y, map2x, map2y;
    cv::initUndistortRectifyMap(K1, D1, R1, P1, imageSize, CV_32FC1, map1x, map1y);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, imageSize, CV_32FC1, map2x, map2y);

    // --- Image finale rectifiée avec épipolaires ---
    cv::Mat leftImg  = cv::imread(leftSubset[0]);
    cv::Mat rightImg = cv::imread(rightSubset[0]);
    cv::Mat leftRect, rightRect;
    cv::remap(leftImg, leftRect, map1x, map1y, cv::INTER_LINEAR);
    cv::remap(rightImg, rightRect, map2x, map2y, cv::INTER_LINEAR);

    // Points détectés pour dessiner les lignes épipolaires
    std::vector<cv::Point2f> ptsL, ptsR;
    cv::Mat grayL, grayR;
    cv::cvtColor(leftRect, grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rightRect, grayR, cv::COLOR_BGR2GRAY);

    cv::findChessboardCorners(grayL, pattern, ptsL);
    cv::findChessboardCorners(grayR, pattern, ptsR);

    cv::Mat F = cv::findFundamentalMat(ptsL, ptsR, cv::FM_8POINT);

    Epipolar epipolar;
    epipolar.drawEpipolarLines(leftRect, rightRect, F, ptsL, ptsR);

    cv::imshow("Rectified Left", leftRect);
    cv::imshow("Rectified Right", rightRect);
    cv::waitKey(0);
}

// =========================================================
// INITIALISATION DES OBJETS POUR LE TEMPS RÉEL
// =========================================================
FeatureMatcher matcher;
Epipolar epipolar;

// On charge les paramètres calculés à l'étape 2
StereoReconstructor reconstructor("stereo_params.yml");
// Étape 7 : Pré-calcul de la rectification pour aligner les images
reconstructor.computeRectification(cv::Size(1280, 720));

cv::VideoCapture capL, capR;
if (showMatches || showEpipolar || show3D) {
    capL.open(0);
    capR.open(1);
    if (!capL.isOpened() || !capR.isOpened()) {
        std::cerr << "ERREUR: Caméras introuvables." << std::endl;
        return -1;
    }
}

std::cout << "\n--- DEMARRAGE DU FLUX TEMPS REEL ---" << std::endl;
cv::Mat frameL, frameR;

while (true) {
    capL >> frameL;
    capR >> frameR;
    if (frameL.empty() || frameR.empty()) break;

    // =========================================================
    // ÉTAPE 3 : CORRESPONDANCES (Matching)
    // =========================================================
    std::vector<cv::KeyPoint> kpL, kpR;
    std::vector<cv::DMatch> goodMatches;
    matcher.findMatches(frameL, frameR, kpL, kpR, goodMatches);

    if (showMatches) {
        matcher.drawMatches(frameL, frameR, kpL, kpR, goodMatches);
    }

    // =========================================================
    // ÉTAPES 4, 5, 6 : GEOMETRIE EPIPOLAIRE
    // =========================================================
    if (showEpipolar && goodMatches.size() >= 8) {
        std::vector<cv::Point2f> ptsL, ptsR;
        for (auto& m : goodMatches) {
            ptsL.push_back(kpL[m.queryIdx].pt);
            ptsR.push_back(kpR[m.trainIdx].pt);
        }

        // Étape 4 & 5 : Calcul des matrices F et E
        cv::Mat F = epipolar.computeFundamental(ptsL, ptsR);
        
        // Étape 6 : Dessin des lignes (Vérifie si le point est sur la ligne)
        // On le dessine sur une copie pour ne pas polluer l'image 3D
        cv::Mat frameEpipolar = frameR.clone();
        epipolar.drawEpipolarLines(frameL, frameEpipolar, F, ptsL, ptsR);
        cv::imshow("Etape 6: Lignes Epipolaires", frameEpipolar);
    }

    // =========================================================
    // ÉTAPES 7 à 10 : RECTIFICATION ET 3D
    // =========================================================
    if (show3D) {
        // Étape 7 & 10 : Rectification et calcul de la carte de disparité
        cv::Mat disparity = reconstructor.computeDisparity(frameL, frameR);

        // Étape 8 & 9 : Triangulation (Calcul de la distance Z)
        // On teste sur le pixel central de l'image
        int centerX = 640; int centerY = 360;
        short dValue = disparity.at<short>(centerY, centerX);
        
        if (dValue > 0) {
            float d = dValue / 16.0f; // Conversion format OpenCV
            cv::Point3f pos3D = reconstructor.projectTo3D(centerX, centerY, d);
            
            // Affichage de la distance sur l'image
            std::string text = "Distance: " + std::to_string(pos3D.z / 10.0f) + " cm";
            cv::putText(frameL, text, cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
            
            // Dessiner un curseur au centre
            cv::drawMarker(frameL, cv::Point(centerX, centerY), cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 20, 2);
        }
        cv::imshow("Etape 10: Vue Gauche + Distance", frameL);
    }

    if (cv::waitKey(1) == 27) break; // Echap pour quitter
}

return 0;
}