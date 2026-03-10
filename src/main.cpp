#include <iostream>
#include "../include/calibration.hpp"
#include "../include/stereo_calibration.hpp"
#include "../include/feature_matcher.hpp"
#include "../include/epipolar.hpp"          // Tes étapes 4, 5, 6
#include "../include/stereo_reconstruction.hpp" // Tes étapes 7, 8, 9, 10
#include <filesystem>
#include <fstream>
namespace fs = std::filesystem;
float fx, fy, cx0, cx1, cy, baseline, doffs;
void loadCalibrationFromTxt(const std::string& filename)
{
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {

        if (line.find("cam0=") != std::string::npos) {
            sscanf(line.c_str(),
                   "cam0=[%f 0 %f; 0 %f %f; 0 0 1]",
                   &fx, &cx0, &fy, &cy);
        }

        if (line.find("cam1=") != std::string::npos) {
            sscanf(line.c_str(),
                   "cam1=[%*f 0 %f; 0 %*f %*f; 0 0 1]",
                   &cx1);
        }

        if (line.find("baseline=") != std::string::npos)
            sscanf(line.c_str(), "baseline=%f", &baseline);

        if (line.find("doffs=") != std::string::npos)
            sscanf(line.c_str(), "doffs=%f", &doffs);
    }

    std::cout << "Calibration chargée depuis calib.txt" << std::endl;
}



int main() {
// --- CONFIGURATION DE BASE ---
cv::Size pattern(7, 11);   // Nombre de coins internes du damier
float squareSize = 25.0f; // Taille d'un carré en mm

// --- FLAGS DE CONTROLE (Active/Désactive ici) ---
loadCalibrationFromTxt("calib.txt");
bool runMonoCalib   = true; // Étape 1
bool runStereoCalib = true; // Étape 2
bool showMatches    = true;  // Étape 3
bool showEpipolar   = true;  // Étape 4, 5, 6
bool show3D         = true;  // Étape 7, 8, 9, 10

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

    // Dessiner les coins détectés
    cv::drawChessboardCorners(leftRect, pattern, ptsL, true);
    cv::drawChessboardCorners(rightRect, pattern, ptsR, true);

    // Fusion horizontale
    cv::Mat canvas;
    cv::hconcat(leftRect, rightRect, canvas);

    // Lignes horizontales de vérification
    for (int y = 0; y < canvas.rows; y += 40)
    {
        cv::line(canvas,
                cv::Point(0, y),
                cv::Point(canvas.cols, y),
                cv::Scalar(0, 255, 0), 1);
    }

    cv::imshow("Rectification Verification", canvas);
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

// =========================================================
// ÉTAPES 3 à 6 SUR DATASET
// =========================================================

    std::vector<std::string> leftImages, rightImages;

    for (const auto& entry : fs::directory_iterator("../leftcamera"))
        leftImages.push_back(entry.path().string());

    for (const auto& entry : fs::directory_iterator("../rightcamera"))
        rightImages.push_back(entry.path().string());

    std::sort(leftImages.begin(), leftImages.end());
    std::sort(rightImages.begin(), rightImages.end());

    size_t N = std::min(leftImages.size(), rightImages.size());

    for (size_t i = 0; i < N; i++)
    {
        std::cout << "Traitement paire : "
                << leftImages[i] << " / "
                << rightImages[i] << std::endl;

        // cv::Mat frameL = cv::imread(leftImages[i]);
        // cv::Mat frameR = cv::imread(rightImages[i]);
        
        cv::Mat frameL = cv::imread("../im0.png");
        cv::Mat frameR = cv::imread("../im1.png");
        
        if (frameL.empty() || frameR.empty())
            continue;

        // ==========================
        // ÉTAPE 3 : MATCHING
        // ==========================
        std::vector<cv::KeyPoint> kpL, kpR;
        std::vector<cv::DMatch> goodMatches;

        matcher.findMatches(frameL, frameR, kpL, kpR, goodMatches);

        // --- NOUVEAU : Nombre et erreur moyenne ---
        std::cout << "Nombre de correspondances trouvées : " << goodMatches.size() << std::endl;
        
        if (!goodMatches.empty()) {
            float totalError = 0.0f;
            for (const auto& match : goodMatches) {
                totalError += match.distance;
            }
            std::cout << "Erreur moyenne de matching : " << (totalError / goodMatches.size()) << std::endl;
        } else {
            std::cout << "Aucun match trouvé." << std::endl;
        }
        // ------------------------------------------

        if (showMatches)
            matcher.drawMatches(frameL, frameR, kpL, kpR, goodMatches);

        // ==========================
        // ÉTAPES 4,5,6 : EPIPOLAIRE
        // ==========================
        if (showEpipolar && goodMatches.size() >= 8)
        {
            std::vector<cv::Point2f> ptsL, ptsR;

            for (auto& m : goodMatches)
            {
                ptsL.push_back(kpL[m.queryIdx].pt);
                ptsR.push_back(kpR[m.trainIdx].pt);
            }

            // --- NOUVEAU : Création du masque pour filtrer les inliers ---
            std::vector<uchar> inliersMask; 
            cv::Mat F = epipolar.computeFundamental(ptsL, ptsR, inliersMask);

            cv::Mat frameEpipolar = frameR.clone();
            
            // On passe le masque à la fonction de dessin
            epipolar.drawEpipolarLines(frameL, frameEpipolar, F, ptsL, ptsR, inliersMask);

            cv::imshow("Etape 6: Lignes Epipolaires", frameEpipolar);
            cv::waitKey(0);
        }
// ==========================================================
// ÉTAPES 7 à 10 : RECTIFICATION + DISPARITÉ + 3D (DATASET)
// ==========================================================
cv::Mat testImg = cv::imread(leftImages[0]);
reconstructor.computeRectification(testImg.size());
if (show3D)
{
    cv::Mat grayL, grayR;
    cv::cvtColor(frameL, grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frameR, grayR, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(16*17, 15);

    cv::Mat disparity;
    stereo->compute(grayL, grayR, disparity);

    cv::Mat disp8;
    disparity.convertTo(disp8, CV_8U, 255/(16.0*17.0));
    cv::imshow("Disparity", disp8);

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

    cv::imshow("Image Gauche + Distance", frameL);
    cv::waitKey(0);
}
    }
return 0;
}