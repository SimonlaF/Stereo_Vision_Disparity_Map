#include <iostream>
#include <vector>
#include <string>
#include <fstream>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

// Tes fichiers
#include "../include/feature_matcher.hpp"
#include "../include/epipolar.hpp"
#include "../include/stereo_reconstruction.hpp"

float fx, fy, cx0, cx1, cy, baseline, doffs;

// Fonction utilitaire pour afficher une image sans qu'elle soit zoomée
void showResized(const std::string& name, const cv::Mat& img, int width = 1280) {
    // WINDOW_NORMAL permet de redimensionner la fenêtre manuellement
    cv::namedWindow(name, cv::WINDOW_NORMAL); 
    
    // On calcule la hauteur proportionnelle pour ne pas déformer l'image
    int height = (img.rows * width) / img.cols;
    
    cv::resizeWindow(name, width, height);
    cv::imshow(name, img);
}

void loadCalibrationFromTxt(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Impossible d'ouvrir " << filename << std::endl;
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("cam0=") != std::string::npos) 
            sscanf(line.c_str(), "cam0=[%f 0 %f; 0 %f %f; 0 0 1]", &fx, &cx0, &fy, &cy);
        if (line.find("cam1=") != std::string::npos) 
            sscanf(line.c_str(), "cam1=[%*f 0 %f; 0 %*f %*f; 0 0 1]", &cx1);
        if (line.find("baseline=") != std::string::npos) 
            sscanf(line.c_str(), "baseline=%f", &baseline);
        if (line.find("doffs=") != std::string::npos) 
            sscanf(line.c_str(), "doffs=%f", &doffs);
    }
}

int main() {
    loadCalibrationFromTxt("../calib.txt");

    cv::Mat frameL = cv::imread("../im0.png");
    cv::Mat frameR = cv::imread("../im1.png");

    if (frameL.empty() || frameR.empty()) return -1;

    FeatureMatcher matcher;
    Epipolar epipolar;
    
    // --- MATCHING ---
    std::vector<cv::KeyPoint> kpL, kpR;
    std::vector<cv::DMatch> goodMatches;
    matcher.findMatches(frameL, frameR, kpL, kpR, goodMatches);
    
    // On récupère l'image des matches dessinés (si ta fonction drawMatches l'affiche elle-même, 
    // il faudra peut-être modifier ta classe pour qu'elle utilise WINDOW_NORMAL)
    matcher.drawMatches(frameL, frameR, kpL, kpR, goodMatches);

    // --- EPIPOLAIRE ---
    if (goodMatches.size() >= 8) {
        std::vector<cv::Point2f> ptsL, ptsR;
        for (auto& m : goodMatches) {
            ptsL.push_back(kpL[m.queryIdx].pt);
            ptsR.push_back(kpR[m.trainIdx].pt);
        }
        
        cv::Mat F = epipolar.computeFundamental(ptsL, ptsR);
        cv::Mat outL = frameL.clone(), outR = frameR.clone();

        // Calcul des lignes pour les deux images
        std::vector<cv::Vec3f> linesL, linesR;
        cv::computeCorrespondEpilines(ptsL, 1, F, linesR); // Lignes sur l'image droite
        cv::computeCorrespondEpilines(ptsR, 2, F, linesL); // Lignes sur l'image gauche

        // Dessin manuel (ou via ta classe si tu la modifies)
        // Ici on s'assure que outL ET outR reçoivent des lignes
        epipolar.drawEpipolarLines(outL, outR, F, ptsL, ptsR); 

        cv::Mat canvas;
        cv::hconcat(outL, outR, canvas);
        showResized("Lignes Epipolaires", canvas, 1600); 
    }

    // --- DISPARITÉ ---
    cv::Mat grayL, grayR, disparity;
    cv::cvtColor(frameL, grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frameR, grayR, cv::COLOR_BGR2GRAY);

    int numDisp = 256; 
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(numDisp, 15);
    bm->compute(grayL, grayR, disparity);

        // Pour avoir le même rendu que la vidéo (Proche = Clair / Rouge) :
    cv::Mat disp8;
    // 1. Normalisation pour que la plus haute disparité (proche) soit à 255
    cv::normalize(disparity, disp8, 0, 255, cv::NORM_MINMAX, CV_8U);

    // 2. Application de la couleur (JET donne Rouge pour les valeurs hautes/proches)


    showResized("Rendu comme la vidéo", disp8, 1280);
    // --- DISTANCE ---
    int cx = frameL.cols / 2;
    int cy = frameL.rows / 2;
    short dValue = disparity.at<short>(cy, cx);

    if (dValue > 0) {
        float d = dValue / 16.0f;
        float Z = (fx * baseline) / (d + doffs);
        std::string text = "Distance: " + std::to_string(Z) + " mm";
        cv::putText(frameL, text, cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 255, 0), 3);
        cv::drawMarker(frameL, cv::Point(cx, cy), cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 40, 3);
    }

    cv::waitKey(0);
    return 0;
}