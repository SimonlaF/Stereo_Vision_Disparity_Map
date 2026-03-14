#include "calibration.hpp"
#include <iostream>

CameraCalibrator::CameraCalibrator(cv::Size patternSize, float squareSize) 
    : _patternSize(patternSize), _squareSize(squareSize),
      _criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001) {}

void CameraCalibrator::generateObjectPoints(std::vector<cv::Point3f>& objp) const {
    for (int i = 0; i < _patternSize.height; ++i) {
        for (int j = 0; j < _patternSize.width; ++j) {
            objp.emplace_back(j * _squareSize, i * _squareSize, 0.0f);
        }
    }
}

bool CameraCalibrator::runCalibration(int camID, const std::string& saveFileName) {
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<cv::Point3f> objp;
    generateObjectPoints(objp);

    cv::VideoCapture cap(camID);
    if (!cap.isOpened()) return false;

    cv::Mat frame, gray;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, _patternSize, corners, 
                                               cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
        
        cv::drawChessboardCorners(frame, _patternSize, corners, found);
        
        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), _criteria);
            cv::drawChessboardCorners(frame, _patternSize, corners, found);
        }

        std::string msg = "Captures: " + std::to_string(imagePoints.size()) + " (Echap pour finir)";
        cv::putText(frame, msg, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Calibration Camera " + std::to_string(camID), frame);

        char key = static_cast<char>(cv::waitKey(1));
        if (key == ' ' && found) {
            imagePoints.push_back(corners);
            objectPoints.push_back(objp);
            std::cout << "Pose enregistrée !" << std::endl;
        } else if (key == 27 && imagePoints.size() >= 10) {
            break;
        }
    }

    if (imagePoints.empty()) {
        std::cerr << "Erreur : Aucune capture réalisée." << std::endl;
        return false;
    }

    // Calcul final
    cv::Mat K, dist; 
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(objectPoints, imagePoints, gray.size(), K, dist, rvecs, tvecs);
    
    std::cout << "\nMatrice K calculée :\n" << K << std::endl;

    // Sauvegarde
    cv::FileStorage fs(saveFileName, cv::FileStorage::WRITE);
    fs << "camera_matrix" << K << "dist_coeffs" << dist << "rms" << rms;
    fs.release();

    // --- PHASE DE VÉRIFICATION ---
    cv::Mat testFrame;
    cap >> testFrame; 
    if (!testFrame.empty()) {
        cv::Mat grayTest;
        cv::cvtColor(testFrame, grayTest, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> detectedCorners;
        bool found = cv::findChessboardCorners(grayTest, _patternSize, detectedCorners);

        if (found) {
            std::vector<cv::Point2f> reprojectedPoints;
            cv::projectPoints(objp, rvecs.back(), tvecs.back(), K, dist, reprojectedPoints);

            for (size_t i = 0; i < detectedCorners.size(); ++i) {
                cv::circle(testFrame, detectedCorners[i], 5, cv::Scalar(0, 255, 0), 2);
                cv::circle(testFrame, reprojectedPoints[i], 2, cv::Scalar(0, 0, 255), -1);
            }
            
            cv::putText(testFrame, "Vert: Detection / Rouge: Reprojection (K)", 
                        cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            cv::imshow("Verification de la Calibration", testFrame);
            cv::waitKey(0);
        }
    }

    std::cout << "Calibration terminée. RMS: " << rms << " Sauvegardé dans " << saveFileName << std::endl;
    cv::destroyAllWindows();
    return true;
}

bool CameraCalibrator::runCalibrationFromFiles(const std::vector<std::string>& imagePaths, const std::string& saveFileName) {
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<cv::Point3f> objp;
    generateObjectPoints(objp);

    cv::Mat frame, gray;
    cv::Size imageSize;

    std::cout << "Début de la calibration à partir de " << imagePaths.size() << " fichiers..." << std::endl;

    for (const std::string& path : imagePaths) {
        frame = cv::imread(path);
        if (frame.empty()) {
            std::cerr << "Impossible de lire l'image : " << path << std::endl;
            continue;
        }

        imageSize = frame.size();
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, _patternSize, corners, 
                                               cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), _criteria);
            
            imagePoints.push_back(corners);
            objectPoints.push_back(objp);
            
            // Visualisation de la détection
            cv::drawChessboardCorners(frame, _patternSize, corners, found);
            cv::imshow("Detection - " + path, frame);
            cv::waitKey(500);
            std::cout << "Coins trouvés dans : " << path << std::endl;
        } else {
            std::cerr << "Damier non trouvé dans : " << path << std::endl;
        }
    }

    if (imagePoints.empty()) {
        std::cerr << "Erreur : Aucune image valide pour la calibration." << std::endl;
        return false;
    }

    // Calcul final
    cv::Mat K, dist; 
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, K, dist, rvecs, tvecs);
    
    std::cout << "\nMatrice K calculée :\n" << K << std::endl;

    // Sauvegarde
    cv::FileStorage fs(saveFileName, cv::FileStorage::WRITE);
    fs << "camera_matrix" << K << "dist_coeffs" << dist << "rms" << rms;
    fs.release();

    // --- PHASE DE VÉRIFICATION ---
    cv::Mat verificationFrame = cv::imread(imagePaths.back());
    if (!verificationFrame.empty()) {
        std::vector<cv::Point2f> reprojectedPoints;
        cv::projectPoints(objp, rvecs.back(), tvecs.back(), K, dist, reprojectedPoints);

        for (size_t i = 0; i < imagePoints.back().size(); ++i) {
            cv::circle(verificationFrame, imagePoints.back()[i], 5, cv::Scalar(0, 255, 0), 2);
            cv::circle(verificationFrame, reprojectedPoints[i], 2, cv::Scalar(0, 0, 255), -1);
        }
        
        cv::putText(verificationFrame, "Vert: Réel / Rouge: Reprojeté", 
                    cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        cv::imshow("Verification Reprojection", verificationFrame);
        cv::waitKey(0);
    }

    std::cout << "Calibration terminée. RMS: " << rms << " Sauvegardé dans " << saveFileName << std::endl;
    cv::destroyAllWindows();
    return true;
}