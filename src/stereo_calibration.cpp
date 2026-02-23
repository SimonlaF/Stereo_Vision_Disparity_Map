#include "stereo_calibration.hpp"
#include <iostream>

StereoCalibrator::StereoCalibrator(cv::Size patternSize, float squareSize) 
    : _patternSize(patternSize), _squareSize(squareSize) {}

bool StereoCalibrator::loadIntrinsics(const std::string& leftFile, const std::string& rightFile) {
    cv::FileStorage fsL(leftFile, cv::FileStorage::READ);
    cv::FileStorage fsR(rightFile, cv::FileStorage::READ);

    if(!fsL.isOpened() || !fsR.isOpened()) return false;

    fsL["camera_matrix"] >> _K1; fsL["dist_coeffs"] >> _dist1;
    fsR["camera_matrix"] >> _K2; fsR["dist_coeffs"] >> _dist2;
    
    return true;
}

void StereoCalibrator::generateObjectPoints(std::vector<cv::Point3f>& objp) {
    for (int i = 0; i < _patternSize.height; i++) {
        for (int j = 0; j < _patternSize.width; j++) {
            objp.push_back(cv::Point3f(j * _squareSize, i * _squareSize, 0));
        }
    }
}

bool StereoCalibrator::runStereoCalibration(int camIdx1, int camIdx2, const std::string& saveFile) {
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imgPts1, imgPts2;
    std::vector<cv::Point3f> objp;
    generateObjectPoints(objp);

    cv::VideoCapture cap1(camIdx1), cap2(camIdx2);
    if (!cap1.isOpened() || !cap2.isOpened()) return false;

    // Ajustement résolution (optionnel, doit être identique à l'étape 1)
    cap1.set(cv::CAP_PROP_FRAME_WIDTH, 1280); cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap2.set(cv::CAP_PROP_FRAME_WIDTH, 1280); cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    cv::Mat frame1, frame2, gray1, gray2;
    std::cout << "Mode Stéréo : Présentez le damier aux deux caméras.\nESPACE pour capturer, ECHAP pour calculer." << std::endl;

    while (true) {
        cap1 >> frame1; cap2 >> frame2;
        if (frame1.empty() || frame2.empty()) break;

        cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners1, corners2;
        bool found1 = cv::findChessboardCorners(gray1, _patternSize, corners1);
        bool found2 = cv::findChessboardCorners(gray2, _patternSize, corners2);

        cv::Mat vis1 = frame1.clone(), vis2 = frame2.clone();
        if (found1 && found2) {
            cv::drawChessboardCorners(vis1, _patternSize, corners1, found1);
            cv::drawChessboardCorners(vis2, _patternSize, corners2, found2);
        }

        cv::imshow("Cam GAUCHE", vis1); cv::imshow("Cam DROITE", vis2);
        
        char key = (char)cv::waitKey(1);
        if (key == ' ' && found1 && found2) {
            imgPts1.push_back(corners1);
            imgPts2.push_back(corners2);
            objectPoints.push_back(objp);
            std::cout << "Paire " << imgPts1.size() << " capturée." << std::endl;
        } else if (key == 27 && imgPts1.size() >= 10) break;
    }

    std::cout << "Calcul des paramètres extrinsèques..." << std::endl;
    cv::Mat R, T, E, F;
    
    // On utilise CALIB_FIX_INTRINSIC car K1, K2 sont déjà parfaits
    double rms = cv::stereoCalibrate(objectPoints, imgPts1, imgPts2,
                    _K1, _dist1, _K2, _dist2, gray1.size(),
                    R, T, E, F, cv::CALIB_FIX_INTRINSIC,
                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));

    cv::FileStorage fs(saveFile, cv::FileStorage::WRITE);
    fs << "K1" << _K1 << "D1" << _dist1 << "K2" << _K2 << "D2" << _dist2;
    fs << "R" << R << "T" << T << "RMS" << rms;
    fs.release();

    std::cout << "Calibration Stéréo OK ! RMS: " << rms << std::endl;
    return true;
}