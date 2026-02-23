#include "calibration.hpp"

CameraCalibrator::CameraCalibrator(cv::Size patternSize, float squareSize) 
    : _patternSize(patternSize), _squareSize(squareSize),
      _criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001) {}

void CameraCalibrator::generateObjectPoints(std::vector<cv::Point3f>& objp) {
    for (int i = 0; i < _patternSize.height; i++) {
        for (int j = 0; j < _patternSize.width; j++) {
            objp.push_back(cv::Point3f(j * _squareSize, i * _squareSize, 0));
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

        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), _criteria);
            cv::drawChessboardCorners(frame, _patternSize, corners, found);
        }

        std::string msg = "Captures: " + std::to_string(imagePoints.size()) + " (Echap pour finir)";
        cv::putText(frame, msg, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Calibration Camera " + std::to_string(camID), frame);

        char key = (char)cv::waitKey(1);
        if (key == ' ' && found) {
            imagePoints.push_back(corners);
            objectPoints.push_back(objp);
            std::cout << "Pose enregistrée !" << std::endl;
        } else if (key == 27 && imagePoints.size() >= 10) break;
    }

    // Calcul final
    cv::Mat K, dist;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(objectPoints, imagePoints, gray.size(), K, dist, rvecs, tvecs);

    // Sauvegarde
    cv::FileStorage fs(saveFileName, cv::FileStorage::WRITE);
    fs << "camera_matrix" << K << "dist_coeffs" << dist << "rms" << rms;
    fs.release();

    std::cout << "Calibration terminée. RMS: " << rms << " Sauvegardé dans " << saveFileName << std::endl;
    cv::destroyAllWindows();
    return true;
}