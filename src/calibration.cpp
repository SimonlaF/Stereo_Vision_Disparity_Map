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
// Function to calibrate from live camera 
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

        std::string msg = "Captures : " + std::to_string(imagePoints.size()) + " (Esc to end)";
        cv::putText(frame, msg, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Camera Calibration" + std::to_string(camID), frame);

        char key = static_cast<char>(cv::waitKey(1));
        if (key == ' ' && found) {
            imagePoints.push_back(corners);
            objectPoints.push_back(objp);
            std::cout << "Pose saved!" << std::endl;
        } else if (key == 27 && imagePoints.size() >= 10) {
            break;
        }
    }

    if (imagePoints.empty()) {
        std::cerr << "Error : No images captured." << std::endl;
        return false;
    }

    // Final calculation
    cv::Mat K, dist; 
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(objectPoints, imagePoints, gray.size(), K, dist, rvecs, tvecs);
    
    std::cout << "\n K Camera matrix calculated :\n" << K << std::endl;

    // Save results
    cv::FileStorage fs(saveFileName, cv::FileStorage::WRITE);
    fs << "camera_matrix" << K << "dist_coeffs" << dist << "rms" << rms;
    fs.release();

    // Checking reprojection on the last captured frame
    cv::Mat testFrame;
    cap >> testFrame; 
    if (!testFrame.empty()) {
        cv::Mat grayTest;
        cv::cvtColor(testFrame, grayTest, cv::COLOR_BGR2GRAY); // Convert to grayscale for corner detection

        std::vector<cv::Point2f> detectedCorners;
        bool found = cv::findChessboardCorners(grayTest, _patternSize, detectedCorners);

        if (found) {
            std::vector<cv::Point2f> reprojectedPoints;
            cv::projectPoints(objp, rvecs.back(), tvecs.back(), K, dist, reprojectedPoints);   // Reprojection the object points to 2D using the camera Matrix

            for (size_t i = 0; i < detectedCorners.size(); ++i) {
                cv::circle(testFrame, detectedCorners[i], 5, cv::Scalar(0, 255, 0), 2);
                cv::circle(testFrame, reprojectedPoints[i], 2, cv::Scalar(0, 0, 255), -1);
            }
            
            cv::putText(testFrame, "Green: Detected / Red: Reprojected (K)", 
                        cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            cv::imshow("Calibration Check", testFrame);
            cv::waitKey(0);
        }
    }

    std::cout << "Calibration done. RMS: " << rms << " Saved in " << saveFileName << std::endl;
    cv::destroyAllWindows();
    return true;
}


// Function to calibrate from a set of images
bool CameraCalibrator::runCalibrationFromFiles(const std::vector<std::string>& imagePaths, const std::string& saveFileName) {
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<cv::Point3f> objp;
    generateObjectPoints(objp);

    cv::Mat frame, gray;
    cv::Size imageSize;

    std::cout << "Starting calibration from " << imagePaths.size() << " files..." << std::endl;

    for (const std::string& path : imagePaths) {
        frame = cv::imread(path);
        if (frame.empty()) {
            std::cerr << "Error: Could not read image : " << path << std::endl;
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
            
            // Dectection visualization
            cv::drawChessboardCorners(frame, _patternSize, corners, found);
            cv::imshow("Detection " + path, frame);
            cv::waitKey(500);
            std::cout << "Corners found in : " << path << std::endl;
        } else {
            std::cerr << "Checkerboard not found in : " << path << std::endl;  // Print a warning if the checkerboard is not detected in the image
        }
    }

    if (imagePoints.empty()) {
        std::cerr << "Error: No valid images for calibration." << std::endl;
        return false;
    }

    // Final computation of K and distortion coefficients
    cv::Mat K, dist; 
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, K, dist, rvecs, tvecs);
    
    std::cout << "\nComputed K Matrix :\n" << K << std::endl;

    // Save the calibration results
    cv::FileStorage fs(saveFileName, cv::FileStorage::WRITE);
    fs << "camera_matrix" << K << "dist_coeffs" << dist << "rms" << rms;
    fs.release();

    // Checking reprojection on the last image
    cv::Mat verificationFrame = cv::imread(imagePaths.back()); // Load the last image for verification
    if (!verificationFrame.empty()) {
        std::vector<cv::Point2f> reprojectedPoints;
        cv::projectPoints(objp, rvecs.back(), tvecs.back(), K, dist, reprojectedPoints);   // Reprojection the object points to 2D using the camera Matrix

        for (size_t i = 0; i < imagePoints.back().size(); ++i) {
            cv::circle(verificationFrame, imagePoints.back()[i], 5, cv::Scalar(0, 255, 0), 2);
            cv::circle(verificationFrame, reprojectedPoints[i], 2, cv::Scalar(0, 0, 255), -1);
        }
        
        cv::putText(verificationFrame, "Green: Detected / Red: Reprojected (K)", 
                    cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        cv::imshow("Calibration Check", verificationFrame);
        cv::waitKey(0);
    }

    std::cout << "Calibration done. RMS: " << rms << " Saved in " << saveFileName << std::endl;
    cv::destroyAllWindows();
    return true;
}