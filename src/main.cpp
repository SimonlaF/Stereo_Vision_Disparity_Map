#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main() {
    // Creating the perfect chessboard 
    cv::Size patternSize(9, 6); 
    float squareSize = 25.0f; 

    // Jsp ce que c'est encore
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001);

    std::vector<std::vector<cv::Point3f>> objectPoints; // Vector to store perfect chessboard 3D points
    std::vector<std::vector<cv::Point2f>> imagePoints;  // Vector to store chessboard 2D points found from the camera

    // Generating the 3D point 
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < patternSize.height; i++) {
        for (int j = 0; j < patternSize.width; j++) {
            objp.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
        }
    }

    // Init cam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Erreur : Camera introuvable !" << std::endl;
        return -1;
    }

    // 720p for the camera 
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    cv::Mat frame, gray;
    std::cout << "Starting Calibration" << std::endl;
    std::cout << "Space : capture a pose" << std::endl;
    std::cout << "Echap : compute and close" << std::endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Corners detection 
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, patternSize, corners, 
                        cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_FAST_CHECK + cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            // Sub Pixel analyzing : if ever a corner is in between two pixel, it will find its real position
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            cv::drawChessboardCorners(frame, patternSize, corners, found);
        }

        // Printing the number of screenshot
        std::string msg = "Screenshot : " + std::to_string(imagePoints.size());
        cv::putText(frame, msg, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        
        cv::imshow("Calibration LifeCam", frame);
        char key = (char)cv::waitKey(1);

        if (key == ' ' && found) {
            imagePoints.push_back(corners);
            objectPoints.push_back(objp);
            std::cout << "Image" << imagePoints.size() << " saved." << std::endl;
        } 
        else if (key == 27) { 
            if (imagePoints.size() < 10) {
                std::cout << "Not enough screenshot (at least 10)!" << std::endl;
                continue;
            }
            break;
        }
    }

    // Computing
    std::cout << "Computing rotation and translation matrix (K)" << std::endl;
    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    
    // Fonction magique à comprendre
    double rms = cv::calibrateCamera(objectPoints, imagePoints, gray.size(), 
                                     cameraMatrix, distCoeffs, rvecs, tvecs);

    // Save the calibration 
    cv::FileStorage fs("calibration_params.yml", cv::FileStorage::WRITE);
    fs << "camera_matrix" << cameraMatrix;
    fs << "dist_coeffs" << distCoeffs;
    fs << "rms" << rms;
    fs.release();

    std::cout << "\nRESULTS:" << std::endl;
    std::cout << "RMS error : " << rms << " (Target : < 0.5)" << std::endl;
    std::cout << "K Matrix :\n" << cameraMatrix << std::endl;
    return 0;
}