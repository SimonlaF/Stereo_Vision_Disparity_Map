#include "stereo_calibration.hpp"
#include <iostream>

StereoCalibrator::StereoCalibrator(cv::Size patternSize, float squareSize) 
    : _patternSize(patternSize), _squareSize(squareSize) {}

bool StereoCalibrator::loadIntrinsics(const std::string& leftFile, const std::string& rightFile) {
    cv::FileStorage fsL(leftFile, cv::FileStorage::READ);
    cv::FileStorage fsR(rightFile, cv::FileStorage::READ);

    if (!fsL.isOpened() || !fsR.isOpened()) return false;

    fsL["camera_matrix"] >> _K1; fsL["dist_coeffs"] >> _dist1;
    fsR["camera_matrix"] >> _K2; fsR["dist_coeffs"] >> _dist2;
    
    return true;
}

void StereoCalibrator::generateObjectPoints(std::vector<cv::Point3f>& objp) const {
    for (int i = 0; i < _patternSize.height; ++i) {
        for (int j = 0; j < _patternSize.width; ++j) {
            objp.emplace_back(j * _squareSize, i * _squareSize, 0.0f);
        }
    }
}

bool StereoCalibrator::runStereoCalibrationFromTwoFiles(const std::string& leftImgPath, 
                                                        const std::string& rightImgPath, 
                                                        const std::string& saveFile) {
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imgPts1, imgPts2;
    std::vector<cv::Point3f> objp;
    generateObjectPoints(objp);

    cv::Mat frame1 = cv::imread(leftImgPath);
    cv::Mat frame2 = cv::imread(rightImgPath);

    if (frame1.empty() || frame2.empty()) {
        std::cerr << "Erreur : Impossible de charger les images." << std::endl;
        return false;
    }

    cv::Mat gray1, gray2;
    cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> corners1, corners2;
    bool found1 = cv::findChessboardCorners(gray1, _patternSize, corners1);
    bool found2 = cv::findChessboardCorners(gray2, _patternSize, corners2);

    if (found1 && found2) {
        cv::cornerSubPix(gray1, corners1, cv::Size(11, 11), cv::Size(-1, -1), 
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));
        cv::cornerSubPix(gray2, corners2, cv::Size(11, 11), cv::Size(-1, -1), 
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));

        imgPts1.push_back(corners1);
        imgPts2.push_back(corners2);
        objectPoints.push_back(objp);
    } else {
        std::cerr << "Échec : Le damier n'est pas visible sur les DEUX images." << std::endl;
        return false;
    }

    cv::Mat R, T, E, F;
    double rms = cv::stereoCalibrate(objectPoints, imgPts1, imgPts2,
                                     _K1, _dist1, _K2, _dist2, gray1.size(),
                                     R, T, E, F, cv::CALIB_FIX_INTRINSIC);

    cv::FileStorage fs(saveFile, cv::FileStorage::WRITE);
    fs << "K1" << _K1 << "D1" << _dist1 << "K2" << _K2 << "D2" << _dist2;
    fs << "R" << R << "T" << T << "RMS" << rms;
    fs.release();

    std::cout << "Calibration finie avec 1 paire. RMS: " << rms << std::endl;
    std::cout << "Génération de la vue rectifiée pour vérification..." << std::endl;

    _K1.convertTo(_K1, CV_64F);
    _K2.convertTo(_K2, CV_64F);
    _dist1.convertTo(_dist1, CV_64F);
    _dist2.convertTo(_dist2, CV_64F);
    R.convertTo(R, CV_64F);
    T.convertTo(T, CV_64F);

    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect validROI[2];

    cv::stereoRectify(_K1, _dist1, _K2, _dist2, gray1.size(), R, T, 
                      R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0, gray1.size(), 
                      &validROI[0], &validROI[1]);

    cv::Mat map11, map12, map21, map22;
    cv::initUndistortRectifyMap(_K1, _dist1, R1, P1, gray1.size(), CV_16SC2, map11, map12);
    cv::initUndistortRectifyMap(_K2, _dist2, R2, P2, gray2.size(), CV_16SC2, map21, map22);

    cv::Mat rect1, rect2;
    cv::remap(frame1, rect1, map11, map12, cv::INTER_LINEAR);
    cv::remap(frame2, rect2, map21, map22, cv::INTER_LINEAR);

    cv::Mat canvas;
    if (rect1.size() != rect2.size() || rect1.type() != rect2.type()) {
        std::cerr << "ERREUR FATALE : Les deux images n'ont pas la même taille ou le même format (Canaux) !" << std::endl;
        return false; 
    }

    cv::hconcat(rect1, rect2, canvas);

    for (int i = 0; i < canvas.rows; i += 32) {
        cv::line(canvas, cv::Point(0, i), cv::Point(canvas.cols, i), cv::Scalar(0, 0, 255), 1);
    }

    cv::imshow("Verification Rectification", canvas);
    cv::waitKey(0); 
    return true;
}

bool StereoCalibrator::runStereoCalibration(int camIdx1, int camIdx2, const std::string& saveFile) {
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imgPts1, imgPts2;
    std::vector<cv::Point3f> objp;
    generateObjectPoints(objp);

    cv::VideoCapture cap1(camIdx1), cap2(camIdx2);
    if (!cap1.isOpened() || !cap2.isOpened()) return false;

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
        
        char key = static_cast<char>(cv::waitKey(1));
        if (key == ' ' && found1 && found2) {
            imgPts1.push_back(corners1);
            imgPts2.push_back(corners2);
            objectPoints.push_back(objp);
            std::cout << "Paire " << imgPts1.size() << " capturée." << std::endl;
        } else if (key == 27 && imgPts1.size() >= 10) {
            break;
        }
    }

    std::cout << "Calcul des paramètres extrinsèques..." << std::endl;
    cv::Mat R, T, E, F;
    
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

bool StereoCalibrator::runStereoCalibrationFromFileSets(const std::vector<std::string>& leftImages,
                                                        const std::vector<std::string>& rightImages,
                                                        const std::string& saveFile) {
    if (leftImages.size() != rightImages.size() || leftImages.empty()) {
        std::cerr << "Erreur : le nombre d'images doit correspondre et > 0" << std::endl;
        return false;
    }

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imgPts1, imgPts2;
    std::vector<cv::Point3f> objp;
    generateObjectPoints(objp);

    for (size_t i = 0; i < leftImages.size(); ++i) {
        cv::Mat imgL = cv::imread(leftImages[i]);
        cv::Mat imgR = cv::imread(rightImages[i]);
        if (imgL.empty() || imgR.empty()) continue;

        cv::Mat grayL, grayR;
        cv::cvtColor(imgL, grayL, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgR, grayR, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> cornersL, cornersR;
        bool foundL = cv::findChessboardCorners(grayL, _patternSize, cornersL);
        bool foundR = cv::findChessboardCorners(grayR, _patternSize, cornersR);

        if (foundL && foundR) {
            cv::cornerSubPix(grayL, cornersL, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));
            cv::cornerSubPix(grayR, cornersR, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));

            imgPts1.push_back(cornersL);
            imgPts2.push_back(cornersR);
            objectPoints.push_back(objp);
        }
    }

    if (objectPoints.empty()) {
        std::cerr << "Erreur : aucune paire valide pour la calibration stéréo" << std::endl;
        return false;
    }

    cv::Mat R, T, E, F;
    cv::Mat imgL = cv::imread(leftImages[0]);
    cv::Size imageSize = imgL.size();

    double rms = cv::stereoCalibrate(objectPoints, imgPts1, imgPts2,
                                     _K1, _dist1, _K2, _dist2, imageSize,
                                     R, T, E, F, cv::CALIB_FIX_INTRINSIC);

    cv::FileStorage fs(saveFile, cv::FileStorage::WRITE);
    fs << "K1" << _K1 << "D1" << _dist1
       << "K2" << _K2 << "D2" << _dist2
       << "R"  << R   << "T"  << T
       << "RMS"<< rms;
    fs.release();

    std::cout << "Calibration stéréo terminée. RMS=" << rms << std::endl;
    return true;
}