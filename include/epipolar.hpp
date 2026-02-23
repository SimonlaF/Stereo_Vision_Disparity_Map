#ifndef EPIPOLAR_H
#define EPIPOLAR_H

#include <opencv2/opencv.hpp>
#include <vector>

class Epipolar {
public:
    // Etape 4 : Matrice Fondamentale
    cv::Mat computeFundamental(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2);

    // Etape 5 : Matrice Essentielle et Décomposition
    void solveEssential(const cv::Mat& F, const cv::Mat& K1, const cv::Mat& K2, 
                        cv::Mat& R, cv::Mat& t, std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2);

    // Etape 6 : Lignes épipolaires (Utilitaires)
    void drawEpipolarLines(cv::Mat& img1, cv::Mat& img2, const cv::Mat& F, 
                           const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2);

private:
    // Nettoyage des points aberrants par vérification de chiralité (points devant la caméra)
    void filterPointsWithE(const cv::Mat& E, const cv::Mat& K1, const cv::Mat& K2, 
                          std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2, cv::Mat& R, cv::Mat& t);
};

#endif