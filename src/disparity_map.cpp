#include "disparity_map.hpp"

cv::Mat DisparityMap::computeBM(const cv::Mat& rectL, const cv::Mat& rectR, 
                                int numDisparities, int blockSize) {
    
    cv::Mat grayL, grayR;
    
  
    cv::cvtColor(rectL, grayL, cv::COLOR_BGR2GRAY);
 
    
    cv::cvtColor(rectR, grayR, cv::COLOR_BGR2GRAY);

    // Creating the StereoBM object with specified parameters
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(numDisparities, blockSize);
    // UniquenessRatio : Delete matches that are not significantly better than the second-best match
    stereo->setUniquenessRatio(15); 
    
    // TextureThreshold : Ignore regions with low texture (few gradients) to avoid unreliable disparity estimates.
    stereo->setTextureThreshold(10);
    
    // Spekle : Filter small isolated regions in the disparity map that are likely to be noise (speckles) by setting a maximum speckle size and a disparity difference threshold.
    stereo->setSpeckleWindowSize(100);
    stereo->setSpeckleRange(32);

    cv::Mat disparity;
    stereo->compute(grayL, grayR, disparity);

    return disparity;
}

cv::Mat DisparityMap::getVisualMap(const cv::Mat& disparity) {
    if (disparity.empty()) return cv::Mat();
    cv::Mat disp8, colorMap;
    // Normalization of the disparity map to the range [0, 255] for visualization
    cv::normalize(disparity, disp8, 0, 255, cv::NORM_MINMAX, CV_8U);
    // Applying a color map to enhance visualization of disparity (depth)
    // Set pixels with invalid disparity (negative values) to black
    colorMap.setTo(cv::Scalar(0,0,0), disparity < 0);
    return disp8;
}


std::vector<SparsePoint> DisparityMap::computeSparse(const std::vector<cv::KeyPoint>& kpL, 
                                                     const std::vector<cv::KeyPoint>& kpR,
                                                     const std::vector<cv::DMatch>& matches) {
    std::vector<SparsePoint> results;

    for (const auto& m : matches) {
        cv::Point2f ptL = kpL[m.queryIdx].pt;
        cv::Point2f ptR = kpR[m.trainIdx].pt;
        // Checking if the matched keypoints are approximately on the same horizontal line (epipolar constraint) and if the disparity is positive
        if (std::abs(ptL.y - ptR.y) < 2.0f) { 
            float disp = ptL.x - ptR.x;   

            if (disp > 0) { 
                SparsePoint sp;
                sp.ptL = ptL;
                sp.disparity = disp;
                results.push_back(sp);
            }
        }
    }
    return results;
}

void DisparityMap::drawSparse(cv::Mat& canvas, const std::vector<SparsePoint>& sparsePoints) {
    if (sparsePoints.empty()) return;

    // Find the min and max disparity to normalize the colors
    float minD = sparsePoints[0].disparity;
    float maxD = sparsePoints[0].disparity;
    for (const auto& sp : sparsePoints) {
        if (sp.disparity < minD) minD = sp.disparity;
        if (sp.disparity > maxD) maxD = sp.disparity;
    }
    // Draw each sparse point with a color corresponding to its disparity (depth)
    for (const auto& sp : sparsePoints) {
        // Normalize disparity to [0, 1] for color mapping
        float range = (maxD - minD > 0) ? (maxD - minD) : 1.0f;
        float norm = (sp.disparity - minD) / range;
        // Scalar(B, G, R)
        cv::Scalar color(255 * (1 - norm), 0, 255 * norm); 
        cv::circle(canvas, sp.ptL, 4, color, -1); 
        cv::circle(canvas, sp.ptL, 4, cv::Scalar(255, 255, 255), 1); 
    }
}