#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <fstream>

// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;
 
// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;
 
// Colors.
cv::Scalar BLACK = cv::Scalar(0,0,0);
cv::Scalar BLUE = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED = cv::Scalar(0,0,255);

class Detector {
public:
    Detector();
    ~Detector();
    void draw_label(cv::Mat& input_image, std::string label, int left, int top);
    std::vector<cv::Mat> pre_process(cv::Mat &input_image, cv::dnn::Net &net)
    cv::Mat post_process(cv::Mat &input_image, 
                         std::vector<cv::Mat> &outputs, 
                         const std::vector<std::string> &class_name)
}