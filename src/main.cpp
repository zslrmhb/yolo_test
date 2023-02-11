#include "Camera.h"
#include "Detector.h"

int main(int argc, char ** argv)
{
    std::vector<std::string> class_list;
    std::ifstream ifs("yolo.names")
    std::string line;
    while (getline(ifs, line))
    {
      class_list.push_back(line);
    }

    cv:dnn::Net net;
    net = cv::dnn::readNet("yolov5n.onnx");
    
    Camera camera;
    camera.init();

    Detector det;
    cv::Mat frame;

    while(True)
    {
        camera.getImage(frame);
        std::vector<cv::Mat> detections;
        detections = det.pre_process(frame, net);
        cv::Mat img = det.post_process(frame.clone(), detections, class_list);

        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = cv::dnn::net.getPerfProfile(layersTimes) / freq;
        std::string label = std::format("Inference time : %.2f ms", t);
        cv::putText(img, label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED);

        cv::imshow("Output", img);
    }

  return 0;

}


// to use cuda acceleration
// -net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
// -net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
