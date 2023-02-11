// Taken from https://learnopencv.com/object-detection-using-yolov5-and-opencv-dnn-in-c-and-python/
#include "auto_aim/Detector.h"


// Draw the predicted bounding box.
void Detector::draw_label(cv::Mat& input_image, 
                          std::string label, 
                          int left, 
                          int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = std::max(top, label_size.height);
    // Top left corner.
    cv::Point tlc = cv::Point(left, top);
    // Bottom right corner.
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw black rectangle.
    cv::rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    cv::putText(input_image, label, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

std::vector<cv::Mat> Detector::pre_process(cv::Mat &input_image, 
                                           cv::dnn::Net &net)
{
    // Convert to blob.
    cv::Mat blob;
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

    cv::dnn::net.setInput(blob);

    // Forward propagate.
    std::vector<cv::Mat> outputs;
    cv::dnn::net.forward(outputs, cv::dnn::net.getUnconnectedOutLayersNames());

    return outputs;
}

cv::Mat Detector::post_process(cv::Mat &input_image, 
                               std::vector<cv::Mat> &outputs, 
                               const std::vector<std::string> &class_name) 
{
    // Initialize vectors to hold respective outputs while unwrapping detections.
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes; 

    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;


    // TODO find the number of detection to iterate through
    const int dimensions = 85; // changes to 7
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i) 
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD) 
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            cv::Mat scores(1, class_name.size(), cv::CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD) 
            {
                // Store class ID and confidence in the pre-defined respective vectors.

                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += 85;  // change to 7
    }

    // Perform Non Maximum Suppression and draw predictions.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++) 
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        cv::rectangle(input_image, cv::Point(left, top), cv::Point(left + width, top + height), BLUE, 3*THICKNESS);

        // Get the label for the class name and its confidence.
        std::string label = std::format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        // Draw class labels.
        Detector::draw_label(input_image, label, left, top);
    }
    return input_image;
}
