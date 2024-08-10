#include <iostream>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/tools/gen_op_registration.h>

using namespace cv;
using namespace std;

const float CONFIDENCE_THRESHOLD = 0.5;

cv::Mat preprocess(cv::Mat &frame, int input_width, int input_height)
{
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(input_width, input_height));
    resized_frame.convertTo(resized_frame, CV_32FC3, 1.0 / 255); // normalize to [0, 1]
    return resized_frame;
}

std::vector<cv::Rect> runInference(cv::Mat &frame, std::unique_ptr<tflite::Interpreter> &interpreter)
{
    int input_height = interpreter->tensor(interpreter->inputs()[0])->dims->data[1];
    int input_width = interpreter->tensor(interpreter->inputs()[0])->dims->data[2];
    cv::Mat input_frame = preprocess(frame, input_width, input_height);

    // copy the input data into the input tensor
    float *input = interpreter->typed_tensor<float>(interpreter->inputs()[0]);
    memcpy(input, input_frame.data, input_width * input_height * 3 * sizeof(float));

    // run inference
    interpreter->Invoke();

    const float *output_boxes = interpreter->typed_output_tensor<float>(0);
    const float *output_scores = interpreter->typed_output_tensor<float>(1);
    const float *output_classes = interpreter->typed_output_tensor<float>(2);

    // retrieve detected boxes
    std::vector<cv::Rect> detected_boxes;
    for (int i = 0; i < interpreter->tensor(interpreter->outputs()[0])->dims->data[1]; ++i)
    {
        if (output_scores[i] >= CONFIDENCE_THRESHOLD)
        {
            int x1 = static_cast<int>(output_boxes[i * 6 + 0] * frame.cols);
            int y1 = static_cast<int>(output_boxes[i * 6 + 1] * frame.rows);
            int x2 = static_cast<int>(output_boxes[i * 6 + 2] * frame.cols);
            int y2 = static_cast<int>(output_boxes[i * 6 + 3] * frame.rows);
            detected_boxes.emplace_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
        }
    }
    return detected_boxes;
}

int main(int argc, char **argv)
{
    // load the TFLite model
    const char *model_path = "models/yolov8m_plates_e05.tflite";
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model)
    {
        cerr << "Failed to load model" << endl;
        return -1;
    }

    // build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::InterpreterBuilder> builder(new tflite::InterpreterBuilder(*model, resolver));
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);

    if (!interpreter)
    {
        cerr << "Failed to create interpreter" << endl;
        return -1;
    }
    interpreter->AllocateTensors();

    cv::VideoCapture cap("datasets/plates/sample_250ms.mp4");
    if (!cap.isOpened())
    {
        cerr << "Error: Could not open video." << endl;
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame))
    {
        // run inference
        std::vector<cv::Rect> detections = runInference(frame, interpreter);

        // draw bounding boxes
        for (const auto &rect : detections)
        {
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("YOLOv8 License Plate Detection", frame);

        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
