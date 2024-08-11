#include <iostream>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/tools/gen_op_registration.h>

using namespace cv;
using namespace std;

const float CONFIDENCE_THRESHOLD = 0.5;

vector<cv::Rect> runInference(cv::Mat &frame, unique_ptr<tflite::Interpreter> &interpreter)
{
    cerr << "Running inference" << endl;

    // image dimensions    
    int input_width = frame.cols;
    int input_height = frame.rows;

    // transform and copy the input data into the input tensor
    auto input_tensor = interpreter->inputs()[0];
    float *input = interpreter->typed_tensor<float>(interpreter->inputs()[0]);
    for (int i = 0; i < input_width * input_height; i++)
    {
        // from HWC to CHW, normalize to [0, 1]
        input[i] = frame.data[2 + i * 3] / 255.0;
        input[i + input_width * input_height] = frame.data[1 + i * 3] / 255.0;
        input[i + 2 * input_width * input_height] = frame.data[i * 3] / 255.0;
    }

    // run inference
    interpreter->Invoke();

    const float *boxes = interpreter->typed_output_tensor<float>(0);
    if (!boxes)
    {
        cerr << "Failed to retrieve output tensors." << endl;
        exit(1);
    }

    // retrieve detected boxes
    vector<cv::Rect> detected_boxes;
    int dims1 = interpreter->tensor(interpreter->outputs()[0])->dims->data[1];
    int dims2 = interpreter->tensor(interpreter->outputs()[0])->dims->data[2];
    for (int i = 0; i < dims2; i++)
    {
        if (boxes[4 * dims2 + i] > CONFIDENCE_THRESHOLD)
        {
            float s1 = boxes[i];
            float s2 = boxes[dims2 + i];
            float w = boxes[2 * dims2 + i] * 0.5;
            float h = boxes[3 * dims2 + i] * 0.5;

            // convert from center, width, height to top-left and right-bottom corners
            int x1 = s1 - w;
            int y1 = s2 - h;
            int x2 = s1 + w;
            int y2 = s2 + h;
            detected_boxes.emplace_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
            cerr << "Detected box: " << x1 << "x" << y1 << " " << x2 << "x" << y2 << endl;
        }
    }

    return detected_boxes;
}

int main(int argc, char **argv)
{
    // load the TFLite model
    const char *model_path = "models/yolov8m_plates_e05.tflite";
    unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model)
    {
        cerr << "Failed to load model" << endl;
        return -1;
    }

    // build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder builder(*model, resolver);
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
        vector<cv::Rect> boxes = runInference(frame, interpreter);

        // draw bounding boxes
        for (const auto &rect : boxes)
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