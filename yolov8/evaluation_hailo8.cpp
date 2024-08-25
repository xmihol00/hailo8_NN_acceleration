#include "hailo/hailort.hpp"
#include <opencv2/opencv.hpp>

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <thread>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace std;
using namespace hailort;
using namespace cv;

struct Arguments
{
    string modelPath;
    string imagesDirPath;
    string labelsDirPath;
    bool showImages;
    float confidenceThreshold;
    int delay;
};

Arguments parseArgs(int argc, char **argv)
{
    Arguments args;
    args.modelPath = "models/yolov8m_plates_e25.hef";
    args.imagesDirPath = "./datasets/plates/test/images/";
    args.labelsDirPath = "./datasets/plates/test/labels/";
    args.confidenceThreshold = 0.5;
    args.showImages = false;
    args.delay = 500;

    for (int i = 1; i < argc - 1; i++) // ensure that the next argument can be always read
    {
        if (string(argv[i]) == "--model" || string(argv[i]) == "-m")
        {
            args.modelPath = argv[++i];
        }
        else if (string(argv[i]) == "--images" || string(argv[i]) == "-i")
        {
            args.imagesDirPath = argv[++i];
        }
        else if (string(argv[i]) == "--labels" || string(argv[i]) == "-l")
        {
            args.labelsDirPath = argv[++i];
        }
        else if (string(argv[i]) == "--confidence" || string(argv[i]) == "-c")
        {
            args.confidenceThreshold = stof(argv[++i]);
        }
        else if (string(argv[i]) == "--show" || string(argv[i]) == "-s")
        {
            args.showImages = true;
        }
        else if (string(argv[i]) == "--delay" || string(argv[i]) == "-d")
        {
            args.delay = stoi(argv[++i]);
        }
    }

    if (string(argv[argc - 1]) == "--show" || string(argv[argc - 1]) == "-s")
    {
        args.showImages = true;
    }

    return args;
}

float computeIoU(const vector<float> &boxes1, const vector<float> &boxes2)
{
    vector<float> result;

    size_t numberOfBoxes1 = boxes1.size() / 4;
    size_t numberOfBoxes2 = boxes2.size() / 4;

    for (size_t i = 0; i < numberOfBoxes1; i++)
    {
        float maxIoU = 0.0f;
        for (size_t j = 0; j < numberOfBoxes2; j++)
        {
            // extract coordinates for the i-th box in boxes1
            float y1_1 = boxes1[i * 4];
            float x1_1 = boxes1[i * 4 + 1];
            float y2_1 = boxes1[i * 4 + 2];
            float x2_1 = boxes1[i * 4 + 3];

            // extract coordinates for the j-th box in boxes2
            float y1_2 = boxes2[j * 4];
            float x1_2 = boxes2[j * 4 + 1];
            float y2_2 = boxes2[j * 4 + 2];
            float x2_2 = boxes2[j * 4 + 3];

            // compute the coordinates of the intersection rectangle
            float inter_y1 = max(y1_1, y1_2);
            float inter_x1 = max(x1_1, x1_2);
            float inter_y2 = min(y2_1, y2_2);
            float inter_x2 = min(x2_1, x2_2);

            // compute the area of the intersection rectangle
            float interArea = max(0.0f, inter_y2 - inter_y1) * max(0.0f, inter_x2 - inter_x1);

            // compute the area of each bounding box
            float area1 = (y2_1 - y1_1) * (x2_1 - x1_1);
            float area2 = (y2_2 - y1_2) * (x2_2 - x1_2);

            // compute the union area
            float unionArea = area1 + area2 - interArea;

            // compute the IoU
            float iou = interArea / unionArea;

            // Update the maximum IoU for the current box in boxes1
            maxIoU = max(maxIoU, iou);
        }
        result.push_back(maxIoU);
    }

    return result.size() > 0 ? accumulate(result.begin(), result.end(), 0.0f) / result.size() : 0.0; // return the mean IoU
}

vector<float> loadBoxesFromFile(const string& filename)
{
    ifstream file(filename);
    vector<float> data;
    string line;

    while (getline(file, line)) // read each line in the file
    {
        istringstream iss(line);
        float classValue, x, y, w, h;

        // parse a row
        iss >> classValue >> x >> y >> w >> h;

        // sort the values
        data.push_back(y - h / 2);
        data.push_back(x - w / 2);
        data.push_back(y + h / 2);
        data.push_back(x + w / 2);
    }

    return data;
}

unique_ptr<Device> configureDevice()
{
    Expected<vector<hailo_pcie_device_info_t>> devices = Device::scan_pcie();
    if (!devices || devices->empty())
    {
        throw runtime_error("No Hailo devices found.");
    }

    Expected<unique_ptr<Device>> expectedDevice = Device::create_pcie(devices.value()[0]);
    if (!expectedDevice)
    {
        throw runtime_error("Failed to create device.");
    }

    cerr << "Device configured." << endl;
    return move(expectedDevice.value());
}

shared_ptr<ConfiguredNetworkGroup> configureModel(Device &device, const string &modelPath)
{
    Expected<Hef> model = Hef::create(modelPath);
    if (!model)
    {
        throw runtime_error("Failed to create model.");
    }

    Expected<NetworkGroupsParamsMap> modelParams = model->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
    if (!modelParams)
    {
        throw runtime_error("Failed to create model params.");
    }

    Expected<ConfiguredNetworkGroupVector> configuration = device.configure(model.value(), modelParams.value());
    if (!configuration)
    {
        throw runtime_error("Failed to configure model.");
    }

    cerr << "Model configured." << endl;
    return move(configuration.value()[0]);
}

pair<vector<InputVStream>, vector<OutputVStream>> configureStreams(ConfiguredNetworkGroup &model)
{
    map<string, hailo_vstream_params_t> inputStreamParams = 
        model.make_input_vstream_params(true, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE).value();
    map<string, hailo_vstream_params_t> outputStreamParams = 
        model.make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE).value();
    Expected<vector<InputVStream>> expectedInputStreams = VStreamsBuilder::create_input_vstreams(model, inputStreamParams);
    Expected<vector<OutputVStream>> expectedOutputStreams = VStreamsBuilder::create_output_vstreams(model, outputStreamParams);
    if (!expectedInputStreams || !expectedOutputStreams)
    {
        throw runtime_error("Failed to create input/output streams.");
    }

    cerr << "Streams configured." << endl;
    return make_pair(move(expectedInputStreams.value()), move(expectedOutputStreams.value()));
}

void runInference(ConfiguredNetworkGroup &model, vector<InputVStream> &inputStreams, vector<OutputVStream> &outputStreams, Arguments &args)
{
    // activate model
    Expected<unique_ptr<ActivatedNetworkGroup>> expectedActivatedModel = model.activate();
    if (!expectedActivatedModel)
    {
        throw runtime_error("Failed to activate model.");
    }
    cerr << "Model activated." << endl;

    InputVStream &inputStream = inputStreams[0];
    size_t inputSize = inputStream.get_frame_size();

    double totalIoU = 0.0f;
    size_t numImages = 0;
    for (const auto& entry : filesystem::directory_iterator(args.imagesDirPath))
    {
        if (entry.path().extension() == ".jpg")
        {
            numImages++;
            string imageFilename = entry.path().filename().string();
            string labelFilename = args.labelsDirPath + imageFilename.replace(imageFilename.find(".jpg"), 4, ".txt");

            // load image
            cv::Mat frame = cv::imread(entry.path().string());
            
            // load bounding boxes from file - ground truth
            vector<float> boxes1 = loadBoxesFromFile(labelFilename);
            if (args.showImages)
            {
                for (size_t i = 0; i < boxes1.size(); i += 4)
                {
                    int y1 = boxes1[i] * frame.cols;
                    int x1 = boxes1[i + 1] * frame.rows;
                    int y2 = boxes1[i + 2] * frame.cols;
                    int x2 = boxes1[i + 3] * frame.rows;

                    cv::Rect rect(cv::Point(x1, y1), cv::Point(x2, y2));
                    cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2);
                }
            }
            
            // write input data to hailo8
            hailo_status status = inputStream.write(MemoryView(frame.data, inputSize));
            if (status != HAILO_SUCCESS)
            {
                throw runtime_error("Failed to write input stream.");
            }

            // read output data from hailo8
            OutputVStream &outputStream = outputStreams[0];
            vector<float> output(outputStream.get_frame_size());
            status = outputStream.read(MemoryView(output.data(), output.size()));
            if (status != HAILO_SUCCESS)
            {
                throw runtime_error("Failed to read output stream.");
            }

            // parse output data
            int numberOfBoxes = static_cast<int>(output[0]);
            float *rawBoxes = output.data() + 1;
            vector<float> boxes2;

            // retrieve bounding boxes
            for (int i = 0; i < numberOfBoxes; i += 5)
            {
                if (rawBoxes[i + 4] > args.confidenceThreshold)
                {

                    boxes2.push_back(rawBoxes[i]);
                    boxes2.push_back(rawBoxes[i + 1]);
                    boxes2.push_back(rawBoxes[i + 2]);
                    boxes2.push_back(rawBoxes[i + 3]);
                    
                    if (args.showImages)
                    {
                        int y1 = rawBoxes[i] * frame.cols;
                        int x1 = rawBoxes[i + 1] * frame.rows;
                        int y2 = rawBoxes[i + 2] * frame.cols;
                        int x2 = rawBoxes[i + 3] * frame.rows;

                        cv::Rect rect(cv::Point(x1, y1), cv::Point(x2, y2));
                        cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
                    }
                }
            }

            if (args.showImages)
            {
                cv::imshow("YOLOv8 License Plate Detection", frame);
                if (cv::waitKey(args.delay) == 'q')
                {
                    break;
                }
            }

            float iou = computeIoU(boxes1, boxes2);
            totalIoU += iou;
            cerr << "IoU for " << imageFilename << ": " << iou << endl;
        }
    }

    double meanIoU = totalIoU / numImages;
    cout << "\nMean IoU: " << meanIoU << endl;
}

int main(int argc, char **argv)
{
    Arguments args = parseArgs(argc, argv);

    unique_ptr<Device> device = configureDevice();
    shared_ptr<ConfiguredNetworkGroup> model = configureModel(*device, args.modelPath);
    vector<InputVStream> inputStreams;
    vector<OutputVStream> outputStreams;
    tie(inputStreams, outputStreams) = configureStreams(*model);

    runInference(*model, inputStreams, outputStreams, args);

    return 0;
}