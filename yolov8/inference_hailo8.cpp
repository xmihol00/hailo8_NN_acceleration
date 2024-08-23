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

using namespace std;
using namespace hailort;
using namespace cv;

struct Arguments
{
    string modelPath;
    string videoPath;
    int delay;
    float confidenceThreshold;
};

Arguments parseArgs(int argc, char **argv)
{
    Arguments args;
    args.modelPath = "models/yolov8m_plates_e25.hef";
    args.videoPath = "datasets/plates/test_img_per_frame.mp4";
    args.delay = 1;
    args.confidenceThreshold = 0.5;

    for (int i = 1; i < argc - 1; i++) // ensure that the next argument can be always read
    {
        if (string(argv[i]) == "--model" || string(argv[i]) == "-m")
        {
            args.modelPath = argv[++i];
        }
        else if (string(argv[i]) == "--video" || string(argv[i]) == "-v")
        {
            args.videoPath = argv[++i];
        }
        else if (string(argv[i]) == "--delay" || string(argv[i]) == "-d")
        {
            args.delay = stoi(argv[++i]);
        }
        else if (string(argv[i]) == "--confidence" || string(argv[i]) == "-c")
        {
            args.confidenceThreshold = stof(argv[++i]);
        }
    }

    return args;
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

void runInference(ConfiguredNetworkGroup &model, vector<InputVStream> &inputStreams, vector<OutputVStream> &outputStreams, VideoCapture &capture, Arguments &args)
{
    // activate model
    Expected<unique_ptr<ActivatedNetworkGroup>> expectedActivatedModel = model.activate();
    if (!expectedActivatedModel)
    {
        throw runtime_error("Failed to activate model.");
    }
    cerr << "Model activated." << endl;

    cv::Mat frame;
    while (capture.read(frame)) // read video frame by frame
    {
        auto start = chrono::high_resolution_clock::now();

        InputVStream &inputStream = inputStreams[0];
        size_t inputSize = inputStream.get_frame_size();

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
        float *boxes = output.data() + 1;

        // draw bounding boxes
        for (int i = 0; i < numberOfBoxes; i += 5)
        {
            if (boxes[i + 4] > args.confidenceThreshold)
            {
                int y1 = boxes[i] * frame.cols;
                int x1 = boxes[i + 1] * frame.rows;
                int y2 = boxes[i + 2] * frame.cols;
                int x2 = boxes[i + 3] * frame.rows;

                cv::Rect rect(cv::Point(x1, y1), cv::Point(x2, y2));
                cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
                cerr << "Detected box: " << x1 << "x" << y1 << " " << x2 << "x" << y2 << " with confidence: " << boxes[i + 4] << endl;
            }
        }

        auto end = chrono::high_resolution_clock::now();
        cout << "Processed frame in " << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0f << " ms." << endl;
        cerr << endl;

        if (args.delay > 0)
        {
            cv::imshow("YOLOv8 License Plate Detection", frame);

            if (cv::waitKey(args.delay) == 'q')
            {
                break;
            }
        }

        frame.release();
    }

    cerr << "Inference completed." << endl;
}

int main(int argc, char **argv)
{
    Arguments args = parseArgs(argc, argv);

    unique_ptr<Device> device = configureDevice();
    shared_ptr<ConfiguredNetworkGroup> model = configureModel(*device, args.modelPath);
    vector<InputVStream> inputStreams;
    vector<OutputVStream> outputStreams;
    tie(inputStreams, outputStreams) = configureStreams(*model);

    VideoCapture capture(args.videoPath);
    if (!capture.isOpened())
    {
        cerr << "Error: Could not open video." << endl;
        return -1;
    }
    
    auto start = chrono::high_resolution_clock::now();
    runInference(*model, inputStreams, outputStreams, capture, args);
    auto end = chrono::high_resolution_clock::now();
    cout << "Inference completed in " << chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000.0f << " seconds." << endl;
    
    capture.release();
    cv::destroyAllWindows();

    return 0;
}