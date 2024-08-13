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

constexpr int FRAME_SIZE = 640;

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

void runInference(ConfiguredNetworkGroup &model, vector<InputVStream> &inputStreams, vector<OutputVStream> &outputStreams, VideoCapture &capture)
{
    Expected<unique_ptr<ActivatedNetworkGroup>> expectedActivatedModel = model.activate();
    if (!expectedActivatedModel)
    {
        throw runtime_error("Failed to activate model.");
    }
    cerr << "Model activated." << endl;

    cv::Mat frame;
    while (capture.read(frame))
    {
        InputVStream &inputStream = inputStreams[0];
        size_t inputSize = inputStream.get_frame_size();
        vector<uint8_t> samples(inputSize);

        // transform from HWC to CHW, normalize to [0, 1]
        for (int i = 0; i < frame.cols * frame.rows; i++)
        {
            samples[i] = frame.data[2 + i * 3];
            samples[i + frame.cols * frame.rows] = frame.data[1 + i * 3];
            samples[i + 2 * frame.cols * frame.rows] = frame.data[i * 3];
        }

        // write input data
        hailo_status status = inputStream.write(MemoryView(samples.data(), inputSize));
        if (status != HAILO_SUCCESS)
        {
            throw runtime_error("Failed to write input stream.");
        }

        OutputVStream &outputStream = outputStreams[0];
        vector<float> output(outputStream.get_frame_size());
        status = outputStream.read(MemoryView(output.data(), output.size()));
        if (status != HAILO_SUCCESS)
        {
            throw runtime_error("Failed to read output stream.");
        }

        /*HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));
        roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<uint8_t *>(feature.m_buffers.data()), feature.m_vstream_info));
        filter(roi, outputStream.name());
        std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);

        for (auto &detection : detections) 
        {
            cerr << "Confidence: " << detection->get_confidence() << endl;
            cerr << "coordinates: " << detection->get_bbox().xmin() << " " << detection->get_bbox().ymin() << " " << detection->get_bbox().xmax() << " " << detection->get_bbox().ymax() << endl;
            if (detection->get_confidence()==0) 
            {
                continue;
            }

            HailoBBox bbox = detection->get_bbox();
        
            cv::rectangle(frame, cv::Point2f(bbox.xmin() * float(frame.cols), bbox.ymin() * float(frame.rows)), 
                        cv::Point2f(bbox.xmax() * float(frame.cols), bbox.ymax() * float(frame.rows)), 
                        cv::Scalar(0, 0, 255), 1);
            
            std::cout << "Detection: " << detection->get_label() << ", Confidence: " << std::fixed << std::setprecision(2) << detection->get_confidence() * 100.0 << "%" << std::endl;
        }*/
        
        int numberOfBoxes = static_cast<int>(output[0]);
        float *boxes = output.data() + 1;

        // draw bounding boxes
        for (int i = 0; i < numberOfBoxes; i++)
        {
            int x1 = boxes[i] * frame.cols;
            int y1 = boxes[i + 1] * frame.rows;
            int x2 = boxes[i + 2] * frame.cols;
            int y2 = boxes[i + 3] * frame.rows;

            cv::Rect rect(cv::Point(x1, y1), cv::Point(x2, y2));
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
            cerr << "Detected box: " << x1 << "x" << y1 << " " << x2 << "x" << y2 << endl;
        }
        cerr << endl;

        cv::imshow("YOLOv8 License Plate Detection", frame);

        if (cv::waitKey(5) == 'q')
        {
            break;
        }

        frame.release();
    }

    cerr << "Inference completed." << endl;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <model.hef>" << endl;
        return 1;
    }

    string modelName = argv[1];
    cout << "Model name: " << modelName << endl;

    unique_ptr<Device> device = configureDevice();
    shared_ptr<ConfiguredNetworkGroup> model = configureModel(*device, modelName);
    vector<InputVStream> inputStreams;
    vector<OutputVStream> outputStreams;
    tie(inputStreams, outputStreams) = configureStreams(*model);

    VideoCapture capture("../datasets/plates/sample_250ms.mp4");
    if (!capture.isOpened())
    {
        cerr << "Error: Could not open video." << endl;
        return -1;
    }
    
    runInference(*model, inputStreams, outputStreams, capture);
    capture.release();
    cv::destroyAllWindows();

    return 0;
}