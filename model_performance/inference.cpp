#include "hailo/hailort.hpp"

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

constexpr int ITERATIONS = 100;

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

void generateInputData(vector<InputVStream> &inputStreams)
{
    InputVStream &inputStream = inputStreams[0];
    auto inputSize = inputStream.get_frame_size();
    vector<uint8_t> samples(inputSize, 1);

    cerr << "Starting input stream..." << endl;
    for (int i = 0; i < ITERATIONS; i++)
    {
        hailo_status status = inputStream.write(MemoryView(samples.data(), inputSize));
        if (status != HAILO_SUCCESS)
        {
            throw runtime_error("Failed to write input stream.");
        }
    }
    cerr << "Input stream completed." << endl;
}

void makePrediction(vector<OutputVStream> &outputStreams)
{
    OutputVStream &outputStream = outputStreams[0];
    cerr << "Output frame size: " << outputStream.get_frame_size() << endl;
    vector<float> data(outputStream.get_frame_size() / sizeof(float));

    cerr << "Starting inference..." << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++)
    {
        hailo_status status = outputStream.read(MemoryView(data.data(), data.size() * sizeof(float)));
        if (status != HAILO_SUCCESS)
        {
            throw runtime_error("Failed to read output stream.");
        }
    }
    auto end = chrono::high_resolution_clock::now();
    cout << "Average inference time: " << chrono::duration<float, chrono::milliseconds::period>((end - start) / ITERATIONS).count() << " ms, FPS: " 
         << (ITERATIONS / chrono::duration<float, chrono::milliseconds::period>(end - start).count()) * 1000 << endl;
}

void runInference(ConfiguredNetworkGroup &model, vector<InputVStream> &inputStreams, vector<OutputVStream> &outputStreams)
{
    Expected<unique_ptr<ActivatedNetworkGroup>> expectedActivatedModel = model.activate();
    if (!expectedActivatedModel)
    {
        throw runtime_error("Failed to activate model.");
    }
    cerr << "Model activated." << endl;

    #pragma omp parallel num_threads(2)
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                generateInputData(inputStreams);
            }
            #pragma omp section
            {
                makePrediction(outputStreams);
            }
        }
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
    
    runInference(*model, inputStreams, outputStreams);

    return 0;
}