#include "hailo/hailort.hpp"

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <thread>
#include <fstream>
#include <iomanip>

#include "weights.h"

using namespace std;
using namespace hailort;

constexpr int INPUT_SIZE = 784;
constexpr int OUTPUT_SIZE = 256;
constexpr int NUM_SAMPLES = 10'000;

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
    ifstream samplesFile("mnist_test_X.bin", ios::binary);
    if (!samplesFile)
    {
        throw runtime_error("Failed to open input file.");
    }
    vector<uint8_t> samples(INPUT_SIZE * NUM_SAMPLES);
    samplesFile.read(reinterpret_cast<char *>(samples.data()), samples.size());

    InputVStream &inputStream = inputStreams[0];
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        hailo_status status = inputStream.write(MemoryView(samples.data() + i * INPUT_SIZE, INPUT_SIZE * sizeof(uint8_t)));
        if (status != HAILO_SUCCESS)
        {
            throw runtime_error("Failed to write input stream.");
        }
        cerr << "Input data " << i + 1 << " written." << endl;
    }

    vector<uint8_t> padding(INPUT_SIZE);
    for (int i = 0; i < 2'000; i++)
    {
        hailo_status status = inputStream.write(MemoryView(padding.data(), INPUT_SIZE * sizeof(uint8_t)));
        if (status != HAILO_SUCCESS)
        {
            throw runtime_error("Failed to write input stream (padding).");
        }
    }
}

void makePrediction(vector<OutputVStream> &outputStreams)
{
    ifstream groundTruthFile("mnist_test_y.bin", ios::binary);
    if (!groundTruthFile)
    {
        throw runtime_error("Failed to open ground truth file.");
    }
    vector<uint8_t> labels(NUM_SAMPLES);
    groundTruthFile.read(reinterpret_cast<char *>(labels.data()), labels.size());

    OutputVStream &outputStream = outputStreams[0];
    vector<float> data(OUTPUT_SIZE);
    vector<float> prediction(10);

    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        hailo_status status = outputStream.read(MemoryView(data.data(), data.size() * sizeof(float)));
        if (status != HAILO_SUCCESS)
        {
            throw runtime_error("Failed to read output stream.");
        }
        cerr << "Output data " << i + 1 << " read." << endl;

        for (int j = 0; j < 10; j++)
        {
            prediction[j] = 0;
        }

        int weightsIndex = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            #pragma omp simd
            for (int k = 0; k < 10; k++)
            {
                prediction[k] += data[j] * weights[weightsIndex++];
            }
        }

        int maxIndex = 0;
        int max = prediction[0];
        for (int j = 1; j < 10; j++)
        {
            if (prediction[j] > max)
            {
                max = prediction[j];
                maxIndex = j;
            }
        }
        cout << "Prediction: " << maxIndex << " -- Label: " << static_cast<int>(labels[i]) << endl;
    }
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

int main()
{
    unique_ptr<Device> device = configureDevice();
    shared_ptr<ConfiguredNetworkGroup> model = configureModel(*device, "model.hef");
    vector<InputVStream> inputStreams;
    vector<OutputVStream> outputStreams;
    tie(inputStreams, outputStreams) = configureStreams(*model);
    
    runInference(*model, inputStreams, outputStreams);

    return 0;
}