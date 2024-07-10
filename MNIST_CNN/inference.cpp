#include "hailo/hailort.hpp"

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <thread>
#include <fstream>
#include <iomanip>
#include <chrono>

#include "weights.h"

using namespace std;
using namespace hailort;

constexpr int INPUT_SIZE = 1024;
constexpr int OUTPUT_SIZE = 256;
//#define TRAINING
#ifdef TRAINING
    constexpr int NUM_SAMPLES = 60'000;
#else
    constexpr int NUM_SAMPLES = 10'000;
#endif

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
#ifdef TRAINING
    ifstream samplesFile("mnist_train_X.bin", ios::binary);
#else
    ifstream samplesFile("mnist_test_X.bin", ios::binary);
#endif
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
        inputStream.flush();
        this_thread::sleep_for(chrono::milliseconds(10));
        cerr << "Input data " << i + 1 << " written." << endl;
    }
    samplesFile.close();
}

void makePrediction(vector<OutputVStream> &outputStreams)
{
    OutputVStream &outputStream = outputStreams[0];
    vector<float> data(OUTPUT_SIZE);

#ifdef TRAINING
    ofstream outputFile("cnn_outputs.bin", ios::binary);
    if (!outputFile)
    {
        throw runtime_error("Failed to open output file.");
    }

    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        hailo_status status = outputStream.read(MemoryView(data.data(), data.size() * sizeof(float)));
        if (status != HAILO_SUCCESS)
        {
            throw runtime_error("Failed to read output stream.");
        }

        outputFile.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
        cerr << "Output data " << i + 1 << " saved." << endl;
    }
    outputFile.close();
#else
    ifstream groundTruthFile("mnist_test_y.bin", ios::binary);
    if (!groundTruthFile)
    {
        throw runtime_error("Failed to open ground truth file.");
    }
    vector<uint8_t> labels(NUM_SAMPLES);
    groundTruthFile.read(reinterpret_cast<char *>(labels.data()), labels.size());

    vector<float> prediction(10);
    int correct = 0;

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
        bool same = maxIndex == static_cast<int>(labels[i]);
        correct += same;
        cout << "Prediction: " << maxIndex << " -- Label: " << static_cast<int>(labels[i]) << " -- Same: " << same << endl;
    }
    groundTruthFile.close();

    cout << "Accuracy: " << fixed << setprecision(2) << 100.0 * correct / NUM_SAMPLES << " %" << endl;
#endif
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