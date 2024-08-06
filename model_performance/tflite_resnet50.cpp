#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"

#include <chrono>

constexpr int BATCH_SIZE = 10;

int main()
{
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("models/resnet_v1_50.tflite");

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&interpreter);
    interpreter->AllocateTensors();

    int input_tensor_index = interpreter->inputs()[0];

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        auto* input_tensor = interpreter->tensor(input_tensor_index);
        float input_data[] = {1.0, 2.0, 3.0};
        memcpy(input_tensor->data.raw, input_data, sizeof(input_data));

        interpreter->Invoke();

        int output_tensor_index = interpreter->outputs()[0];
        auto* output_tensor = interpreter->tensor(output_tensor_index);
        float* output_data_ptr = output_tensor->data.f;
    }
    auto end = std::chrono::high_resolution_clock::now();

    printf("Output: %f\n", output_data_ptr[0]);
    printf("Time: %f ms\n", std::chrono::duration<float, std::milli>(end - start).count());

    interpreter.reset();
    model.reset();

    return 0;
}
