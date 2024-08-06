#include "tensorflow/lite/c/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    // Load model
    TfLiteModel* model = TfLiteModelCreateFromFile("path_to_your_model.tflite");
    if (model == NULL) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Create interpreter
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteInterpreterOptionsDelete(options);

    if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
        fprintf(stderr, "Failed to allocate tensors\n");
        return 1;
    }

    // Prepare input tensor
    int input_tensor_index = TfLiteInterpreterGetInputTensorIndex(interpreter, 0);
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, input_tensor_index);

    float input_data[] = {1.0, 2.0, 3.0}; // Example input data
    if (TfLiteTensorCopyFromBuffer(input_tensor, input_data, sizeof(input_data)) != kTfLiteOk) {
        fprintf(stderr, "Failed to copy data to input tensor\n");
        return 1;
    }

    // Run inference
    if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
        fprintf(stderr, "Failed to invoke interpreter\n");
        return 1;
    }

    // Retrieve output tensor
    int output_tensor_index = TfLiteInterpreterGetOutputTensorIndex(interpreter, 0);
    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, output_tensor_index);

    float output_data[1];
    if (TfLiteTensorCopyToBuffer(output_tensor, output_data, sizeof(output_data)) != kTfLiteOk) {
        fprintf(stderr, "Failed to copy data from output tensor\n");
        return 1;
    }

    // Example: Print output
    printf("Output: %f\n", output_data[0]);

    // Clean up
    TfLiteInterpreterDelete(interpreter);
    TfLiteModelDelete(model);

    return 0;
}
