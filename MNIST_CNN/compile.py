from hailo_sdk_client import ClientRunner, InferenceContext
import os
import numpy as np

HAILO_HW = "hailo8"
MODEL_NAME = "MNIST_CNN"

if __name__ == "__main__":
    runner = ClientRunner(hw_arch=HAILO_HW)
    runner.translate_tf_model("weights/saved_model.pb", MODEL_NAME, start_node_names=["input"], end_node_names=["MNIST_CNN/quant_maxpool3/MaxPool"])

    with InferenceContext(runner, MODEL_NAME) as context:
        tf_model = runner.get_keras_model(context)
        print("Model summary:")
        tf_model.summary()
        for layer in tf_model.layers:
            print(layer.name, layer.input_shape, layer.output_shape)
    exit()

    with open("mnist_test_X.bin", "rb") as f:
        X_test = f.read()
        X_test = np.frombuffer(X_test, dtype=np.uint8).reshape(-1, 32, 32, 1)
    with open("mnist_test_y.bin", "rb") as f:
        y_test = f.read()
        y_test = np.frombuffer(y_test, dtype=np.uint8)

    runner.optimize(X_test)
    hef = runner.compile()
    runner.model_summary()

    with open(f"model_quantized.hef", "wb") as f:
        f.write(hef)

    os.system(f"rm *.log")
        
