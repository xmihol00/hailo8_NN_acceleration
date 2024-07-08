from hailo_sdk_client import ClientRunner
import os
import idx2numpy as idx

HAILO_HW = "hailo8"
MODEL_NAME = "MNIST_CNN"

runner = ClientRunner(hw_arch=HAILO_HW)
runner.translate_tf_model("weights/saved_model.pb", MODEL_NAME, end_node_names=["MNIST_CNN/relu3/Relu"])

X_test = idx.convert_from_file("../../mnist_ML/mnist/t10k-images.idx3-ubyte") # download the dataset from http://yann.lecun.com/exdb/mnist/
X_test = X_test.reshape(-1, 28, 28, 1)

runner.optimize(X_test)
hef = runner.compile()

with open(f"model.hef", "wb") as f:
    f.write(hef)

os.system(f"rm *.log")