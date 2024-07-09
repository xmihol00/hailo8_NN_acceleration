from hailo_sdk_client import ClientRunner, InferenceContext
import os
import idx2numpy as idx
import numpy as np

HAILO_HW = "hailo8"
MODEL_NAME = "MNIST_CNN"

runner = ClientRunner(hw_arch=HAILO_HW)
runner.translate_tf_model("weights/saved_model.pb", MODEL_NAME, end_node_names=["MNIST_CNN/relu3/Relu"])

X_test = idx.convert_from_file("../../mnist_ML/mnist/t10k-images.idx3-ubyte") # download the dataset from http://yann.lecun.com/exdb/mnist/
X_test = X_test.reshape(-1, 28, 28, 1)
X_test.tofile("mnist_test_X.bin")
y_test = idx.convert_from_file("../../mnist_ML/mnist/t10k-labels.idx1-ubyte")
y_test.tofile("mnist_test_y.bin")

runner.optimize(X_test)
hef = runner.compile()
runner.model_summary()
with runner.infer_context(InferenceContext.SDK_QUANTIZED) as context:
    sample = np.ones((1, 28, 28, 1))
    for i in range(28):
        sample[0, i, :, 0] = i
    print(sample.flatten())
    pred = runner.infer(context, sample)
    print(pred)
#with runner.infer_context(InferenceContext.SDK_NATIVE) as context:
#    pred = runner.infer(context, X_test[:1])
#    print(pred.flatten())

#print(X_test[:1], X_test.flatten()[0:28*28])    

with open(f"model.hef", "wb") as f:
    f.write(hef)

os.system(f"rm *.log")
