try:
    import tensorflow.lite as tf
    accelerated = False
except ImportError:
    import tflite_runtime.interpreter as tf
    accelerated = True
import cv2
import numpy as np
import argparse
import time
import os
from time import time

# setup execution delegate, if empty, uses CPU
if accelerated:
    delegates = [tf.load_delegate("/usr/lib/libvx_delegate.so")]
else:
    delegates = []

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="Path to a model file")
args = parser.parse_args()

interpreter = tf.Interpreter(model_path=args.model, experimental_delegates=delegates, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details, output_details, sep="\n\n")

inputs = np.random.rand(*input_details[0]['shape']).astype(input_details[0]['dtype'])
for _ in range(20):
    start = time()
    interpreter.set_tensor(input_details[0]['index'], inputs)
    interpreter.invoke()
    outputs = interpreter.get_tensor(output_details[0]['index'])
    print(f"Inference time: {time() - start}, FPS: {1 / (time() - start)}")