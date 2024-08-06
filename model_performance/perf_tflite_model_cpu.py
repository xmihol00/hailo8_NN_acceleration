import tensorflow as tf
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, choices={"centernet_res18_with_postprocess.tflite", "mobilenet_v3_edgetpu.tflite", "inception_v1.tflite", "efficientnet_lite0.tflite"})
args = parser.parse_args()

MODEL = f"models/{args.model}"

interpreter = tf.lite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if args.model == "centernet_res18_with_postprocess.tflite":
    data = np.ones((1, 512, 512, 3), dtype=np.float32)
elif args.model == "mobilenet_v3_edgetpu.tflite" or args.model == "inception_v1.tflite" or args.model == "efficientnet_lite0.tflite":
    data = np.ones((1, 224, 224, 3), dtype=np.float32)

REPETITIONS = 10
start = time.time()
for _ in range(REPETITIONS):
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
end = time.time()

print(f"Average inference time: {(end - start) / REPETITIONS} seconds, FPS: {REPETITIONS / (end - start)}")