import tensorflow as tf
import numpy as np
import time

# download at: https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/centernet/centernet_resnet_v1_18/pretrained/2023-07-18/centernet_resnet_v1_18.zip
MODEL = "models/centernet_res18_with_postprocess.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

data = np.ones((1, 512, 512, 3), dtype=np.float32)

REPETITIONS = 100
start = time.time()
for _ in range(REPETITIONS):
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
end = time.time()

print(f"Average inference time: {(end - start) / REPETITIONS} seconds")