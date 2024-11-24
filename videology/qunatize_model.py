import tensorflow as tf
import onnx
import onnx_tf
from ultralytics import YOLO
import os
import argparse
import cv2
import numpy as np

def representative_dataset_from_video(X):
    while True:
        ret, frame = X.read()
        if not ret:
            break

        # normalize the frame
        frame_normalized = frame / 255.0

        # add a batch dimension
        input_data = tf.expand_dims(frame_normalized, axis=0)
        input_data = tf.cast(input_data, tf.float32)
        input_data = tf.transpose(input_data, perm=[0, 3, 1, 2])

        yield [input_data]

def representative_dataset_from_file_list(X):
    for file in X:
        frame = cv2.imread(file)
        print(file)
        resized_frame = cv2.resize(frame, (224, 224))
        frame_normalized = resized_frame / 255.0
        input_data = tf.expand_dims(frame_normalized, axis=0)
        input_data = tf.cast(input_data, tf.float32)
        #input_data = tf.transpose(input_data, perm=[0, 3, 1, 2])
        yield [input_data]

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="models/yolov8m_plates_e25.pt", help="Path to a model file")
parser.add_argument("-v", "--video", type=str, default="", help="Path to a video file")
parser.add_argument("-d", "--dataset", type=str, default="", help="Path to a dataset")
parser.add_argument("-q", "--quantize", action="store_true", help="Quantize the model")
parser.add_argument("-o", "--onnx_export", action="store_true", help="Perform onnx export")
args = parser.parse_args()

input_arrays = ["input"]
output_arrays = ["MobilenetV2/Predictions/Softmax"]

# load the pb model
converter = tf.lite.TFLiteConverter.from_saved_model(args.model)

# convert onxx to tflite
if args.quantize:
    if args.video != "":
        cap = cv2.VideoCapture(args.video)
    elif args.dataset != "":
        files = os.listdir(args.dataset)
        files = [os.path.join(args.dataset, file) for file in files]
    else:
        print("Please provide a video or dataset for quantization")
        exit()

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.int8
    if args.video != "":
        converter.representative_dataset = lambda x=cap: representative_dataset_from_video(x)
    elif args.dataset != "":
        converter.representative_dataset = lambda x=files: representative_dataset_from_file_list(x)
    tflite_model = converter.convert()

    tf_model_name = os.path.basename(args.model).replace(".pb", "_quantized.tflite")
    with open(tf_model_name, "wb") as f:
        f.write(tflite_model)
else:
    tflite_model = converter.convert()

    tf_model_name = os.path.basename(args.model).replace(".pb", ".tflite")
    with open(tf_model_name, "wb") as f:
        f.write(tflite_model)
