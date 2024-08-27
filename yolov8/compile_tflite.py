import tensorflow as tf
import onnx
import onnx_tf
from ultralytics import YOLO
import os
import argparse
import cv2
import numpy as np

def representative_dataset(X):
    while True:
        ret, frame = X.read()
        if not ret:
            break

        # normalize the frame
        frame_normalized = frame / 255.0

        # add a batch dimension
        input_data = tf.expand_dims(frame_normalized, axis=0)
        input_data = tf.cast(input_data, tf.float32)
        inputs = np.zeros((1, 3, 640, 640), dtype=np.float32)

        # convert the image from WHC to CHW
        inputs[0, 0] = input_data[0, :, :, 2]
        inputs[0, 1] = input_data[0, :, :, 1]
        inputs[0, 2] = input_data[0, :, :, 0]

        yield [inputs]

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="models/yolov8m_plates_e25.pt", help="Path to the model file")
parser.add_argument("-v", "--video", type=str, default="datasets/plates/test_img_per_frame.mp4", help="Path to the video file")
parser.add_argument("-q", "--quantize", type=bool, default=True, help="Quantize the model")
parser.add_argument("-o", "--onnx_export", type=bool, default=False, help="Quantize the model")
args = parser.parse_args()

pb_model_name = args.model.replace(".pt", ".pb")
if args.onnx_export:
    #convert pytorch to onnx
    os.system(f"yolo export model={args.model} imgsz=640 format=onnx opset=11")

    # load the onnx model
    onnx_model_name = args.model.replace(".pt", ".onnx")
    onnx_model = onnx.load(onnx_model_name)
    
    # export the onnx model to a pb file
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_rep.export_graph(pb_model_name)

# load the pb model
converter = tf.lite.TFLiteConverter.from_saved_model(pb_model_name)

# convert onxx to tflite
if args.quantize:
    cap = cv2.VideoCapture(args.video)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = lambda x=cap: representative_dataset(x)
    tflite_model = converter.convert()

    tf_model_name = args.model.replace(".pt", "_quantized.tflite")
    with open(tf_model_name, "wb") as f:
        f.write(tflite_model)
else:
    tflite_model = converter.convert()

    tf_model_name = args.model.replace(".pt", ".tflite")
    with open(tf_model_name, "wb") as f:
        f.write(tflite_model)
