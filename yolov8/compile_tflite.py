import tensorflow as tf
import onnx
import onnx_tf
from ultralytics import YOLO
import os

#convert pytorch to onnx
os.system("yolo export model=models/yolov8m_plates_e25.pt imgsz=640 format=onnx opset=11")

# convert onxx to tflite
onnx_model = onnx.load("models/yolov8m_plates_e25.onnx")
tf_rep = onnx_tf.backend.prepare(onnx_model)
tf_rep.export_graph("models/yolov8m_plates_e25.pb")
converter = tf.lite.TFLiteConverter.from_saved_model("models/yolov8m_plates_e25.pb")
tflite_model = converter.convert()
with open("models/yolov8m_plates_e25.tflite", "wb") as f:
    f.write(tflite_model)
