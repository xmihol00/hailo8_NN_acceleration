#!/bin/bash

model_name=$1
if [ -z "$1" ]; then
    model_name="yolov8m_plates_e25"
fi

#yolo export model=models/$model_name.pt imgsz=640 format=onnx opset=11
#hailomz parse --ckpt models/$model_name.onnx --hw-arch hailo8 --yaml yolov8m.yaml
#hailomz optimize --har yolov8m.har --classes 1 --calib-path datasets/humans/train.tfrecord --yaml yolov8m.yaml
#hailomz eval --har yolov8m.har --calib-path datasets/humans/train.tfrecord --yaml yolov8m.yaml --data-path datasets/humans/train.tfrecord
hailomz compile --ckpt models/$model_name.onnx --calib-path datasets/coco/calib_small --yaml hailo_yolov8m.yaml --classes 80 --hw-arch hailo8
