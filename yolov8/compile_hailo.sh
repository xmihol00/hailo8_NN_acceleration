#!/bin/bash

model_name=$1
if [ -z "$1" ]; then
    model_name="yolov8m_plates_e25"
fi

yolo export model=models/$model_name.pt imgsz=640 format=onnx opset=11
hailomz compile --ckpt models/$model_name.onnx --calib-path datasets/plates/valid/images --yaml yolov8m.yaml --classes 1
