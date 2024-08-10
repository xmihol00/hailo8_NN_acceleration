yolo export model=models/best.pt imgsz=640 format=onnx opset=11
hailomz compile --ckpt models/best.onnx --calib-path datasets/plates/calibration/ --yaml yolov8m.yaml --classes 1