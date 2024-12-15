from ultralytics import YOLO

model = YOLO("yolov8m.yaml")
results = model.train(data="datasets/coco/coco.yaml", epochs=250, batch=32, imgsz=640, lr0=0.01, lrf=0.001, device=1)
model.eval()
