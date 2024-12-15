from ultralytics import YOLO
import torch
import ultralytics.nn.modules.conv as nn
import ultralytics

model = YOLO("yolov8s.yaml")
model.eval()

for m in model.modules():
    if type(m) is nn.Conv and hasattr(m, 'bn'):
        torch.ao.quantization.fuse_modules(m, [["conv", "bn"]], True)
model.model.train()
model.model = model.model.float()
print(next(model.model.parameters()).device)

model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
torch.quantization.prepare_qat(model.model, inplace=True)
results = model.train(data="/home/david/projs/hailo8_NN_acceleration/yolov8/datasets/plates/data.yaml", epochs=1, batch=2, imgsz=640, lr0=0.01, lrf=0.001, device=0, amp=False)

