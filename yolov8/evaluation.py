import torch
from torchvision.ops import box_iou
from ultralytics import YOLO
import glob
import cv2
import pandas as pd
import numpy as np

model = YOLO("models/yolov8m_plates_e25.pt")

total_iou = 0
for i, labelPath in enumerate(glob.glob("datasets/plates/test/labels/*.txt")):
    imagePath = labelPath.replace("labels", "images").replace(".txt", ".jpg")
    
    # load the image
    image = cv2.imread(imagePath)
    # load labels
    labels = pd.read_csv(labelPath, header=None, sep=" ", names=["class", "x", "y", "w", "h"])
    # retrieve the bounding box coordinates
    coordinates_gt = labels[["x", "y", "w", "h"]].values
    # convert to xyxy format
    coordinates_gt_xyxy = np.zeros_like(coordinates_gt)
    coordinates_gt_xyxy[:, 0] = (coordinates_gt[:, 0] - coordinates_gt[:, 2] / 2) * image.shape[1]
    coordinates_gt_xyxy[:, 1] = (coordinates_gt[:, 1] - coordinates_gt[:, 3] / 2) * image.shape[0]
    coordinates_gt_xyxy[:, 2] = (coordinates_gt[:, 0] + coordinates_gt[:, 2] / 2) * image.shape[1]
    coordinates_gt_xyxy[:, 3] = (coordinates_gt[:, 1] + coordinates_gt[:, 3] / 2) * image.shape[0]

    results = model(image) # perform inference on the frame
    coordinates_pred = [box.xyxy[0].tolist() for result in results for box in result.boxes]
    if not coordinates_pred:
        continue

    print(coordinates_gt_xyxy, coordinates_pred)
    iou = box_iou(torch.tensor(coordinates_gt_xyxy), torch.tensor(coordinates_pred))
    total_iou += iou.max(dim=1).values.mean().item()

print(f"Average IoU: {total_iou / (i + 1):.4f}")