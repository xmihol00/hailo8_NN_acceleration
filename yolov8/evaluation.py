import torch
from torchvision.ops import box_iou
from ultralytics import YOLO
import glob
import cv2
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="models/yolov8n.pt", help="Path to the model file")
parser.add_argument("-i", "--images", type=str, default="datasets/coco/valid/images", help="Path to the images directory")
parser.add_argument("-l", "--labels", type=str, default="datasets/coco/valid/labels", help="Path to the labels directory")
parser.add_argument("-d", "--delay", type=int, default=250, help="Delay between frames, 0 means do not show output video.")

args = parser.parse_args()
model = YOLO(args.model)

images = glob.glob(os.path.join(args.images, "*"))
total_iou = 0
for imagePath in images:
    # load the image
    image = cv2.imread(imagePath)
    
    labelPath = imagePath.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")

    with open(labelPath, "r") as f:
        data = f.read().strip().split("\n")

    # retrieve the ground truth bounding boxes (first 4 entries without the class)
    coordinates_gt = []
    for row in data:
        coordinates_gt.append(list(map(float, row.split()[1:5])))
    coordinates_gt = np.array(coordinates_gt)

    # convert to xyxy format
    coordinates_gt_xyxy = np.zeros_like(coordinates_gt)
    coordinates_gt_xyxy[:, 0] = (coordinates_gt[:, 0] - coordinates_gt[:, 2] / 2) * image.shape[1]
    coordinates_gt_xyxy[:, 1] = (coordinates_gt[:, 1] - coordinates_gt[:, 3] / 2) * image.shape[0]
    coordinates_gt_xyxy[:, 2] = (coordinates_gt[:, 0] + coordinates_gt[:, 2] / 2) * image.shape[1]
    coordinates_gt_xyxy[:, 3] = (coordinates_gt[:, 1] + coordinates_gt[:, 3] / 2) * image.shape[0]

    # display the ground truth bounding boxes
    #for box in coordinates_gt_xyxy:
    #    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

    results = model(image) # perform inference on the frame
    coordinates_pred = [box.xyxy[0].tolist() for result in results for box in result.boxes]
    
    # display the predicted bounding boxes
    for box in coordinates_pred:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    
    # display the image
    if args.delay > 0:
        cv2.imshow("YOLOv8 prediction vs ground truth", image)
        if cv2.waitKey(args.delay) & 0xFF == ord('q'):
            break
    
    if len(coordinates_pred) == 0:
        print("No predictions found.")
        continue
    iou = box_iou(torch.tensor(coordinates_gt_xyxy), torch.tensor(coordinates_pred))
    print(f"IoU: {iou.max(dim=1).values.mean().item():.4f}")
    total_iou += iou.max(dim=1).values.mean().item()

print(f"Total IoU: {total_iou / len(images):.4f}")