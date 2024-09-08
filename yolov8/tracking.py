import torch
from torchvision.ops import box_iou
from ultralytics import YOLO
import glob
import cv2
import pandas as pd
import numpy as np
import argparse
import os
from sort.tracker import SortTracker

MAGENTA = (255, 0, 255)
ORANGE = (0, 165, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)

COLORS = [MAGENTA, ORANGE, YELLOW, GREEN, BLUE, RED]

def track(image):
    results = model(image) # perform inference on the frame
    coordinates_pred = [box.xyxy[0].tolist() + [box.conf.item(), 0] for result in results for box in result.boxes]
    if not coordinates_pred:
        tracks = tracker.update(np.empty((0, 5)))[:, :5]    
    else:
        tracks = tracker.update(np.array(coordinates_pred))[:, :5]

    # display the predicted bounding boxes
    for box in tracks:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), COLORS[int(box[4])], 2)
    
    # display the image
    cv2.imshow("YOLOv8 tracking", image)
    if cv2.waitKey(args.delay) & 0xFF == ord('q'):
        exit()

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="models/yolov8m_humans.pt", help="Path to the model file")
parser.add_argument("-i", "--images", type=str, default="datasets/humans/valid/images", help="Path to the images directory")
parser.add_argument("-l", "--labels", type=str, default="datasets/humans/valid/labels", help="Path to the labels directory")
parser.add_argument("-d", "--delay", type=int, default=50, help="Delay between frames, 0 means do not show output video.")

args = parser.parse_args()
model = YOLO(args.model)
tracker = SortTracker()


if args.images.endswith("mp4"):
    cap = cv2.VideoCapture(args.images)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        track(frame)
else:
    images = sorted(glob.glob(os.path.join(args.images, "*")))
    for imagePath in images:
        # load the image
        image = cv2.imread(imagePath)
        track(image)
