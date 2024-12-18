import torch
from torchvision.ops import box_iou
from ultralytics import YOLO
import glob
import cv2
import pandas as pd
import numpy as np
import argparse
import os
import yaml

def compute_IoUs_TPs_FPs_FNs(gt_boxes, actual_boxes, confidences, conf_threshold=0.5, iou_threshold=0.0):
    if len(gt_boxes) == 0 and len(actual_boxes) == 0:
        return [], 0, 0, 0
    elif len(gt_boxes) == 0:
        return [0.0] * len(actual_boxes), 0, len(actual_boxes), 0
    elif len(actual_boxes) == 0:
        return [0.0] * len(gt_boxes), 0, 0, len(gt_boxes)

    IoUs = []
    tp = 0
    fp = 0
    fn = 0

    used_indices = {}

    for i, (actual_box, conf) in enumerate(zip(actual_boxes, confidences)):
        if conf < conf_threshold:
            continue

        y1_1, x1_1, y2_1, x2_1 = actual_box
        max_iou = 0.0

        for j, gt_box in enumerate(gt_boxes):
            y1_2, x1_2, y2_2, x2_2 = gt_box

            # compute the coordinates of the intersection rectangle
            inter_y1 = max(y1_1, y1_2)
            inter_x1 = max(x1_1, x1_2)
            inter_y2 = min(y2_1, y2_2)
            inter_x2 = min(x2_1, x2_2)

            # compute the area of the intersection rectangle
            inter_area = max(0.0, inter_y2 - inter_y1) * max(0.0, inter_x2 - inter_x1)

            # compute the area of each bounding box
            area1 = (y2_1 - y1_1) * (x2_1 - x1_1)
            area2 = (y2_2 - y1_2) * (x2_2 - x1_2)

            # compute the union area
            union_area = area1 + area2 - inter_area

            # compute the IoU
            iou = inter_area / union_area if union_area > 0 else 0.0

            # update the maximum IoU for the current box in boxes1
            if iou > max_iou and iou > iou_threshold:
                max_iou = iou
                max_index = j
        
        if max_iou == 0.0:
            fp += 1
        else:
            if max_index in used_indices:
                if max_iou > used_indices[max_index][0]:
                    fp += 1
                    IoUs[used_indices[max_index][1]] = 0.0 # set the IoU of the previous prediction to 0
                    used_indices[max_index] = (max_iou, i)
            else:
                tp += 1
                used_indices[max_index] = (max_iou, i)
        IoUs.append(max_iou)

    fn = len(gt_boxes) - tp        
    return IoUs, tp, fp, fn

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="models/best.pt", help="Path to the model file")
parser.add_argument("-i", "--images", type=str, default="datasets/coco/valid/images", help="Path to the images directory")
parser.add_argument("-l", "--labels", type=str, default="datasets/coco/valid/labels", help="Path to the labels directory")
parser.add_argument("-c", "--classes", type=str, default="datasets/coco/coco.yaml", help="Path to the classes file")
parser.add_argument("-d", "--delay", type=int, default=500, help="Delay between frames, 0 means do not show output video.")

args = parser.parse_args()
model = YOLO(args.model)

images = glob.glob(os.path.join(args.images, "*"))
IoUs = []
TPs = 0
FPs = 0
FNs = 0

# parse the classes file
with open(args.classes, "r") as file:
    all_class_names = yaml.load(file, Loader=yaml.FullLoader)["names"]

for imagePath in images:
    # load the image
    image = cv2.imread(imagePath)
    labelPath = imagePath.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")    

    try:
        labels = pd.read_csv(labelPath, header=None, sep=" ", names=["class", "x", "y", "w", "h"])
    except:
        print(f"Labels not found for {imagePath}")
        continue

    # retrieve the bounding box coordinates
    coordinates_gt = labels[["x", "y", "w", "h"]].values
    classes = labels["class"].values
    class_names = [all_class_names[int(cls)] for cls in classes]

    # convert to xyxy format
    coordinates_gt_xyxy = np.zeros_like(coordinates_gt)
    coordinates_gt_xyxy[:, 0] = (coordinates_gt[:, 0] - coordinates_gt[:, 2] / 2) * image.shape[1]
    coordinates_gt_xyxy[:, 1] = (coordinates_gt[:, 1] - coordinates_gt[:, 3] / 2) * image.shape[0]
    coordinates_gt_xyxy[:, 2] = (coordinates_gt[:, 0] + coordinates_gt[:, 2] / 2) * image.shape[1]
    coordinates_gt_xyxy[:, 3] = (coordinates_gt[:, 1] + coordinates_gt[:, 3] / 2) * image.shape[0]


    predictions_gt = [[] for _ in range(len(all_class_names))]
    # display the ground truth bounding boxes
    for box, class_name, cls in zip(coordinates_gt_xyxy, class_names, classes):
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        cv2.putText(image, class_name, (int(box[0]), int(box[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        predictions_gt[int(cls)].append(box)

    results = model(image, verbose=False) # perform inference on the frame
    coordinates_pred = [box.xyxy[0].tolist() for result in results for box in result.boxes]
    classes = [int(cls) for result in results for cls in result.boxes.cls]
    confidences = [conf for result in results for conf in result.boxes.conf]
    class_names = [all_class_names[cls] for cls in classes]
    
    predictions_actual = [[[], []] for _ in range(len(all_class_names))]
    # display the predicted bounding boxes
    for box, class_name, cls, conf in zip(coordinates_pred, class_names, classes, confidences):
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(image, class_name, (int(box[0]), int(box[3]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        predictions_actual[cls][0].append(box)
        predictions_actual[cls][1].append(conf)
    
    # display the image
    if args.delay > 0:
        cv2.imshow("YOLOv8 prediction vs ground truth", image)
        if cv2.waitKey(args.delay) & 0xFF == ord('q'):
            break

    for gt_box, (actual_box, actual_conf) in zip(predictions_gt, predictions_actual):
        if len(gt_box) == 0 and len(actual_box) == 0:
            continue
        next_IoU, next_TP, next_FP, next_FN = compute_IoUs_TPs_FPs_FNs(gt_box, actual_box, actual_conf)
        IoUs.extend(next_IoU)
        TPs += next_TP
        FPs += next_FP
        FNs += next_FN

print(f"Average IoU: {np.mean(IoUs):.4f}")
print(f"Precision: {TPs / (TPs + FPs):.4f}")
print(f"Recall: {TPs / (TPs + FNs):.4f}")
print(f"F1 Score: {2 * TPs / (2 * TPs + FPs + FNs):.4f}")
