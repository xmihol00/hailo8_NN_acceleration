import torch
from torchvision.ops import box_iou
import glob
import cv2
import pandas as pd
import numpy as np
import argparse
import os
import tensorflow as tf
import json

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="mobile_net_v2/ssd_mobilenet_v2_coco_quant_postprocess.tflite", help="Path to the model file")
parser.add_argument("-l", "--labels", type=str, default="coco_labels.json", help="Path to the labels file")
parser.add_argument("-i", "--images", type=str, default="coco_samples/", help="Path to the images directory")
parser.add_argument("-d", "--delay", type=int, default=1000, help="Delay between frames, 0 means do not show output video.")

args = parser.parse_args()

with open(args.labels, "r") as f:
    labels_json = json.load(f)

labels = {}
for key, value in labels_json.items():
    labels[value] = key

# load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=args.model)
interpreter.allocate_tensors()

# get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print()
print(output_details)

images = glob.glob(os.path.join(args.images, "*"))
for imagePath in images:
    # load the image and resize it to the input shape
    image = cv2.imread(imagePath)
    inputs = cv2.resize(image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))

    # add a batch dimension
    inputs = np.expand_dims(inputs, axis=0)
    # set the input tensor
    interpreter.set_tensor(input_details[0]['index'], inputs)
    # perform inference
    interpreter.invoke()

    # get bounding boxes
    bounding_boxes = interpreter.get_tensor(output_details[0]['index'])
    # get the confidence scores
    confidence_scores = interpreter.get_tensor(output_details[2]['index'])
    # get the class IDs
    class_ids = interpreter.get_tensor(output_details[1]['index'])

    for bounding_box, confidence_score, class_id in zip(bounding_boxes[0], confidence_scores[0], class_ids[0]):
        if confidence_score > 0.5:
            y1, x1, y2, x2 = bounding_box
            # draw the bounding box
            cv2.rectangle(image, (int(x1 * image.shape[1]), int(y1 * image.shape[0])), (int(x2 * image.shape[1]), int(y2 * image.shape[0])), (0, 255, 0), 2)
            # draw the class ID
            cv2.putText(image, labels[int(class_id)], (int(x1 * image.shape[1]), int(y1 * image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    # display the image
    if args.delay > 0:
        cv2.imshow("SSD mobilenet v2 COCO", image)
        if cv2.waitKey(args.delay) & 0xFF == ord('q'):
            break
