import tensorflow as tf
import cv2
import numpy as np
import torch
import argparse
import time
import os

def boxes(predictions, confidence_scores, class_ids, conf_thres=0.25):
    detections = confidence_scores.flatten() > conf_thres  # candidates
    humans = class_ids.flatten() == 0
    candidates = detections & humans

    predictions = predictions[0, candidates]

    return predictions

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="yolov8/yolov8_detection_quantized.tflite", help="Path to a model file")
parser.add_argument("-s", "--samples", type=str, default="coco_samples/", help="Path to sample images")
parser.add_argument("-d", "--delay", type=int, default=1000, help="Delay between frames, 0 means do not show the frames.")
parser.add_argument("-c", "--confidence", type=float, default=0.4, help="Confidence threshold")
parser.add_argument("-q", "--quantized", action="store_true", help="Quantized model")

args = parser.parse_args()

print(args)
quantized = "quantized" in args.model or args.quantized

# load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=args.model)
interpreter.allocate_tensors()

# get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details, output_details, sep="\n\n")
        
try:
    for imagePath in os.listdir(args.samples):
        imagePath = os.path.join(args.samples, imagePath)
        image = cv2.imread(imagePath)
        resized_image = cv2.resize(image, (640, 640))

        if quantized:
            input_data = tf.cast(resized_image, tf.uint8)

            #resized_image = resized_image.astype(np.int16) - 128
            #inputs = np.zeros((1, 3, 640, 640), dtype=np.int8)
            #input_data = tf.cast(resized_image, tf.int8)
        else:
            # normalize the frame
            resized_image = resized_image / 255.0
            input_data = tf.cast(resized_image, tf.float32)

        # add a batch dimension and convert from NHWC to NCHW
        input_data = tf.expand_dims(input_data, axis=0)
        #input_data = tf.transpose(input_data, perm=[0, 3, 1, 2])

        # set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # perform inference
        interpreter.invoke()

        # get the output tensor
        bounding_boxes = (interpreter.get_tensor(output_details[0]['index']) - output_details[0]["quantization"][1]) * output_details[0]["quantization"][0]
        confidence_scores = (interpreter.get_tensor(output_details[1]['index']) - output_details[1]["quantization"][1]) * output_details[1]["quantization"][0]
        class_ids = interpreter.get_tensor(output_details[2]['index'])
        bounding_boxes = bounding_boxes
        results = boxes(bounding_boxes, confidence_scores, class_ids, args.confidence)
        
        for result in results: # draw bounding boxes
            x1, y1, x2, y2 = map(int, result)
            print(f"Detected box at: ({x1}, {y1}) ({x2}, {y2})")
            cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        if args.delay > 0:
            cv2.imshow("YOLOv8 Human Detection", resized_image)
            if cv2.waitKey(args.delay) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    pass
