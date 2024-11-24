try:
    import tensorflow.lite as tf
    accelerated = False
except ImportError:
    import tflite_runtime.interpreter as tf
    accelerated = True
import cv2
import numpy as np
import argparse
import time
import os
from time import time

def boxes(predictions, confidence_scores, class_ids, conf_thres=0.25):
    detections = confidence_scores.flatten() > conf_thres  # candidates
    humans = class_ids.flatten() == 0
    candidates = detections & humans

    predictions = predictions[0, candidates]

    return predictions

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="yolov8_detection_quantized.tflite", help="Path to a model file")
parser.add_argument("-s", "--samples", type=str, default="../coco_samples/", help="Path to sample images")
parser.add_argument("-d", "--delay", type=int, default=1000, help="Delay between frames, 0 means do not show the frames.")
parser.add_argument("-c", "--confidence", type=float, default=0.4, help="Confidence threshold")
parser.add_argument("-q", "--quantized", action="store_true", help="Quantized model")

args = parser.parse_args()

# setup execution delegate, if empty, uses CPU
if accelerated:
    delegates = [tf.load_delegate("/usr/lib/libvx_delegate.so")]
else:
    delegates = []

# load TFLite model and allocate tensors.
interpreter = tf.Interpreter(model_path=args.model, experimental_delegates=delegates, num_threads=4)
interpreter.allocate_tensors()

# get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details, output_details, sep="\n\n")
        
try:
    for i, imagePath in enumerate(os.listdir(args.samples)):
        imagePath = os.path.join(args.samples, imagePath)
        image = cv2.imread(imagePath)
        resized_image = cv2.resize(image, (input_details[0]['shape'][2], input_details[0]['shape'][2]))

        # add a batch dimension and convert to the correct data type
        if input_details[0]['dtype'] == np.uint8:
            input_data = (np.expand_dims(resized_image, axis=0)).astype(np.uint8)
        elif input_details[0]['dtype'] == np.float32:
            input_data = (np.expand_dims(resized_image, axis=0) * (1 / 255.0)).astype(np.float32)
        elif input_details[0]['dtype'] == np.int8:
            input_data = (np.expand_dims(resized_image, axis=0) - 128).astype(np.int8)

        # set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # perform inference
        start = time()
        interpreter.invoke()
        print(f"Inference time: {time() - start}, inference FPS: {1 / (time() - start)}")

        if output_details[0]['dtype'] == np.int8:
            predictions = (interpreter.get_tensor(output_details[0]['index']) - output_details[0]["quantization"][1]) * output_details[0]["quantization"][0]
        elif output_details[0]['dtype'] == np.float32:
            predictions = interpreter.get_tensor(output_details[0]['index'])

        print(sum(predictions[0, 4] > 0.5))
        boxes = predictions[0, :4, (predictions[0, 4, :] > 0.5)]
        print(boxes)
        for box in boxes: # draw bounding boxes
            w_half = int(box[2] * image.shape[1] * 0.5)
            h_half = int(box[3] * image.shape[0] * 0.5)
            x1 = int(box[0] * image.shape[1]) - w_half
            y1 = int(box[1] * image.shape[0]) - h_half
            x2 = int(box[0] * image.shape[1]) + w_half
            y2 = int(box[1] * image.shape[0]) + h_half
            print(f"Detected box at: ({x1}, {y1}) ({x2}, {y2})")
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if args.delay > 0:
            cv2.imshow("YOLOv8 Human Detection", image)
            if cv2.waitKey(args.delay) & 0xFF == ord('q'):
                break
    
except KeyboardInterrupt:
    pass
