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
        resized_image = cv2.resize(image, (640, 640))

        # add a batch dimension and convert from NHWC to NCHW
        input_data = np.expand_dims(resized_image, axis=0)
        #input_data = tf.transpose(input_data, perm=[0, 3, 1, 2])

        # set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # perform inference
        interpreter.invoke()

        # get the output tensor
        bounding_boxes = (interpreter.get_tensor(output_details[0]['index']) - output_details[0]["quantization"][1]) * output_details[0]["quantization"][0]
        confidence_scores = (interpreter.get_tensor(output_details[1]['index']) - output_details[1]["quantization"][1]) * output_details[1]["quantization"][0]
        class_ids = interpreter.get_tensor(output_details[2]['index'])

        detections = bounding_boxes[0, (confidence_scores.flatten() > args.confidence) & (class_ids.flatten() == 0)] / 640
        
        for detection in detections: # draw bounding boxes
            x1 = int(detection[0] * image.shape[1])
            y1 = int(detection[1] * image.shape[0])
            x2 = int(detection[2] * image.shape[1])
            y2 = int(detection[3] * image.shape[0])
            #print(f"Detected box at: ({x1}, {y1}) ({x2}, {y2})")
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        print(i)
                
        if args.delay > 0:
            cv2.imshow("YOLOv8 Human Detection", image)
            if cv2.waitKey(args.delay) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    pass

#real	0m33.488s
#user	0m24.156s
#sys	0m1.041s