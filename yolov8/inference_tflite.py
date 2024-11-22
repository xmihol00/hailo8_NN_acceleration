import tensorflow as tf
import cv2
import numpy as np
import torch
import argparse
import time

def boxes(prediction, conf_thres=0.25):
    xc = prediction[:, 4:5].amax(1) > conf_thres  # candidates

    prediction = prediction.transpose(-1, -2)  # sort values in the tensor
    y = torch.empty_like(prediction[..., :4])
    xy = prediction[..., :2]  # centers
    wh = prediction[..., 2:4] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    x = y[0, xc[0]]       # pick based on confidence
    return x

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="models/yolov8m_plates_e25.tflite", help="Path to the model file")
parser.add_argument("-v", "--video", type=str, default="datasets/plates/test_img_per_frame.mp4", help="Path to the video file")
parser.add_argument("-d", "--delay", type=int, default=1, help="Delay between frames, 0 means do not show output video.")
parser.add_argument("-c", "--confidence", type=float, default=0.5, help="Confidence threshold")
parser.add_argument("-q", "--quantized", action="store_true", help="Quantized model")

args = parser.parse_args()

quantized = "quantized" in args.model or args.quantized

# load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=args.model)
interpreter.allocate_tensors()

# get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print()
print(output_details)

cap = cv2.VideoCapture(args.video)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

start = time.time()
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_sample = time.time()

        if quantized:
            frame_normalized = frame.astype(np.int16) - 128
            inputs = np.zeros((1, 3, 640, 640), dtype=np.int8)
            input_data = tf.cast(frame_normalized, tf.int8)
        else:
            # normalize the frame
            frame_normalized = frame / 255.0
            inputs = np.zeros((1, 3, 640, 640), dtype=np.float32)
            input_data = tf.cast(frame_normalized, tf.float32)


        # convert the image from WHC to CHW
        inputs[0, 0] = input_data[:, :, 2]
        inputs[0, 1] = input_data[:, :, 1]
        inputs[0, 2] = input_data[:, :, 0]

        # set the input tensor
        interpreter.set_tensor(input_details[0]['index'], inputs)

        # perform inference
        interpreter.invoke()

        # get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = torch.from_numpy(output_data)
        results = boxes(output_data, args.confidence)
        
        for result in results: # draw bounding boxes
            x1, y1, x2, y2 = map(int, result[:4])
            print(f"Detected box at: ({x1}, {y1}) ({x2}, {y2})")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        end_sample = time.time()
        print(f"Processed frame in {(end_sample - start_sample) * 1000:.3f} ms.", end="\n\n")
        
        if args.delay > 0:
            cv2.imshow("YOLOv8 License Plate Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
except KeyboardInterrupt:
    pass
end = time.time()
print(f"\nInference completed in {end - start:.4f} s.")
