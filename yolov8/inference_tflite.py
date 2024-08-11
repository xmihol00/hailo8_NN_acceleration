import tensorflow as tf
import cv2
import numpy as np
from ultralytics.utils.ops import non_max_suppression
import torch

def boxes(
    prediction,
    conf_thres=0.25,
):
    xc = prediction[:, 4:5].amax(1) > conf_thres  # candidates
    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    y = torch.empty_like(prediction[..., :4])
    xy = prediction[..., :2]  # centers
    wh = prediction[..., 2:4] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    x = y[0, xc[0]]  # confidence
    return x


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="models/yolov8m_plates_e05.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the video
video_path = "datasets/plates/sample_250ms.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Normalize the frame
    frame_normalized = frame / 255.0

    # Add a batch dimension
    input_data = tf.expand_dims(frame_normalized, axis=0)
    input_data = tf.cast(input_data, tf.float32)
    #input_data = tf.transpose(input_data, perm=[0, 3, 1, 2])
    inputs = np.zeros((1, 3, 640, 640), dtype=np.float32)
    inputs[0, 0] = input_data[0, :, :, 2]
    inputs[0, 1] = input_data[0, :, :, 1]
    inputs[0, 2] = input_data[0, :, :, 0]

    print(input_data.shape, inputs.flatten()[:10], inputs.flatten()[640*640: 640*640 + 10], inputs.flatten()[2*640*640:2*640*640 + 10]) # [    0.24314     0.22745     0.18824     0.17647     0.19608     0.23137     0.27843     0.30588     0.23529     0.18431] [     0.2902     0.27451     0.23529     0.22353     0.24314     0.27843     0.34118     0.36863      0.3098     0.25882] [    0.21961     0.20392     0.16471     0.15294     0.17255     0.20784     0.26667     0.29412     0.23137     0.18039]

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], inputs)

    # Perform inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    max_conf = output_data[:, 4] > 0.5
    #print(max_conf, max_conf.shape)
    output_data = torch.from_numpy(output_data)
    results = boxes(output_data, 0.5)
    #nms = non_max_suppression(output_data, 0.5, 0.5)
    
    for result in results:
        x1, y1, x2, y2 = map(int, result[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Loop through the detections and draw the bounding boxes
    #for result in nms:
    #    for box in result:
    #        x1, y1, x2, y2 = map(int, box[:4])  # extract bounding box coordinates
    #        conf = box[4]  # confidence score
    #        if conf > 0.5:
    #            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # green bounding box
    #            label = f"Plate {conf:.2f}"
    #            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("YOLOv8 License Plate Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


