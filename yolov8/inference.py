import cv2
from ultralytics import YOLO
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="models/yolov8m_plates_e25.pt", help="Path to the model file")
parser.add_argument("-v", "--video", type=str, default="datasets/plates/test_img_per_frame.mp4", help="Path to the video file")
parser.add_argument("-d", "--delay", type=int, default=1, help="Delay between frames, 0 means do not show output video.")
parser.add_argument("-c", "--confidence", type=float, default=0.5, help="Confidence threshold")

args = parser.parse_args()

model = YOLO(args.model)
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

        results = model(frame) # perform inference on the frame

        # loop through the detections and draw the bounding boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # extract bounding box coordinates
                conf = box.conf[0]  # confidence score
                if conf > args.confidence:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # green bounding box
                    label = f"Plate {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if args.delay > 0:
            # display the frame with bounding boxes
            cv2.imshow("YOLOv8 License Plate Detection", frame)

            # terminate if 'q' is pressed
            if cv2.waitKey(args.delay) & 0xFF == ord('q'):
                break
except KeyboardInterrupt:
    pass

end = time.time()
print(f"\nInference completed in {end - start:.4f} seconds")

cap.release()
cv2.destroyAllWindows()

