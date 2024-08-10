import cv2
from ultralytics import YOLO

model = YOLO("models/yolov8m_plates_e05.pt")
video_path = "datasets/plates/sample_1s.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame) # perform inference on the frame

    # Loop through the detections and draw the bounding boxes
    for result in results:
        print(vars(result), type(result))
        boxes = result.boxes
        print(vars(boxes))
        for box in boxes:
            print(vars(box))
            exit()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # extract bounding box coordinates
            conf = box.conf[0]  # confidence score
            if conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # green bounding box
                label = f"Plate {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("YOLOv8 License Plate Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()

