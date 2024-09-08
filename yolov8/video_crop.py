from ultralytics import YOLO
import glob
import cv2
import numpy as np
import argparse
import os
from sort.tracker import SortTracker

def crop(frame, bbox, target_width=720, target_height=1280):
    # unpack the bounding box
    x1, y1, x2, y2 = bbox

    # calculate the center of the bounding box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # initial crop dimensions (centered on the person)
    top_left_x = max(center_x - target_width // 2, 0)
    top_left_y = max(center_y - target_height // 2, 0)
    bottom_right_x = min(center_x + target_width // 2, frame.shape[1])
    bottom_right_y = min(center_y + target_height // 2, frame.shape[0])

    # get the crop dimensions
    current_width = bottom_right_x - top_left_x
    current_height = bottom_right_y - top_left_y

    # expand the crop to match the target size if necessary
    if current_width < target_width:
        delta_x = target_width - current_width
        top_left_x = max(top_left_x - delta_x // 2, 0)
        bottom_right_x = min(bottom_right_x + delta_x // 2, frame.shape[1])
        
        delta_x = bottom_right_x - top_left_x
        if top_left_x == 0:
            bottom_right_x = min(bottom_right_x + delta_x, frame.shape[1])
        elif bottom_right_x == frame.shape[1]:
            top_left_x = max(top_left_x - delta_x, 0)
    elif current_width > target_width:
        delta_x = current_width - target_width
        delta_x_left = delta_x // 2
        delta_x_right = delta_x - delta_x_left
        top_left_x += delta_x_left
        bottom_right_x -= delta_x_right

    if current_height < target_height:
        delta_y = target_height - current_height
        top_left_y = max(top_left_y - delta_y // 2, 0)
        bottom_right_y = min(bottom_right_y + delta_y // 2, frame.shape[0])

        delta_y = bottom_right_y - top_left_y
        if top_left_y == 0:
            bottom_right_y = min(bottom_right_y + delta_y, frame.shape[0])
        elif bottom_right_y == frame.shape[0]:
            top_left_y = max(top_left_y - delta_y, 0)
    elif current_height > target_height:
        delta_y = current_height - target_height
        delta_y_top = delta_y // 2
        delta_y_bottom = delta_y - delta_y_top
        top_left_y += delta_y_top
        bottom_right_y -= delta_y_bottom

    # final crop, ensuring exact target size
    top_left_y = int(max(top_left_y, 0))
    top_left_x = int(max(top_left_x, 0))
    bottom_right_y = int(min(bottom_right_y, frame.shape[0]))
    bottom_right_x = int(min(bottom_right_x, frame.shape[1]))
    cropped_frame = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    return cropped_frame

def track(image):
    results = model(image) # perform inference on the frame
    coordinates_pred = [box.xyxy[0].tolist() + [box.conf.item(), 0] for result in results for box in result.boxes]
    if not coordinates_pred:
        tracks = tracker.update(np.empty((0, 5)))[:, :5]    
    else:
        tracks = tracker.update(np.array(coordinates_pred))[:, :5]
    
    return tracks[0]

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="models/yolov8m_humans.pt", help="Path to the model file")
parser.add_argument("-i", "--images", type=str, default="datasets/humans/valid/images", help="Path to the images directory or video file")
parser.add_argument("-o", "--output", type=str, default="cropped/video.mp4", help="Output path of the cropped video.")
parser.add_argument("-d", "--delay", type=int, default=50, help="Delay between frames, 0 means do not show output video.")

args = parser.parse_args()
model = YOLO(args.model)
tracker = SortTracker()
os.makedirs(os.path.dirname(args.output), exist_ok=True)

if args.images.endswith("mp4"):
    cap = cv2.VideoCapture(args.images)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_size = (522, 864)
    output_video = cv2.VideoWriter(args.output, fourcc, fps, output_size)
    if not output_video.isOpened():
        print("Error: Could not open output video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = track(frame)[:4]
        cropped_frame = crop(frame, bbox, output_size[0], output_size[1])
        if cropped_frame.shape[1] != output_size[0] or cropped_frame.shape[0] != output_size[1]:
            print(f"Warning: Cropped frame size is not {output_size}, actual {(cropped_frame.shape[1], cropped_frame.shape[0])}.")
        output_video.write(cropped_frame)
        if args.delay > 0:
            cv2.imshow("Cropped", cropped_frame)
            if cv2.waitKey(args.delay) & 0xFF == ord('q'):
                break
        
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
else:
    images = sorted(glob.glob(os.path.join(args.images, "*")))
    for imagePath in images:
        # load the image
        image = cv2.imread(imagePath)
        track(image)
