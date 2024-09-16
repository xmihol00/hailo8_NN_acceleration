from ultralytics import YOLO
import glob
import cv2
import numpy as np
import argparse
import os

class TrackCropping:
    def __init__(self, model, target_width=576, target_height=810, current_width=1920, current_height=1080, tracks=2, min_box_area=25000):
        self.model = model
        self.target_width = target_width
        self.target_height = target_height
        self.tracks = tracks
        self.min_box_area = min_box_area
        self.rounding_boxes = []
        self.centers = []
        self.width_per_track = current_width // tracks
        self.half_width_per_track = self.width_per_track // 2
        self.width_shift = self.width_per_track // 8
        self.original_centers = []
        self.colors = []
        
        for i in range(tracks):
            self.centers.append((self.half_width_per_track + self.width_per_track * i, current_height // 2))
            self.original_centers.append((self.half_width_per_track + self.width_per_track * i, current_height // 2))
            self.colors.append((0, 0, 255))
        
        self.last_centers = [self.centers[i] for i in range(self.tracks)]
    
    def track_2(self, frame):
        results = model(frame) # perform inference on the frame
        bounding_boxes = [box.xywh[0].tolist() for result in results for box in result.boxes if box.conf.item() > 0.5]

        left_center_idx, left_center_min, right_center_idx, right_center_min = None, None, None, None
        left_lambda = lambda x: (x[1][0] - (self.original_centers[0][0] - self.width_shift)) ** 2 + (x[1][1] - self.original_centers[0][1]) ** 2
        right_lambda = lambda x: (x[1][0] - (self.original_centers[1][0] + self.width_shift)) ** 2 + (x[1][1] - self.original_centers[1][1]) ** 2

        bounding_boxes = list(filter(lambda x: x[2] * x[3] > self.min_box_area, bounding_boxes))
        if len(bounding_boxes) > 0:
            left_center_idx, left_center_min = min(enumerate(bounding_boxes), key=left_lambda)
            right_center_idx, right_center_min = min(enumerate(bounding_boxes), key=right_lambda)
        
            if left_center_idx == right_center_idx:
                left_distance = left_lambda((None, left_center_min))
                right_distance = right_lambda((None, right_center_min))
                if left_distance <= right_distance:
                    self.last_centers[0] = left_center_min
                    bounding_boxes.pop(right_center_idx)
                    if len(bounding_boxes) > 0:
                        right_center_idx, right_center_min = min(enumerate(bounding_boxes), key=right_lambda)
                        self.last_centers[1] = right_center_min
                    else:
                        right_center_min = self.last_centers[1]
                        self.colors[1] = (0, 0, 255)
                else:
                    self.last_centers[1] = right_center_min
                    bounding_boxes.pop(left_center_idx)
                    if len(bounding_boxes) > 0:
                        left_center_idx, left_center_min = min(enumerate(bounding_boxes), key=left_lambda)
                        self.last_centers[0] = left_center_min
                    else:
                        left_center_min = self.last_centers[0]
                        self.colors[0] = (0, 0, 255)
            else:
                self.last_centers = [left_center_min, right_center_min]
        else:
            self.colors[0] = (0, 0, 255)
            self.colors[1] = (0, 0, 255)
            left_center_min, right_center_min = self.last_centers

        return left_center_min[0], left_center_min[1], right_center_min[0], right_center_min[1]
    
    def crop_tracks_2(self, frame, bounding_boxes):
        bounding_boxes = np.array(bounding_boxes)
        left_center_avg = np.average(bounding_boxes[:, :2], axis=0)
        right_center_avg = np.average(bounding_boxes[:, 2:], axis=0)
        
        self.centers[0] = int(left_center_avg[0]), int(left_center_avg[1])
        self.colors[0] = (0, 255, 0)
        
        self.centers[1] = int(right_center_avg[0]), int(right_center_avg[1])
        self.colors[1] = (0, 255, 0)
        
        cropped_frame = np.zeros((self.target_height, self.target_width * self.tracks, 3), dtype=np.uint8)
        for i, (center_x, center_y) in enumerate(self.centers):
            # initial crop dimensions (centered on the person)
            top_left_x = max(center_x - self.target_width // 2, 0)
            top_left_y = max(center_y - self.target_height // 2, 0)
            bottom_right_x = min(center_x + self.target_width // 2, frame.shape[1])
            bottom_right_y = min(center_y + self.target_height // 2, frame.shape[0])

            if top_left_x == 0:
                bottom_right_x += self.target_width - (bottom_right_x - top_left_x)
            elif bottom_right_x == frame.shape[1]:
                top_left_x -= self.target_width - (bottom_right_x - top_left_x)
            
            if top_left_y == 0:
                bottom_right_y += self.target_height - (bottom_right_y - top_left_y)
            elif bottom_right_y == frame.shape[0]:
                top_left_y -= self.target_height - (bottom_right_y - top_left_y)

            # final crop, ensuring exact target size
            cropped_frame[:, i * self.target_width:(i + 1) * self.target_width, :] = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            cv2.rectangle(cropped_frame, (self.centers[i][0] - 3 - top_left_x + self.target_width * i, self.centers[i][1] - 3 - top_left_y), 
                                         (self.centers[i][0] + 3 - top_left_x + self.target_width * i, self.centers[i][1] + 3 - top_left_y), self.colors[i], 3)

        return cropped_frame

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="models/yolov8m_humans.pt", help="Path to the model file")
parser.add_argument("-i", "--images", type=str, help="Path to the images directory or video file")
parser.add_argument("-o", "--output", type=str, default="", help="Output path of the cropped video, by default it is going to be the input video stored in a 'cropped' directory.")
parser.add_argument("-d", "--delay", type=int, default=50, help="Delay between frames, 0 means do not show output video.")
parser.add_argument("-t", "--tracks", type=int, default=1, help="Number of object to be followed in the scene")
parser.add_argument("-hc", "--height", type=int, default=1080, help="Height of the cropped video")
parser.add_argument("-wc", "--width", type=int, default=720, help="Width of the cropped video")
parser.add_argument("-rb", "--rounding_box", type=float, default=0.15, help="Size of a rounding box in percentages of the original image")

args = parser.parse_args()
model = YOLO(args.model, verbose=False)
cropper = TrackCropping(model, target_width=args.width, target_height=args.height, tracks=args.tracks)
if not args.output:
    args.output = os.path.join("cropped", os.path.basename(args.images))
os.makedirs(os.path.dirname(args.output), exist_ok=True)

if args.images.endswith("mp4"):
    cap = cv2.VideoCapture(args.images)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_size = (args.width * args.tracks, args.height if args.height < frame.shape[1] - 100 else frame.shape[1] - 100)
    output_video = cv2.VideoWriter(args.output, fourcc, fps, output_size)
    if not output_video.isOpened():
        print("Error: Could not open output video.")
        exit()
    
    frames = [frame]
    bounding_boxes = [cropper.track_2(frame)] * 24
    
    width_per_track = frame.shape[1] // args.tracks
    half_width_per_track = width_per_track // 2
    height_per_track = frame.shape[0]

    for _ in range(24):
        ret, frame = cap.read()
        frames.append(frame)
        bounding_boxes.append(cropper.track_2(frame))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        bounding_boxes.append(cropper.track_2(frame))
        frame = frames.pop(0)
        bounding_boxes.pop(0)
        cropped_frame = cropper.crop_tracks_2(frame, bounding_boxes)
        if cropped_frame.shape[1] != output_size[0] or cropped_frame.shape[0] != output_size[1]:
            print(f"ERROR: Cropped frame size is not {output_size}, actual {(cropped_frame.shape[1], cropped_frame.shape[0])}.")
            exit()
        output_video.write(cropped_frame)
        if args.delay > 0:
            cv2.imshow("Cropped", cropped_frame)
            if cv2.waitKey(args.delay) & 0xFF == ord('q'):
                break

    for frame in frames:
        bounding_boxes.append(cropper.track_2(frame))
        cropped_frame = cropper.crop_tracks_2(frame, bounding_boxes)
        if cropped_frame.shape[1] != output_size[0] or cropped_frame.shape[0] != output_size[1]:
            print(f"ERROR: Cropped frame size is not {output_size}, actual {(cropped_frame.shape[1], cropped_frame.shape[0])}.")
            exit()
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
