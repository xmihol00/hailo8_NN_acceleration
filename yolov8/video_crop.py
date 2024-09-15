from ultralytics import YOLO
import glob
import cv2
import numpy as np
import argparse
import os

class TrackCropping:
    def __init__(self, target_width=720, target_height=810, current_width=1920, current_height=1080, tracks=2, min_box_area=25000, max_box_move=0.1, rounding_box=0.15):
        self.target_width = target_width
        self.target_height = target_height
        self.tracks = tracks
        self.max_move_x = int(max_box_move * target_width)
        self.max_move_y = int(max_box_move * target_height)
        self.rounding_box_height = int(rounding_box * target_height)
        self.rounding_box_width = int(rounding_box * target_width)
        self.min_box_area = min_box_area
        self.rounding_boxes = []
        self.centers = []
        self.width_per_track = current_width // tracks
        self.half_width_per_track = self.width_per_track // 2
        self.width_shift = self.width_per_track // 8
        self.left_centers = []
        self.right_centers = []
        self.averaged_frames = 24
        self.original_centers = []
        
        for i in range(tracks):
            self.centers.append((self.half_width_per_track + self.width_per_track * i, current_height // 2))
            self.original_centers.append((self.half_width_per_track + self.width_per_track * i, current_height // 2))
        
        self.left_centers = [self.centers[0]] * self.averaged_frames
        self.right_centers = [self.centers[1]] * self.averaged_frames
        
        weights_sum = sum(range(1, self.averaged_frames + 1))
        self.weights = [i / weights_sum for i in range(1, self.averaged_frames + 1)]
    
    def crop_tracks_2(self, frame, bounding_boxes):
        left_center_idx, left_center_min, right_center_idx, right_center_min = None, None, None, None
        left_lambda = lambda x: (x[1][0] - (self.original_centers[0][0] - self.width_shift)) ** 2 + (x[1][1] - self.centers[0][1]) ** 2
        right_lambda = lambda x: (x[1][0] - (self.original_centers[1][0] + self.width_shift)) ** 2 + (x[1][1] - self.centers[1][1]) ** 2

        print("before:", bounding_boxes)
        bounding_boxes = list(filter(lambda x: x[2] * x[3] > self.min_box_area, bounding_boxes))
        print("after:", bounding_boxes)
        if len(bounding_boxes) > 0:
            left_center_idx, left_center_min = min(enumerate(bounding_boxes), key=left_lambda)
            right_center_idx, right_center_min = min(enumerate(bounding_boxes), key=right_lambda)
            print("left:", left_center_min, "right:", right_center_min)
        
            if left_center_idx == right_center_idx:
                left_distance = left_lambda((None, left_center_min))
                right_distance = right_lambda((None, right_center_min))
                if left_distance <= right_distance:
                    bounding_boxes.pop(right_center_idx)
                    if len(bounding_boxes) > 0:
                        right_center_idx, right_center_min = min(enumerate(bounding_boxes), key=right_lambda)
                    else:
                        right_center_idx, right_center_min = None, None
                else:
                    bounding_boxes.pop(left_center_idx)
                    if len(bounding_boxes) > 0:
                        left_center_idx, left_center_min = min(enumerate(bounding_boxes), key=left_lambda)
                    else:
                        left_center_idx, left_center_min = None, None
        
        print("left:", left_center_min, "right:", right_center_min)
        left_center_avg = [0, 0]
        right_center_avg = [0, 0]
        for i in range(self.averaged_frames):
            left_center_avg[0] += self.weights[i] * self.left_centers[i][0]
            left_center_avg[1] += self.weights[i] * self.left_centers[i][1]
            right_center_avg[0] += self.weights[i] * self.right_centers[i][0]
            right_center_avg[1] += self.weights[i] * self.right_centers[i][1]
        
        self.left_centers.pop(0)
        if left_center_idx is not None:
            if (left_center_min[0] - left_center_avg[0]) ** 2 + (left_center_min[1] - left_center_avg[1]) ** 2 < self.max_move_x ** 2:
                self.centers[0] = int(left_center_min[0]), int(left_center_min[1])
            else:
                move_x = left_center_avg[0] - self.centers[0][0]
                if abs(move_x) > self.max_move_x:
                    move_x = self.max_move_x if move_x > 0 else -self.max_move_x
                move_y = left_center_avg[1] - self.centers[0][1]
                if abs(move_y) > self.max_move_y:
                    move_y = self.max_move_y if move_y > 0 else -self.max_move_y
                self.centers[0] = (int(self.centers[0][0] + move_x), int(self.centers[0][1] + move_y))
        self.left_centers.append(self.centers[0])
        
        self.right_centers.pop(0)
        if right_center_idx is not None:
            if (right_center_min[0] - right_center_avg[0]) ** 2 + (right_center_min[1] - right_center_avg[1]) ** 2 < self.max_move_x ** 2:
                self.centers[1] = int(right_center_min[0]), int(right_center_min[1])
            else:
                move_x = right_center_avg[0] - self.centers[1][0]
                if abs(move_x) > self.max_move_x:
                    move_x = self.max_move_x if move_x > 0 else -self.max_move_x
                move_y = right_center_avg[1] - self.centers[1][1]
                if abs(move_y) > self.max_move_y:
                    move_y = self.max_move_y if move_y > 0 else -self.max_move_y
                self.centers[1] = (int(self.centers[1][0] + move_x), int(self.centers[1][1] + move_y))
        self.right_centers.append(self.centers[1])
        
        print(self.centers)
        cropped_frame = np.zeros((self.target_height, self.target_width * self.tracks, 3), dtype=np.uint8)
        for i, (center_x, center_y) in enumerate(self.centers):
            # initial crop dimensions (centered on the person)
            top_left_x = max(center_x - self.target_width // 2, 0)
            top_left_y = max(center_y - self.target_height // 2, 0)
            bottom_right_x = min(center_x + self.target_width // 2, frame.shape[1])
            bottom_right_y = min(center_y + self.target_height // 2, frame.shape[0])

            # get the crop dimensions
            current_height = bottom_right_y - top_left_y
            current_width = bottom_right_x - top_left_x

            # expand the crop to match the target size if necessary
            if current_width < self.target_width:
                delta_x = self.target_width - current_width
                delta_x_left = delta_x // 2
                delta_x_right = delta_x - delta_x_left
                top_left_x = max(top_left_x - delta_x_left, 0)
                bottom_right_x = min(bottom_right_x + delta_x_right, frame.shape[1])
                
                delta_x = bottom_right_x - top_left_x
                if top_left_x == 0:
                    bottom_right_x = min(bottom_right_x + delta_x, frame.shape[1])
                elif bottom_right_x == frame.shape[1]:
                    top_left_x = max(top_left_x - delta_x, 0)

            current_width = bottom_right_x - top_left_x
            if current_width > self.target_width:
                delta_x = current_width - self.target_width
                delta_x_left = delta_x // 2
                delta_x_right = delta_x - delta_x_left
                top_left_x += delta_x_left
                bottom_right_x -= delta_x_right

            if current_height < self.target_height:
                delta_y = self.target_height - current_height
                delta_y_top = delta_y // 2
                delta_y_bottom = delta_y - delta_y_top
                top_left_y = max(top_left_y - delta_y_top, 0)
                bottom_right_y = min(bottom_right_y + delta_y_bottom, frame.shape[0])

                delta_y = bottom_right_y - top_left_y
                if top_left_y == 0:
                    bottom_right_y = min(bottom_right_y + delta_y, frame.shape[0])
                elif bottom_right_y == frame.shape[0]:
                    top_left_y = max(top_left_y - delta_y, 0)
            
            current_height = bottom_right_y - top_left_y
            if current_height > self.target_height:
                delta_y = current_height - self.target_height
                delta_y_top = delta_y // 2
                delta_y_bottom = delta_y - delta_y_top
                top_left_y += delta_y_top
                bottom_right_y -= delta_y_bottom

            # final crop, ensuring exact target size
            top_left_y = int(max(top_left_y, 0))
            top_left_x = int(max(top_left_x, 0))
            bottom_right_y = int(min(bottom_right_y, frame.shape[0]))
            bottom_right_x = int(min(bottom_right_x, frame.shape[1]))
            cropped_frame[:, i * self.target_width:(i + 1) * self.target_width, :] = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        return cropped_frame
            

def track(image):
    results = model(image) # perform inference on the frame
    coordinates_pred = [box.xywh[0].tolist() for result in results for box in result.boxes if box.conf.item() > 0.5]
    return coordinates_pred

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
cropper = TrackCropping(target_width=args.width, target_height=args.height, tracks=args.tracks, rounding_box=args.rounding_box)
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
    
    width_per_track = frame.shape[1] // args.tracks
    half_width_per_track = width_per_track // 2
    height_per_track = frame.shape[0]

    while ret:
        bounding_boxes = track(frame)
        cropped_frame = cropper.crop_tracks_2(frame, bounding_boxes)
        if cropped_frame.shape[1] != output_size[0] or cropped_frame.shape[0] != output_size[1]:
            print(f"ERROR: Cropped frame size is not {output_size}, actual {(cropped_frame.shape[1], cropped_frame.shape[0])}.")
            exit()
        output_video.write(cropped_frame)
        if args.delay > 0:
            cv2.imshow("Cropped", cropped_frame)
            if cv2.waitKey(args.delay) & 0xFF == ord('q'):
                break

        ret, frame = cap.read()
        
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
else:
    images = sorted(glob.glob(os.path.join(args.images, "*")))
    for imagePath in images:
        # load the image
        image = cv2.imread(imagePath)
        track(image)
