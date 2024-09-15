from ultralytics import YOLO
import glob
import cv2
import numpy as np
import argparse
import os
from sort.tracker import SortTracker

roundingBoxes = []

def crop(frame, bboxes, target_width=720, target_height=1280, tracks=1, max_box_move=100):
    current_height, current_width = frame.shape[:2]
    print(current_width, current_height)
    
    if len(bboxes) == 0:
        centers_x = [current_width // 2] * tracks
        centers_y = [current_height // 2] * tracks
    else:
        bboxes = bboxes[-1]
        
        global roundingBoxes
        closestBoxes = []
        print(roundingBoxes, bboxes)
        if bboxes:
            for roundingBox in roundingBoxes:
                closestBoxes.append(sorted(bboxes, key=lambda x: (x[0] - roundingBox[0])**2 + (x[1] - roundingBox[1])**2)[0])
            closestBoxes = np.array(closestBoxes)

            # calculate the center of the bounding box
            closest_centers_x = closestBoxes[:, 0]
            closest_centers_y = closestBoxes[:, 1]
        else:
            closest_centers_x = [roundingBox[4] for roundingBox in roundingBoxes]
            closest_centers_y = [roundingBox[5] for roundingBox in roundingBoxes]

        centers_x = []
        centers_y = []
        for rounding_box, center_x, center_y in zip(roundingBoxes, closest_centers_x, closest_centers_y):
            if rounding_box[0] > center_x:
                new_center_x = rounding_box[0] - max_box_move
                if new_center_x < 0:
                    new_center_x = 0
                move_x = rounding_box[0] - new_center_x
            elif rounding_box[2] < center_x:
                new_center_x = rounding_box[2] + max_box_move
                if new_center_x > current_width:
                    new_center_x = current_width
                move_x = new_center_x - rounding_box[2]
            else:
                move_x = 0
            
            if rounding_box[1] > center_y:
                new_center_y = rounding_box[1] - max_box_move
                if new_center_y < 0:
                    new_center_y = 0
                move_y = rounding_box[1] - new_center_y
            elif rounding_box[3] < center_y:
                new_center_y = rounding_box[3] + max_box_move
                if new_center_y > current_height:
                    new_center_y = current_height
                move_y = new_center_y - rounding_box[3]
            else:
                move_y = 0
            
            rounding_box[0] -= move_x
            rounding_box[1] -= move_y
            rounding_box[2] -= move_x
            rounding_box[3] -= move_y
            rounding_box[4] -= move_x
            rounding_box[5] -= move_y

            centers_x.append(rounding_box[4])
            centers_y.append(rounding_box[5])   
    
    print(centers_x, centers_y)
    cropped_frame = np.zeros((target_height, target_width * tracks, 3), dtype=np.uint8)
    for i, (center_x, center_y) in enumerate(zip(centers_x, centers_y)):
        # initial crop dimensions (centered on the person)
        top_left_x = max(center_x - target_width // 2, 0)
        top_left_y = max(center_y - target_height // 2, 0)
        bottom_right_x = min(center_x + target_width // 2, frame.shape[1])
        bottom_right_y = min(center_y + target_height // 2, frame.shape[0])

        # get the crop dimensions
        current_height = bottom_right_y - top_left_y
        current_width = bottom_right_x - top_left_x

        # expand the crop to match the target size if necessary
        if current_width < target_width:
            delta_x = target_width - current_width
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
        if current_width > target_width:
            delta_x = current_width - target_width
            delta_x_left = delta_x // 2
            delta_x_right = delta_x - delta_x_left
            top_left_x += delta_x_left
            bottom_right_x -= delta_x_right

        if current_height < target_height:
            delta_y = target_height - current_height
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
        if current_height > target_height:
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
        print(top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        cropped_frame[:, i * target_width:(i + 1) * target_width, :] = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    return cropped_frame

def track(image):
    results = model(image) # perform inference on the frame
    coordinates_pred = [box.xywh[0].tolist() for result in results for box in result.boxes if box.conf.item() > 0.5]
    return coordinates_pred

    if not coordinates_pred:
        tracks = tracker.update(np.empty((0, 5)))[:, :5].tolist()    
    else:
        tracks = tracker.update(np.array(coordinates_pred))[:, :5].tolist()
    
    if len(tracks) == 0:
        return None
    else:
        return tracks

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
tracker = SortTracker()
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
    print(width_per_track, height_per_track)
    for i in range(args.tracks):
        center_x = half_width_per_track + width_per_track * i
        center_y = height_per_track // 2
        roundingBoxes.append([
            int(center_x - args.rounding_box * width_per_track), 
            int(center_y - args.rounding_box * height_per_track), 
            int(center_x + args.rounding_box * width_per_track), 
            int(center_y + args.rounding_box * height_per_track),
            center_x,
            center_y
        ])

    bboxes = []
    while ret:
        bbox = track(frame)
        if bbox is not None:
            bboxes.append(bbox)
        if len(bboxes) > 10:
            bboxes.pop(0)
        cropped_frame = crop(frame, bboxes, output_size[0] // args.tracks, output_size[1], args.tracks)
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
