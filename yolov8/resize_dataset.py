import os
import cv2

target_height = 640
target_width = 640

source_dataset = "datasets/humans/train/images"
target_dataset = "datasets/humans/train/images"

for file in os.listdir(source_dataset):
    img = cv2.imread(os.path.join(source_dataset, file))
    #resized_img = cv2.resize(img, (target_width, target_height))
    os.remove(os.path.join(source_dataset, file))
    cv2.imwrite(os.path.join(target_dataset, file.replace(".png", ".jpg")), img)