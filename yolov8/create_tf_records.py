import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from PIL import Image

def _int64_feature(value):
    if isinstance(value, list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_label(label_path):
    bounding_boxes = []

    with open(label_path, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            bounding_box = {
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            }
            bounding_boxes.append(bounding_box)

    return bounding_boxes

def create_tfrecord(filenames, label_dir, tfrecords_filename, num_images):
    
    progress_bar = tqdm(filenames[:num_images])
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, img_path in enumerate(progress_bar):            
            img_jpeg = open(img_path, "rb").read()
            img = np.array(Image.open(img_path))
            image_height = img.shape[0]
            image_width = img.shape[1]

            label_path = os.path.join(label_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
            bbox_annotations = load_label(label_path)
            
            xmin, xmax, ymin, ymax, category_id = [], [], [], [], []
            
            for object_annotations in bbox_annotations:
                x_center = object_annotations["x_center"]
                y_center = object_annotations["y_center"]
                width = object_annotations["width"]
                height = object_annotations["height"]
                
                # Convert center coordinates to corner coordinates
                x = x_center - width / 2
                y = y_center - height / 2

                xmin.append(x)
                xmax.append(x + width)
                ymin.append(y)
                ymax.append(y + height)
                category_id.append(object_annotations["class_id"])

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "height": _int64_feature(image_height),
                        "width": _int64_feature(image_width),
                        "num_boxes": _int64_feature(len(bbox_annotations)),
                        "xmin": _float_list_feature(xmin),
                        "xmax": _float_list_feature(xmax),
                        "ymin": _float_list_feature(ymin),
                        "ymax": _float_list_feature(ymax),
                        "category_id": _int64_feature(category_id),
                        "image_name": _bytes_feature(str.encode(os.path.basename(img_path))),
                        "image_jpeg": _bytes_feature(img_jpeg),
                        "image_id": _int64_feature(i)
                    }
                )
            )
            writer.write(example.SerializeToString())

    return i + 1

dataset = "test"
val_image_dir = f"datasets/plates/{dataset}/images"
label_dir = f"datasets/plates/{dataset}/labels"

image_filenames = [os.path.join(val_image_dir, fname) for fname in os.listdir(val_image_dir) if fname.endswith('.jpg')]
num_images = len(image_filenames)
output_name = f"datasets/plates/{dataset}.tfrecord"

create_tfrecord(image_filenames, label_dir, output_name, num_images)
