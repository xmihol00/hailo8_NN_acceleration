#!/usr/bin/env python
########################################################
#
# WORKING OBJECT DETECTION OVER RTSP
# 1280x720@30 or 1920x1080@24 with 10 ms INFERENCE TIME
#
# (C)2024 VIDEOLOGY INDUSTRIAL-GRADE CAMERAS
########################################################
import sys, getopt
import numpy as np
import os
import cv2
import gi
import argparse
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib
import json
import tflite_runtime.interpreter as tf

class InferenceDataFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, args, **properties):
        super(InferenceDataFactory, self).__init__(**properties)
        # Setup frame counter for timestamps
        self.number_frames = 0
        self.duration = (1.0 / args.fps) * Gst.SECOND  # duration of a frame in nanoseconds

        # create opencv Video Capture
        self.cap = cv2.VideoCapture(f'v4l2src device=/dev/video0 extra-controls="controls,horizontal_flip=0,vertical_flip=0" ' \
                                    f'! video/x-raw,width={args.width},height={args.height},framerate={args.fps}/1 ' \
                                    f'! imxvideoconvert_g2d  ' \
                                    f'! video/x-raw,format=BGRA ' \
                                    f'! appsink', cv2.CAP_GSTREAMER)

        # create output stream
        self.launch_string = f'appsrc name=source is-live=true format=GST_FORMAT_TIME ' \
                             f'! video/x-raw,format=BGRA,width={args.width},height={args.height},framerate={args.fps}/1 ' \
                             f'! vpuenc_h264 ' \
                             f'! rtph264pay config-interval=1 name=pay0 pt=96 '
        
        # setup execution delegate, if empty, uses CPU
        if args.accelerated:
            delegates = [tf.load_delegate("/usr/lib/libvx_delegate.so")]
        else:
            delegates = []

        # load labels
        with open(args.labels, "r") as f:
            labels_json = json.load(f)
        self.labels = {}
        for key, value in labels_json.items():
            self.labels[value] = key

        # construct TFLite interpreter
        self.interpreter = tf.Interpreter(model_path=os.path.join(args.model), experimental_delegates=delegates, num_threads=4)
        self.interpreter.allocate_tensors()
        
        # get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = self.input_details[0]['shape']
        

    def on_need_data(self, src, length):
        """
        This function is called when the RTSP server needs data. It will read a frame from the camera, run inference on it and send it to the client
        """
        
        if self.cap.isOpened():
            # read an image from the camera
            ret, image_original = self.cap.read()
            if ret:
                # resize the image to the size required for inference
                image = cv2.resize(image_original, (self.input_size[1], self.input_size[2]))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.expand_dims(image, axis=0)

                # set the input tensor
                self.interpreter.set_tensor(self.input_details[0]['index'], image)
                # execute the inference
                self.interpreter.invoke()
                
                # get bounding boxes
                bounding_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
                # get the confidence scores
                confidence_scores = self.interpreter.get_tensor(self.output_details[2]['index'])
                # get the class IDs
                class_ids = self.interpreter.get_tensor(self.output_details[1]['index'])

                for bounding_box, confidence_score, class_id in zip(bounding_boxes[0], confidence_scores[0], class_ids[0]):
                    if confidence_score > 0.5:
                        y1, x1, y2, x2 = bounding_box
                        # draw the bounding box
                        cv2.rectangle(image_original, (int(x1 * image_original.shape[1]), 
                                                       int(y1 * image_original.shape[0])), 
                                                      (int(x2 * image_original.shape[1]), 
                                                       int(y2 * image_original.shape[0])), (0, 255, 0), 2)
                        # draw the class ID
                        cv2.putText(image_original, self.labels[int(class_id)], (int(x1 * image_original.shape[1]), 
                                                                                 int(y1 * image_original.shape[0])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                # create and setup buffer for the output stream
                data = GLib.Bytes.new_take(image_original.tobytes())
                buf = Gst.Buffer.new_wrapped_bytes(data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1
                print(f"Frame {self.number_frames} sent")

                retval = src.emit('push-buffer', buf)
                if retval != Gst.FlowReturn.OK:
                    print(retval)

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

class RtspServer(GstRtspServer.RTSPServer):
    def __init__(self, args, **properties):
        super(RtspServer, self).__init__(**properties)
        self.set_service(str(args.port))
        self.set_address("0.0.0.0") # use IPv4

        # create a factory and attach it
        self.factory = InferenceDataFactory(args=args)
        self.factory.set_shared(True)
        self.get_mount_points().add_factory("/stream", self.factory)
        self.attach(None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="ssd_mobilenet_v2_quantized.tflite", help="Path to the model file")
    parser.add_argument("-l", "--labels", type=str, default="coco_labels.json", help="Path to the labels file")
    parser.add_argument("-x", "--width", type=int, default=1920, help="Width of the captured video")
    parser.add_argument("-y", "--height", type=int, default=1080, help="Height of the captured video")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Frames per second of the captured video")
    parser.add_argument("-d", "--device", type=str, default="/dev/video0", help="Video device")
    parser.add_argument("-a", "--accelerated", action="store_false", help="Use hardware accelerated inference")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("-p", "--port", type=int, default=25512, help="Port for the RTSP server")
    args = parser.parse_args()
    
    Gst.init(None)
    server = RtspServer(args)
    loop = GLib.MainLoop()
    loop.run()