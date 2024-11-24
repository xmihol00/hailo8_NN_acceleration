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
from time import time

class TrackCropping:
    def __init__(self, target_width=576, target_height=810, current_width=1920, current_height=1080, average_count=10, show_centers=False):
        self.target_width = target_width
        self.half_target_width = target_width // 2
        self.target_height = target_height
        self.half_target_height = target_height // 2
        self.show_centers = show_centers
        self.average_count = average_count
        self.color = (0, 0, 255)
        
        self.centers = np.array([(current_width // 2, current_height // 2)] * average_count)
        self.next_center_idx = 1
        self.last_center_idx = 0
    
    def crop_frame(self, frame, centers=[]):
        if centers is not None and len(centers) > 0:
            centers[:, 0] = centers[:, 0] * frame.shape[1]
            centers[:, 1] = centers[:, 1] * frame.shape[0]
            last_center = self.centers[self.last_center_idx % self.average_count]
            closest_center = centers[np.argmin(np.sum((centers - last_center) ** 2, axis=1))]
            self.centers[self.next_center_idx % self.average_count] = closest_center
            self.last_center_idx += 1
            self.next_center_idx += 1
        elif centers is not None:
            print("No detection available")

        average_center = np.average(self.centers, axis=0)
        top_left_x = int(max(average_center[0] - self.half_target_width, 0))
        top_left_y = int(max(average_center[1] - self.half_target_height, 0))
        bottom_right_x = int(min(average_center[0] + self.half_target_width, frame.shape[1]))
        bottom_right_y = int(min(average_center[1] + self.half_target_height, frame.shape[0]))

        if top_left_x == 0:
            bottom_right_x += self.target_width
        elif bottom_right_x == frame.shape[1]:
            top_left_x = frame.shape[1] - self.target_width

        if top_left_y == 0:
            bottom_right_y += self.target_height
        elif bottom_right_y == frame.shape[0]:
            top_left_y = frame.shape[0] - self.target_height       
        
        if self.show_centers:
            cv2.rectangle(frame, (int(closest_center[0]) - 3, int(closest_center[1]) - 3), (int(closest_center[0]) + 3, int(closest_center[1]) + 3), self.color, 3)

        cropped_frame = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        return cropped_frame

class InferenceDataFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, args, interpreter, **properties):
        super(InferenceDataFactory, self).__init__(**properties)
        
        # setup frame counter for timestamps
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
                             f'! video/x-raw,format=BGRA,width={args.cropped_width},height={args.cropped_height},framerate={args.fps}/1 ' \
                             f'! vpuenc_h264 bitrate=10000000 ' \
                             f'! rtph264pay config-interval=1 name=pay0 pt=96 '

        self.interpreter = interpreter
        
        # get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = self.input_details[0]['shape']

        # setup tracking
        self.cropper = TrackCropping(target_width=args.cropped_width, target_height=args.cropped_height, current_width=args.width, 
                                     current_height=args.height, average_count=args.crop_average, show_centers=args.show_centers)

        # get the confidence threshold for detection
        self.confidence = args.confidence
        self.skip_frames = args.skip_frames

        # setup time counter for FPS
        self.last_time = time()

        

    def on_need_data(self, src, length):
        """
        This function is called when the RTSP server needs data. It will read a frame from the camera, run inference on it and send it to the client
        """
        
        if self.cap.isOpened():
            # read an image from the camera
            ret, image_original = self.cap.read()
            if ret:
                if self.number_frames % self.skip_frames == 0:
                    # resize the image to the size required for inference
                    image = cv2.resize(image_original, (self.input_size[1], self.input_size[2]))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = np.expand_dims(image, axis=0)

                    # add a batch dimension and convert to the correct data type
                    if self.input_details[0]['dtype'] == np.uint8:
                        input_data = image.astype(np.uint8)
                    elif self.input_details[0]['dtype'] == np.float32:
                        input_data = (image * (1 / 255.0)).astype(np.float32)
                    elif self.input_details[0]['dtype'] == np.int8:
                        input_data = (image - 128).astype(np.int8)

                    # set the input tensor
                    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                    # execute the inference
                    self.interpreter.invoke()

                    # get the output tensor and convert it to the correct data type
                    if self.output_details[0]['dtype'] == np.int8:
                        predictions = (self.interpreter.get_tensor(self.output_details[0]['index']) - self.output_details[0]["quantization"][1]) * self.output_details[0]["quantization"][0]
                    elif self.output_details[0]['dtype'] == np.float32:
                        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
                    
                    # get the centers of the detected objects
                    centers = predictions[0, :2, (predictions[0, 4, :] >= self.confidence)]
                else:
                    centers = None

                # crop the frame
                cropped_frame = self.cropper.crop_frame(image_original, centers)

                # create and setup buffer for the output stream
                data = GLib.Bytes.new_take(cropped_frame.tobytes())
                buf = Gst.Buffer.new_wrapped_bytes(data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1
                
                if self.number_frames & 0x1F == 0:
                    print(f"FPS: {32.0 / (time() - self.last_time)}")
                    self.last_time = time()

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
    def __init__(self, args, interpreter, **properties):
        super(RtspServer, self).__init__(**properties)
        self.set_service(str(args.port))
        self.set_address("0.0.0.0") # use IPv4

        # create a factory and attach it
        self.factory = InferenceDataFactory(args=args, interpreter=interpreter)
        self.factory.set_shared(True)
        self.get_mount_points().add_factory("/stream", self.factory)
        self.attach(None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m",  "--model", type=str, default="yolov8m_humans_quantized.tflite", help="Path to the model file")
    parser.add_argument("-x",  "--width", type=int, default=1280, help="Width of the captured video")
    parser.add_argument("-cx", "--cropped_width", type=int, default=480, help="Width of the cropped video")
    parser.add_argument("-y",  "--height", type=int, default=720, help="Height of the captured video")
    parser.add_argument("-cy", "--cropped_height", type=int, default=640, help="Height of the cropped video")
    parser.add_argument("-f",  "--fps", type=int, default=30, help="Frames per second of the captured video")
    parser.add_argument("-d",  "--device", type=str, default="/dev/video0", help="Video device")
    parser.add_argument("-ac", "--accelerated", action="store_false", help="Use hardware accelerated inference")
    parser.add_argument("-c",  "--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("-sc", "--show_centers", action="store_true", help="Show centers of the detected objects")
    parser.add_argument("-ca", "--crop_average", type=int, default=5, help="Average number of frames used to calculate the center of detected objects")
    parser.add_argument("-sf", "--skip_frames", type=int, default=1, help="Number of frames to skip between inference")
    parser.add_argument("-p",  "--port", type=int, default=25512, help="Port for the RTSP server")
    args = parser.parse_args()

    # setup execution delegate, if empty, uses CPU
    if args.accelerated:
        delegates = [tf.load_delegate("/usr/lib/libvx_delegate.so")]
    else:
        delegates = []

    interpreter = tf.Interpreter(model_path=os.path.join(args.model), experimental_delegates=delegates, num_threads=4)
    interpreter.allocate_tensors()

    sample_input = np.zeros(interpreter.get_input_details()[0]['shape'], dtype=interpreter.get_input_details()[0]['dtype'])
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], sample_input)
    interpreter.invoke()

    print("App initialized, launching RTSP server...")
    
    Gst.init(None)
    server = RtspServer(args, interpreter)
    loop = GLib.MainLoop()
    loop.run()