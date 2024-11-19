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
from time import time
import os
import cv2
import gi
from argparse import ArgumentParser
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib
## Import tflite runtime
import tflite_runtime.interpreter as tf
CUR_PATH = os.path.dirname(__file__)
## Read Environment Variables
USE_HW_ACCELERATED_INFERENCE = os.environ.get("USE_HW_ACCELERATED_INFERENCE")
if not USE_HW_ACCELERATED_INFERENCE:
    USE_HW_ACCELERATED_INFERENCE = 1
MINIMUM_SCORE = os.environ.get("MINIMUM_SCORE")
if not MINIMUM_SCORE:
    MINIMUM_SCORE = 0.55
CAPTURE_RESOLUTION_X = os.environ.get("CAPTURE_RESOLUTION_X")
if not CAPTURE_RESOLUTION_X:
    CAPTURE_RESOLUTION_X = 1920
CAPTURE_RESOLUTION_Y = os.environ.get("CAPTURE_RESOLUTION_Y")
if not CAPTURE_RESOLUTION_Y:
    CAPTURE_RESOLUTION_Y = 1080
CAPTURE_FRAMERATE = os.environ.get("CAPTURE_FRAMERATE")
if not CAPTURE_FRAMERATE:
    CAPTURE_FRAMERATE = 60
DEVICE = os.environ.get("DEVICE")
if not DEVICE:
    DEVICE = "/dev/video0"
STREAM_BITRATE = os.environ.get("STREAM_BITRATE")
if not STREAM_BITRATE:
    STREAM_BITRATE = 0
## Helper function to draw bounding boxes
def draw_bounding_boxes(img,labels,score,x1,x2,y1,y2,object_class):
    # Define some colors to display bounding boxes
    box_colors=[(254,153,143),(253,156,104),(253,157,13),(252,204,26),
             (254,254,51),(178,215,50),(118,200,60),(30,71,87),
             (1,48,178),(59,31,183),(109,1,142),(129,14,64)]
    text_colors=[(0,0,0),(0,0,0),(0,0,0),(0,0,0),
             (0,0,0),(0,0,0),(0,0,0),(255,255,255),
            (255,255,255),(255,255,255),(255,255,255),(255,255,255)]
    cv2.rectangle(img,(x2,y2),(x1,y1),
                box_colors[object_class%len(box_colors)],2)
    
    if object_class < len(labels):
        label = labels[object_class]
    else:
        label = 'unknown'
    cv2.rectangle(img,(x1+len(label)*20,y1+15),(x1,y1),
                box_colors[object_class%len(box_colors)],-1)
    cv2.putText(img,label+' '+score,(x1,y1+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                text_colors[(object_class)%len(text_colors)],1,cv2.LINE_AA)

def put_text(img,y,txt,value):
    cv2.putText(img,txt,(0,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,":",(70,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,"%d" % value,(80,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)

def put_text_ms(img,y,txt,value):
    value=value*1000
    cv2.putText(img,txt,(0,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,":",(70,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,"%.2fms" % value,(80,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)

def put_text_us(img,y,txt,value):
    value=value*1000000
    cv2.putText(img,txt,(0,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,":",(70,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,"%.2fus" % value,(80,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)

t10_=t11=t12=t13=t1_=time()
prev_frame_time = 0
new_frame_time = 0

## Media factory that runs inference
class InferenceDataFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(InferenceDataFactory, self).__init__(**properties)
        # Setup frame counter for timestamps
        self.number_frames = 0
        self.duration = (1.0 / CAPTURE_FRAMERATE) * Gst.SECOND  # duration of a frame in nanoseconds
        # Create opencv Video Capture
        self.cap = cv2.VideoCapture(f'v4l2src device={DEVICE} extra-controls="controls,horizontal_flip=0,vertical_flip=0" ' \
                                    f'! video/x-raw,width={CAPTURE_RESOLUTION_X},height={CAPTURE_RESOLUTION_Y},framerate={CAPTURE_FRAMERATE}/1 ' \
                                    f'! imxvideoconvert_g2d  ' \
                                    f'! video/x-raw,format=BGRA ' \
                                    f'! appsink', cv2.CAP_GSTREAMER)
        # Create factory launch string
        self.launch_string = f'appsrc name=source is-live=true format=GST_FORMAT_TIME ' \
                             f'! video/x-raw,format=BGRA,width={CAPTURE_RESOLUTION_X},height={CAPTURE_RESOLUTION_Y},framerate={CAPTURE_FRAMERATE}/1 ' \
                             f'! vpuenc_h264 bitrate={STREAM_BITRATE} ' \
                             f'! rtph264pay config-interval=1 name=pay0 pt=96 '
        # Setup execution delegate, if empty, uses CPU
        if(USE_HW_ACCELERATED_INFERENCE):
            delegates = [tf.load_delegate("/usr/lib/libvx_delegate.so")]
        else:
            delegates = []
        # Load the Object Detection model and its labels
        with open(os.path.join(MODEL_PATH, LABEL), "r") as file:
            self.labels = file.read().splitlines()
        # Create the tensorflow-lite interpreter
        self.interpreter = tf.Interpreter(model_path=os.path.join(MODEL_PATH, MODEL),
                                          experimental_delegates=delegates, num_threads=4)
        # Allocate tensors.
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size=self.input_details[0]['shape'][1]
        input_details  = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        # If the expected input type is int8 (quantized model), rescale data
        input_type = input_details[0]['dtype']
        if input_type == np.uint8:
            input_scale, input_zero_point = input_details[0]['quantization']

    # Funtion to be ran for every frame that is requested for the stream
    def on_need_data(self, src, length):
        global t10_,t11,t12,t13,t1_
        global prev_frame_time,new_frame_time
        if self.cap.isOpened():
            # Read the image from the camera
            t1=time()
            ret, image_original = self.cap.read()
            new_frame_time = time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            if ret:
                # Resize the image to the size required for inference
                t3=time()
                height1=image_original.shape[0]
                width1=image_original.shape[1]
                image=cv2.resize(image_original,
                                (self.input_size,int(self.input_size*height1/width1)),
                                interpolation=cv2.INTER_NEAREST)
                t4=time()
                height2=image.shape[0]
                scale=height1/height2
                border_top=int((self.input_size-height2)/2)
                image=cv2.copyMakeBorder(image,
                                border_top,
                                self.input_size-height2-border_top,
                                0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
                t5=time()
                # Set the input tensor
                input=np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB)],dtype=np.uint8)
                self.interpreter.set_tensor(self.input_details[0]['index'], input)
                t6=time()
                # Execute the inference
                print("Invoke")
                self.interpreter.invoke()
                t7=time()
                # Check the detected object locations, classes and scores.
                locations = (self.interpreter.get_tensor(self.output_details[0]['index'])[0]*width1).astype(int)
                locations[locations < 0] = 0
                locations[locations > width1] = width1
                classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0].astype(int)
                t8=time()
                scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
                n_detections = self.interpreter.get_tensor(self.output_details[3]['index'])[0].astype(int)
                t9=time()
                # Draw the bounding boxes for the detected objects
                img = image_original
                for i in range(n_detections):
#                    print(self.labels[classes[i]], "{:.0%}".format(scores[i]))
                    if (scores[i]>MINIMUM_SCORE):
                        y1 = locations[i,0]-int(border_top*scale)
                        x1 = locations[i,1]
                        y2 = locations[i,2]-int(border_top*scale)
                        x2 = locations[i,3]
                        draw_bounding_boxes(img,self.labels,"{:.0%}".format(scores[i]),x1,x2,y1,y2,classes[i])
                t10=time()
                # Draw the inference time
                cv2.rectangle(img,(0,0),(180,20),(255,0,0),-1)
                cv2.putText(img,"Inf time: %.6fs" % (t7-t6),(0,15),cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255,255,255),1,cv2.LINE_AA)
                cv2.rectangle(img,(0,20),(180,300),(255,0,0),-1)
                cv2.putText(img,"%2dx%2d@%2d" % (CAPTURE_RESOLUTION_X,CAPTURE_RESOLUTION_Y,fps),(0,35),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
                y=55
                put_text(img, y, 'frames ', self.number_frames); y=y+20
                put_text_ms(img, y, 'capture', (t3-t1)); y=y+20
                put_text_ms(img, y, 'resize',  (t4-t3)); y=y+20
                put_text_ms(img, y, 'copymsk', (t5-t4)); y=y+20
                put_text_ms(img, y, 'set tsr', (t6-t5)); y=y+20
                put_text_ms(img, y, 'invoke',  (t7-t6)); y=y+20
                put_text_ms(img, y, 'get tsr', (t8-t7)); y=y+20
                put_text_ms(img, y, 'tflite',  (t9-t5)); y=y+20
                put_text_ms(img, y, 'draw',    (t10-t9)); y=y+20
                put_text_ms(img, y, 'OSD',     (t11-t10_)); y=y+20
                put_text_ms(img, y, 'buf',     (t12-t11)); y=y+20
                put_text_us(img, y, 'send buf',(t13-t12)); y=y+20
                put_text_ms(img, y, 'Total',   (t13-t1_))
                t10_=t10
                t11=time()
                # Create and setup buffer
                data = GLib.Bytes.new_take(img.tobytes())
                buf = Gst.Buffer.new_wrapped_bytes(data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1
                t12=time()
               # Emit buffer
                retval = src.emit('push-buffer', buf)
                if retval != Gst.FlowReturn.OK:
                    print(retval)
                t13=time()
                t1_=t1

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

class RtspServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(RtspServer, self).__init__(**properties)
        self.set_service("25512")
        self.set_address("0.0.0.0")
        # Create factory
        self.factory = InferenceDataFactory()
        # Set the factory to shared so it supports multiple clients
        self.factory.set_shared(True)
        # Add to "stream" mount point.
        # The stream will be available at rtsp://<board-ip>:8554/stream
        self.get_mount_points().add_factory("/stream", self.factory)
        self.attach(None)

def main():
    global MODEL,MODEL_PATH,LABEL,DEVICE
    global CAPTURE_RESOLUTION_X, CAPTURE_RESOLUTION_Y,CAPTURE_FRAMERATE
    global USE_HW_ACCELERATED_INFERENCE
    parser = ArgumentParser(description='Obeject detection - TensorFlow Lite')
    parser.add_argument('--model_path', '-p', help='Path to model and label files', default='{}/tflite-models'.format(CUR_PATH))
    parser.add_argument('--model', '-m',  help='Name of model file', default='lite-model_ssd_mobilenet_v1_1_metadata_2.tflite')
    parser.add_argument('--label', '-l',  help='Name of label file', default='labelmap.txt')
    parser.add_argument('--device', '-d', help='Video device [/dev/video0]', default='/dev/video0')
    parser.add_argument('--resolution', '-r', help='[1080p] or 720p', default='1080p')
    parser.add_argument('--framerate', '-f', help='Capture framrate [60] or 30', default='60')
    parser.add_argument('--accl', '-a', help='Use accelerator [1] or 0', default='1')
    args = parser.parse_args()
    if args.model_path == None:
        MODEL_PATH = default_model_path
    else:
        MODEL_PATH = args.model_path
    if args.model == None:
        MODEL = default_model
    else:
        MODEL = args.model
    if args.label == None:
        LABEL = default_label
    else:
        LABEL = args.label
    if args.device == None:
        DEVICE = default_label
    else:
        DEVICE = args.device
    if args.resolution == None:
        CAPTURE_RESOLUTION_X=1920
        CAPTURE_RESOLUTION_Y=1080
    if args.resolution == '1080p':
        CAPTURE_RESOLUTION_X=1920
        CAPTURE_RESOLUTION_Y=1080
    if args.resolution == '720p':
        CAPTURE_RESOLUTION_X=1280
        CAPTURE_RESOLUTION_Y=720
    if args.framerate == None:
        CAPTURE_FRAMERATE = 60
    else:
        CAPTURE_FRAMERATE = int(args.framerate)
    print(args.accl)
    if args.accl == None:
        USE_HW_ACCELERATED_INFERENCE=1
    if args.accl == '1':
        USE_HW_ACCELERATED_INFERENCE=1
    if args.accl == '0':
        USE_HW_ACCELERATED_INFERENCE=0
    print(os.path.join(MODEL_PATH, MODEL))
    print(os.path.join(MODEL_PATH, LABEL))
    print(CAPTURE_RESOLUTION_X,'x',CAPTURE_RESOLUTION_Y,'@',CAPTURE_FRAMERATE)
    Gst.init(None)
    server = RtspServer()
    loop = GLib.MainLoop()
    loop.run()

if __name__ == "__main__":
    main()