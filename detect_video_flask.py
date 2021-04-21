#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 17:04:35 2020

@author: danish
"""

import threading
import queue
import cv2
import sys
import time
import numpy as np
import os
import pickle
import os
import argparse
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('\n---------------------- Configuring TF GPU ----------------------\n')
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import cfg, count_objects
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import http.client
import json
import numpy as np
from flask import Flask, request, Response, jsonify, send_from_directory, abort

flask_app = Flask(__name__)
#API configuration
conn = http.client.HTTPSConnection("api.psychicsystems.ai")

parser = argparse.ArgumentParser()
parser.add_argument('--framework', type=str, default='tf', help='framework on which the OD will run either `tf` or `tflite`')
parser.add_argument('--weights', type=str, default='./ckpt/yolov4-416', help='weights for the OD `yolov4-416` for `tf` or `yolov4-custom.tflite` for `tflite`')

""" Setting Flags """
args = parser.parse_args()
FLAGS_TINY=False
FLAGS_SIZE=416
FLAGS_VIDEO=0
FLAGS_FRAMEWORK=args.framework #(frameworks:, 'tf', '(tf, tflite, trt')
FLAGS_WEIGHTS=args.weights
FLAGS_IOU=0.45
FLAGS_SCORE=0.50
FLAGS_COUNT=True
FLAGS_INFO=False
FLAGS_PLATE=False


############## λ->detect start ############## 
print('\n-------------------- Configuring TF Session --------------------\n')
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS_TINY)
input_size = FLAGS_SIZE
video_path = FLAGS_VIDEO

print('\n------------------------ Loading Weights -----------------------\n')
if FLAGS_FRAMEWORK == 'tflite':
    interpreter = tf.lite.Interpreter(model_path=FLAGS_WEIGHTS)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
else:
    saved_model_loaded = tf.saved_model.load(FLAGS_WEIGHTS, 
                                             tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    
    
def get_od(frame):
    api_resp = {}
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #image = Image.fromarray(frame)
    #frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    
    
    if FLAGS_FRAMEWORK == 'tflite':
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        """ from here yolov3 support is removed!"""
        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                        input_shape=tf.constant([input_size, 
                                                                 input_size]))
    else:
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
            
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
    scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
    max_output_size_per_class=50,
    max_total_size=50,
    iou_threshold=FLAGS_IOU,
    score_threshold=FLAGS_SCORE)

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = frame.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())
 
    if FLAGS_COUNT:
        # count objects found
        counted_obj = count_objects(pred_bbox, by_class = False, allowed_classes=allowed_classes)
        # count classes found
        counted_classes = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)
        #adding obj to classes
        for key, value in counted_obj.items():
            counted_classes[key] = value
        # loop through dict and print
        info = ''
        i=1
        for key, value in counted_classes.items():
            api_resp[key] = str(value)
            if len(counted_classes)==i:
                tmp = info
                info = "Number of {0}s: {1}  |  ".format(key, value)
                info += tmp
            else:
                info += "Number of {0}s: {1} |  ".format(key, value)
            i+=1
            #print("Number of {}s: {}-".format(key, value), sep=' ', end='', flush=True)
        image = utils.draw_bbox(frame, pred_bbox, FLAGS_INFO, counted_classes, allowed_classes=allowed_classes, read_plate=FLAGS_PLATE)
    else:
        image = utils.draw_bbox(frame, pred_bbox, FLAGS_INFO, allowed_classes=allowed_classes, read_plate=FLAGS_PLATE)
    return image, info, api_resp

#The queue size for keeping video frame for processing. Cannot less than 2
frame_buffer_size = 5 
#The queue for keeping video frame for processing
frame_buffer = queue.Queue(maxsize=frame_buffer_size)

def get_cam_stream():
    if os.path.exists('session.config'):
        with open('session.config', 'rb') as f:
            config = pickle.load(f)
    else:
        raise FileNotFoundError('Configuration file not found!')
    #cap = cv2.VideoCapture(config['vid_path'])
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        if type(config['vid_path'])==str:
            raise ConnectionRefusedError('Unable to connect to IP cam! possible solutions:\n1. Check your cam if its working. \n2. Verify & renter your credentionals & IP.')
        else:
            raise ConnectionError('Unable to connect to Web/USB cam at index 0, either you don`t have installed Web/USB cam, or its npt working!')
    return cap, config['res']

cap, res = get_cam_stream()

# De-allocate any associated memory usage and exit the program
def deallocateAndExit():
    # De-allocate any associated memory usage
    cap.release()# release camera
    cv2.destroyAllWindows()# release screen
    sys.exit() # exit program

# This is a thread function to keep reading frames and put the frames into frame_buffer for preventing lag of frames reading.
def rtsp_read_buffer():
    # ret will be False when cap.read() timeout or error
    ret = True
    while (ret):
        # If frame_buffer queue is full, get the first queue element out of the queue
        if frame_buffer.full():
            frame_buffer.get()
        # Read frame-by-frame
        # capturing each frame
        ret, buffer_frame = cap.read()
        # Put the capturing frame to the queue
        frame_buffer.put(buffer_frame)
    # Exit program
    deallocateAndExit()
    
def overprint(text, length):
    st = '  '*length
    print(st, end='\r')
    print(text, end="\r")


# Main function to start the program
@flask_app.route('/detection')
def main():
    headers = {'Content-Type': 'application/json'}
    # Start thead functions to continue their task parallelly
    threading.Thread(target=rtsp_read_buffer, daemon=True).start()
    frame_num = 0
    print("\nIntiating OD!\nPress Esc to exit from Object Detection Window!\n")
    length=1
    # Check cv2.VideoCapture(fn) is open
    if cap.isOpened():
        # Check if frame_buffer queue has frames waiting to process or not
        # If some processes are waiting, let calculate it
        if frame_buffer.empty() != True:
            frame_num += 1
            start_time = time.time()
            # Get a frame from the frame_buffer queue
            frame_out = frame_buffer.get()

            frame = frame_out.copy()# output frame
            frame = cv2.resize(frame, res, interpolation=cv2.INTER_AREA)
            #Do processing here
            image, info, api_resp = get_od(frame)
            print(info)
            try:
                return jsonify({"response":api_resp})
            except FileNotFoundError:
                abort(404)

    # Exit program
    
# Initialize Flask application

#start process
if __name__ == '__main__':
    #main()
    flask_app.run(host = '127.0.0.1', port=5000)



