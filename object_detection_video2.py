######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a video.
# It draws boxes, scores, and labels around the objects of interest in each
# frame of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
from grabscreen import grab_screen
from directkeys import PressKey, ReleaseKey, W, A, S, D, SPACE
from getkeys import key_check

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'test5.mov'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 3

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#paused or not
paused = True

while True:

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = grab_screen((0,0,640,480))
    objects = []
    class_str = ""
    frame_width = frame.shape[0]
    frame_height = frame.shape[1]
    
    
    rows, cols = frame.shape[:2]
    
    left_boundary = [int(cols*0.40), int(rows*0.95)]
    left_boundary_top = [int(cols*0.40), int(rows*0.20)]
    right_boundary = [int(cols*0.60), int(rows*0.95)]
    right_boundary_top = [int(cols*0.60), int(rows*0.20)]
    bottom_left  = [int(cols*0.20), int(rows*0.95)]
    top_left     = [int(cols*0.20), int(rows*0.20)]
    bottom_right = [int(cols*0.80), int(rows*0.95)]
    top_right    = [int(cols*0.80), int(rows*0.20)]
    
    
    cv2.line(frame,tuple(bottom_left),tuple(bottom_right), (255, 0, 0), 5)
    cv2.line(frame,tuple(bottom_right),tuple(top_right), (255, 0, 0), 5)
    cv2.line(frame,tuple(top_left),tuple(bottom_left), (255, 0, 0), 5)
    cv2.line(frame,tuple(top_left),tuple(top_right), (255, 0, 0), 5)
    cv2.line(frame,tuple(left_boundary),tuple(left_boundary_top), (255, 0, 0), 5)
    cv2.line(frame,tuple(right_boundary),tuple(right_boundary_top), (255, 0, 0), 5)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)
    print(frame_width,frame_height)
    ymin = int((boxes[0][0][0]*frame_width))
    xmin = int((boxes[0][0][1]*frame_height))
    ymax = int((boxes[0][0][2]*frame_width))
    xmax = int((boxes[0][0][3]*frame_height))
    Result = np.array(frame[ymin:ymax,xmin:xmax])

    ymin_str='y min  = %.2f '%(ymin)
    ymax_str='y max  = %.2f '%(ymax)
    xmin_str='x min  = %.2f '%(xmin)
    xmax_str='x max  = %.2f '%(xmax)

    cv2.putText(frame,ymin_str, (50, 50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    cv2.putText(frame,ymax_str, (50, 70),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    cv2.putText(frame,xmin_str, (50, 90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    cv2.putText(frame,xmax_str, (50, 110),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    print(scores.max())
     
    print("left_boundary[0],right_boundary[0] :", left_boundary[0], right_boundary[0])
    print("left_boundary[1],right_boundary[1] :", left_boundary[1], right_boundary[1])
    print("xmin, xmax :", xmin, xmax)
    print("ymin, ymax :", ymin, ymax)

    if not paused:
        if scores.max() > 0.78:
            print("inif")
        if(xmin >= left_boundary[0]):
            print("move LEFT - 1st !!!")
            PressKey(A)
            ReleaseKey(D)
            cv2.putText(frame,'Move LEFT!', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
        elif(xmax <= right_boundary[0]):
            print("move Right - 2nd !!!")
            PressKey(D)
            ReleaseKey(A)
            cv2.putText(frame,'Move RIGHT!', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
        elif(xmin <= left_boundary[0] and xmax >= right_boundary[0]):
            print("STOPPPPPP !!!! - 3nd !!!")
            PressKey(S)
            ReleaseKey(S)
            ReleaseKey(A)
            ReleaseKey(D)
            cv2.putText(frame,' STOPPPPPP!!!', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)

    else:
        cv2.putText(frame,'Paused', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
        print("paused")

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    #Pausing is important
    keys = key_check()
    if 'P' in keys:
        if paused:
            paused = False
            time.sleep(1)
        else:
            paused = True
            ReleaseKey(A)
            ReleaseKey(W)
            ReleaseKey(D)
            time.sleep(1)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()