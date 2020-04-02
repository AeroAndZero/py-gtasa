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
from grabscreen import grab_screen
from driver import forward, right, left, backward, brake, releaseAll, releaseExcept
from directkeys import PressKey, ReleaseKey, W, A, S, D, SPACE
from getkeys import key_check
import time

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'FrontView2_color.mp4'

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

#Some Variables
    #Thresholds
boxThreshold = 0.75
carAreaThreshold = 15000
bikeAreaThreshold = 5000
personAreaThreshold = 5000
    #Imp For Objects
objectCords = []
objectClass = []
closeObjects = []
    #Some Other Thresholds
firstXLine = 246 #2/5 of the width + 10
secondXLine = 416   #3/5 of the width - 10
leftLimit = 84 #1/3 of firstXLine
rightLimit = 554 #2/3 of secondXLine
reqTurn = []
    #Pausing
paused = True
    #Game Related Settings
forwardPress = 0    # keeps track of forward key press
brakesOnFrame = 7  # Presses brake on every 15th frame

def getArea(cords,imageWidth,imageHeight):
    #[ymin, xmin, ymax, xmax]
    width = (cords[3] - cords[1]) * imageWidth
    height = (cords[2] - cords[0]) * imageHeight
    area = width * height
    return area

def getMaxCount(a):
    maxCount = []
    found = 0

    #Counting Number's Occurance
    for i in range(len(a)):
        for j in range(len(maxCount)):
            if maxCount[j][0] == a[i]:
                maxCount[j][1] += 1
                found = 1
        if found == 0:
            maxCount.append([a[i],1])

    #dont mind me im just debugging
    #print("getMaxCount : ",maxCount)
    #print("actual array : ",a)

    #Getting Max Occurance
    maxNumber = 0
    maxNumberIndex = 0
    for i in range(len(maxCount)):
        if maxNumber < maxCount[i][0]:
            maxNumber = maxCount[i][0]
            maxNumberIndex = i

    if len(maxCount) != 0:
        return maxCount[maxNumberIndex][0]
    else:
        return -1

while True:
    #Init
    objectCords = []
    objectClass = []
    closeObjects = []
    reqTurn = []

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = grab_screen((0,0,640,480))
    frame = cv2.resize(frame,(640,480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)
    frame = cv2.cvtColor(frame_rgb,cv2.COLOR_RGB2BGR)

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
        min_score_thresh=boxThreshold)

    #Getting Box Area
    final_score = np.squeeze(scores)
    final_boxes = np.squeeze(boxes)
    final_classes = np.squeeze(classes).astype(np.int32)
    for i in range(len(final_score)):
        if final_score[i] > boxThreshold:
            objectCords.append(final_boxes[i])
            objectClass.append(final_classes[i])

    # Calculating Area
    # objectCords and objectClass are assigned parallely
    for i in range(len(objectCords)):
        objectArea = getArea(objectCords[i],frame.shape[1],frame.shape[0])
        #print("Area",objectArea)
        if objectClass[i] == 1 and objectArea > carAreaThreshold:
            closeObjects.append(objectCords[i])
        elif objectClass[i] == 2 and objectArea > bikeAreaThreshold:
            closeObjects.append(objectCords[i])
        elif objectClass[i] == 3 and objectArea > personAreaThreshold:
            closeObjects.append(objectCords[i])
        else:
            pass

    #Decision making on cords of the object
    #boxes syntax = [ymin, xmin, ymax, xmax]
    for i in range(len(closeObjects)):
        objectMidX = int((closeObjects[i][1] + closeObjects[i][3]) / 2 * frame.shape[1])
        cv2.circle(frame,(objectMidX,0),2,(255,0,0),3)
        if  leftLimit < objectMidX <= firstXLine:
            reqTurn.append(0)
        elif firstXLine <= objectMidX <= secondXLine:
            reqTurn.append(1)
        elif secondXLine <= objectMidX < rightLimit:
            reqTurn.append(2)
        else:
            pass

    #Requested Turns :
    #   0 = Turn Right         1 = Stop/Slow Down/Reverse        2 = Turn Left
    #       (Majority Object       (Majority objects in              ( Majority objects in
    #       On Left)                Middle)                            Right )

    #Taking turns
    if not paused:
        whichDirection = getMaxCount(reqTurn)
        if whichDirection == 0:
            print("Majority : Right")
            ReleaseKey(A)
            PressKey(D)
            PressKey(W)
        elif whichDirection == 1:
            print("Majority : Brake")
            ReleaseKey(W)
            PressKey(S)
        elif whichDirection == 2:
            print("Majority : Left")
            ReleaseKey(D)
            PressKey(A)
            PressKey(W)
        else:
            #Forward
            print("Nothing Detected. Pressing Forward")
            ReleaseKey(S)
            PressKey(W)
            forwardPress += 1

        #Slowing Down Constantly
        if forwardPress % brakesOnFrame == 0:
            releaseAll()
            PressKey(S)
    
    else:
        print("Paused")

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