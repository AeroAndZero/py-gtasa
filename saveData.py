import cv2
import numpy as np
import grabscreen
from directkeys import PressKey, ReleaseKey, W, A, S, D, SPACE
import time
from getkeys import key_check
import os
inputKeys = [W,A,S,D,SPACE]

#RLControlPoint = (320,340)
RLControlPoint = (320,360)
FWControlPoint = (320,340)

#returns distance between control point and first white pixel is detects
def check(edgeImage,point,direction):
    pointX = point[0]
    pointY = point[1]
    distance = 0

    if direction == "n":
        while np.all(edgeImage[pointY,pointX] <= 200) and 0 <= pointY < edgeImage.shape[0]-1 and 0 <= pointX < edgeImage.shape[1]-1:
            pointY -= 1
            distance += 1

    elif direction == "s":
        while np.all(edgeImage[pointY,pointX] <= 200) and 0 <= pointY < edgeImage.shape[0]-1 and 0 <= pointX < edgeImage.shape[1]-1:
            pointY += 1
            distance += 1

    elif direction == "e":
        while np.all(edgeImage[pointY,pointX] <= 200) and 0 <= pointY < edgeImage.shape[0]-1 and 0 <= pointX < edgeImage.shape[1]-1:
            pointX += 1
            distance += 1

    elif direction == "w":
        while np.all(edgeImage[pointY,pointX] <= 200) and 0 <= pointY < edgeImage.shape[0]-1 and 0 <= pointX < edgeImage.shape[1]-1:
            pointX -= 1
            distance += 1
    elif direction == "ne":
        while np.all(edgeImage[pointY,pointX] <= 200) and 0 <= pointY < edgeImage.shape[0]-1 and 0 <= pointX < edgeImage.shape[1]-1:
            pointX += 1
            pointY -= 1
            distance += 1
    elif direction == "nw":
        while np.all(edgeImage[pointY,pointX] <= 200) and 0 <= pointY < edgeImage.shape[0]-1 and 0 <= pointX < edgeImage.shape[1]-1:
            pointX -= 1
            pointY -= 1
            distance += 1

    elif direction == "nne":
        while np.all(edgeImage[pointY,pointX] <= 200) and 0 <= pointY < edgeImage.shape[0]-1 and 0 <= pointX < edgeImage.shape[1]-1:
            pointX += 2
            pointY -= 1
            distance += 1
    elif direction == "nnw":
        while np.all(edgeImage[pointY,pointX] <= 200) and 0 <= pointY < edgeImage.shape[0]-1 and 0 <= pointX < edgeImage.shape[1]-1:
            pointX -= 1
            pointY -= 2
            distance += 1
    else:
        print("Invalid Direction")

    return distance

#Global variables
time.sleep(5)   #For giving some time before script start
paused = True   #For pausing the game
gameLoop = 0    #keeps track of total processed frames
limit = 10      #for FWcontrolPoint part selection
DiagonalThreshold = 45  #threshold for north-east and north-west direction
RLThreshold = 50    #threshold for east and west direction

#Line Detection Parameters
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 50  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 60  # minimum number of pixels making up a line
max_line_gap = 10  # maximum gap in pixels between connectable line segments

trainingData = []

fileIndex = 1
file_name = 'training_data-{}.npy'.format(fileIndex)
while os.path.isfile(file_name):
    fileIndex += 1
    file_name = 'training_data-{}.npy'.format(fileIndex)

while True:
    #Syntax :
    #[ [ [n,s,e,w],[w,a,s,d,space] ],... ]

    # W = 0, A = 1, S = 3, D = 4, SPACE = 5
    inputKeys = [0,0,0,0,0]

    #Getting gameplay footage
    img = grabscreen.grab_screen((0,0,640,480))
    line_image = np.zeros_like(img)
    h,w = img.shape[0],img.shape[1]
    blur = cv2.GaussianBlur(img,(3,3),1)
    
    #Blurring Center Part More
    central_part = img[0:h-1,RLControlPoint[0]-limit:RLControlPoint[0]+limit]
    part_blur = cv2.GaussianBlur(central_part,(11,11),0)
    blur[0:h-1,RLControlPoint[0]-limit:RLControlPoint[0]+limit] = part_blur

    #Acutal edge detection
    edge = cv2.Canny(blur,100,120)

    #Finding Lines
    lines = cv2.HoughLinesP(edge, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

    #drawing lines on a different image
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255),5)
    
    #Just a handy variable
    processImage = line_image

    #Condition checking
    if not paused:
        #Saving Data
        n_dist = check(processImage,FWControlPoint,"n")
        s_dist = check(processImage,FWControlPoint,"s")
        e_dist = check(processImage,RLControlPoint,"e")
        w_dist = check(processImage,RLControlPoint,"w")
        ne_dist = check(processImage,FWControlPoint,"ne")
        nw_dist = check(processImage,FWControlPoint,"nw")
        keys = key_check()

        for key in keys:
            if key == 'W':
                inputKeys[0] = 1
            elif key == 'S':
                inputKeys[1] = 1
            elif key == 'A':
                inputKeys[2] = 1
            elif key == 'D':
                inputKeys[3] = 1
            elif key == ' ':
                inputKeys[4] = 1
            else:
                pass

        dists = [n_dist,s_dist,e_dist,w_dist,ne_dist,nw_dist]
        print(inputKeys,dists)

        trainingData.append([dists,inputKeys])
        
        if gameLoop % 300 == 0:
            np.save(file_name,trainingData)
            print("----------------------- SAVED -----------------------")
        
        gameLoop += 1
    else:
        #Not Recording
        print("Not recording")

    #Showing Output
    cv2.circle(processImage,RLControlPoint,2,(0,0,255),3)
    cv2.circle(processImage,FWControlPoint,2,(0,255,0),3)
    cv2.imshow("Lines",processImage)

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

    #Quiting opencv
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Destroooyyyyyyy
cv2.destroyAllWindows()