import cv2
import numpy as np
import grabscreen
from directkeys import PressKey, ReleaseKey, W, A, S, D, SPACE
import time
from getkeys import key_check
inputKeys = [W,A,S,D,SPACE]

#RLControlPoint = (320,340)
RLControlPoint = (320,360)
FWControlPoint = (320,340)
forwardPress = 0
#Map Pixels :
#   Top-left = 32,360
#   bottom-right = 142,459

#Basic movement operations
def forward():
    global forwardPress
    releaseExcept(W)
    PressKey(W)
    forwardPress += 1

def right():
    ReleaseKey(A)
    PressKey(D)

def left():
    ReleaseKey(D)
    PressKey(A)

def backward():
    releaseExcept(S)
    PressKey(S)

def brake():
    PressKey(SPACE)
    PressKey(S)

def releaseAll():
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    ReleaseKey(SPACE)

def releaseExcept(key):
    for inputKey in inputKeys:
        if key == inputKey:
            pass
        else:
            ReleaseKey(inputKey)

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

    return distance,pointX,pointX

def drawTemplateOutline(res,drawOnImage):
    h,w = drawOnImage.shape[0],drawOnImage.shape[1]
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(drawOnImage, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
    return drawOnImage

#Global variables
time.sleep(3)   #For giving some time before script start
paused = True   #For pausing the game
gameLoop = 0    #keeps track of total processed frames
limit = 10      #for FWcontrolPoint part selection
DiagonalThreshold = 45  #threshold for north-east and north-west direction
RLThreshold = 50    #threshold for east and west direction

#Line Detection Parameters
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 30  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 20  # minimum number of pixels making up a line
max_line_gap = 2  # maximum gap in pixels between connectable line segments

targetIcon = cv2.imread('mapIcon.png',cv2.IMREAD_GRAYSCALE)
th,tw = 10,10
templateThreshold = 0.8

while True:
    #Resetting control points
    pointX,pointY = RLControlPoint

    #Getting gameplay footage
    img = grabscreen.grab_screen((0,0,640,480))

    #Extracting Map
    gameMap_org = img[360:459,32:142]
    gameMap_gray = cv2.cvtColor(gameMap_org,cv2.COLOR_BGR2GRAY)
    
    #Template Matching Target
    targetMatch = cv2.matchTemplate(gameMap_gray,targetIcon,cv2.TM_CCOEFF_NORMED)
    loc = np.where( targetMatch >= templateThreshold)   #Drawing Target
    for pt in zip(*loc[::-1]):
        cv2.rectangle(gameMap_org, pt, (pt[0] + tw, pt[1] + th), (0,255,255), 2)

    #Condition checking
    if not paused:
        #Not Paused
        #print("Running..")
        #Keep track of gameloop
        gameLoop += 1
    else:
        #For proof that I am not driving
        #print("AI Paused")
        pass

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

    cv2.circle(gameMap_org,(int(gameMap_org.shape[1]/2),int(gameMap_org.shape[0]/2)),1,(0,0,255),2)
    gameMap = cv2.resize(gameMap_org,(500,500))
    cv2.imshow("Map",gameMap)

    #Quiting opencv
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Destroooyyyyyyy
cv2.destroyAllWindows()