import cv2
import numpy as np
import grabscreen
from directkeys import PressKey, ReleaseKey, W, A, S, D, SPACE
import time
from getkeys import key_check
import mapper

inputKeys = [W,A,S,D,SPACE]

#RLControlPoint = (320,340)
RLControlPoint = (320,360)
FWControlPoint = (320,340)
forwardPress = 0

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
threshold = 50  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 60  # minimum number of pixels making up a line
max_line_gap = 10  # maximum gap in pixels between connectable line segments

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
    releaseExcept(SPACE)
    PressKey(SPACE)

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

def getLines(edge,rho = 1,theta = np.pi/180,threshold = 50,min_line_length=60,max_line_gap=10):
    line_image = np.zeros_like(edge)

    #Finding Lines
    lines = cv2.HoughLinesP(edge, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

    #drawing lines on a different image
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255),5)

    return line_image

def keepInLane(processImage):
    global FWControlPoint,RLControlPoint,DiagonalThreshold,RLThreshold,forwardPress
    
    #Resetting control points
    pointX,pointY = RLControlPoint

    if check(processImage,FWControlPoint,"n")[0] < 30:
        brake()
        print("Brakes!")

    #If car crashes, take reverse
    if 0 < check(processImage,FWControlPoint,"s")[0] < 30: #or check(processImage,FWControlPoint,"n")[0] < 20:
        backward()
        print("Reverse")

    #For Slowing down
    elif forwardPress > 5:
        backward()
        forwardPress = 0

    #Forward obviously
    else:
        forward()
        print("Forward")

    #Condition for turning left
    if 10 < check(processImage,RLControlPoint,"ne")[0] < DiagonalThreshold or 10 < check(processImage,RLControlPoint,"e")[0] < RLThreshold:
        pointX,pointY = check(processImage,RLControlPoint,"ne")[1],check(processImage,RLControlPoint,"ne")[2]
        left()
        print("Left")
        
        #for output display
        cv2.line(processImage,(pointX,pointY),RLControlPoint,(0,0,255),2)
    
    #Condition for turning right
    if 10 < check(processImage,RLControlPoint,"nw")[0] < DiagonalThreshold or 10 < check(processImage,RLControlPoint,"w")[0] < RLThreshold:
        pointX,pointY = check(processImage,RLControlPoint,"nw")[1],check(processImage,RLControlPoint,"nw")[2]
        right()
        print("Right")
        
        #for output display
        cv2.line(processImage,(pointX,pointY),RLControlPoint,(0,0,255),2)

    return processImage

while True:

    #Getting gameplay footage
    img = grabscreen.grab_screen((0,0,640,480))

    h,w = img.shape[0],img.shape[1]
    blur = cv2.GaussianBlur(img,(3,3),1)

    #Acutal edge detection
    #edge = cv2.Canny(blur,100,120)     ----------
    
    #Just a handy variable
    processImage = np.zeros_like(img)#getLines(edge)        ----------

    #Condition checking
    if not paused:
        #Keeps in lane
        #processImage = keepInLane(processImage)    ----------

        #Keep track of gameloop
        gameLoop += 1
    else:
        #For proof that I am not driving
        print("AI Paused")
    
    try:
        #Getting map
        gameMap_org = mapper.getMap(img)
        gameMap,targetPoint = mapper.findTarget(gameMap_org)

        #Showing Output
        #processImage = cv2.cvtColor(processImage,cv2.COLOR_GRAY2BGR)   ----------
        cv2.circle(processImage,RLControlPoint,2,(0,0,255),3)
        cv2.circle(processImage,FWControlPoint,2,(0,0,255),3)
        cv2.imshow("Lines",processImage)

        #Drawing Path
        mapCenter = (int(gameMap_org.shape[1]/2),int(gameMap_org.shape[0]/2))
        gameMap_org = mapper.findPath(gameMap_org,mapCenter,targetPoint,threshold=10,drawOn=True)

        #Output map
        cv2.circle(gameMap_org,targetPoint,1,(255,0,255),0)
        
        gameMap_resized = cv2.resize(gameMap_org,(500,500))
        cv2.imshow("Game map",gameMap_resized)
    except Exception as e:
        print(e)

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