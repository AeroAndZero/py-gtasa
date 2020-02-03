import cv2
import numpy as np
import grabscreen
from directkeys import PressKey, ReleaseKey, W, A, S, D, SPACE
import time
from getkeys import key_check
inputKeys = [W,A,S,D,SPACE]

RLControlPoint = (320,340)
FWControlPoint = (320,420)
forwardPress = 0

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

def releaseAll():
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    ReleaseKey(SPACE)

def brake():
    PressKey(SPACE)
    PressKey(S)

def releaseExcept(key):
    for inputKey in inputKeys:
        if key == inputKey:
            pass
        else:
            ReleaseKey(inputKey)

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
    else:
        print("Invalid Direction")

    return distance

time.sleep(7)
paused = True
gameLoop = 0
limit = 10
brakesOn = 15
brakeLength = 10
DiagonalThreshold = 50
RLThreshold = 60

while True:
    img = grabscreen.grab_screen((0,0,640,480))
    h,w = img.shape[0],img.shape[1]
    blur = cv2.GaussianBlur(img,(3,3),1)
    
    central_part = img[0:h-1,RLControlPoint[0]-limit:RLControlPoint[0]+limit]
    part_blur = cv2.GaussianBlur(central_part,(15,15),0)
    
    blur[0:h-1,RLControlPoint[0]-limit:RLControlPoint[0]+limit] = part_blur

    edge = cv2.Canny(blur,100,120)

    if not paused:
        
        if check(edge,FWControlPoint,"n") < 20:
            brake()
            print("Slow Down Brakes!")

        if 0 < check(edge,FWControlPoint,"s") < 30:
            backward()
            print("Reverse")
        elif forwardPress > 5:
            backward()
            forwardPress = 0
        else:
            forward()
            print("Forward")

        if check(edge,RLControlPoint,"ne") < DiagonalThreshold or check(edge,RLControlPoint,"e") < RLThreshold:
            left()
            print("Went Left")
        if check(edge,RLControlPoint,"nw") < DiagonalThreshold or check(edge,RLControlPoint,"w") < RLThreshold:
            right()
            print("Went Right")
        
        gameLoop += 1
    else:
        print("AI Paused")

    cv2.imshow("Edges",edge)

    keys = key_check()
    if 'P' in keys:
            if paused:
                paused = False
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()