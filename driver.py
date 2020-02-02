import cv2
import numpy as np
import grabscreen
from directkeys import PressKey, ReleaseKey, W, A, S, D
import time
from pynput.keyboard import Key, Controller
from getkeys import key_check
keyboard = Controller()

RLControlPoint = (320,300)
FWControlPoint = (312,428)

def forward():
    '''
    keyboard.press('w')
    keyboard.release('a')
    keyboard.release('s')
    keyboard.release('d')
    '''
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    #ReleaseKey(W)
    print("Forward")

def right():
    '''
    keyboard.press('d')
    keyboard.release('w')
    keyboard.release('s')
    keyboard.release('d')
    '''
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)

def left():
    '''
    keyboard.press('a')
    keyboard.release('w')
    keyboard.release('s')
    keyboard.release('d')
    '''
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(W)
    ReleaseKey(S)

def backward():
    '''
    keyboard.press('s')
    keyboard.release('w')
    keyboard.release('s')
    keyboard.release('d')
    '''
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

def releaseAll():
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)


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
    else:
        print("Invalid Direction")

    return distance

time.sleep(7)
paused = False
gameLoop = 0
while True:
    img = grabscreen.grab_screen((0,0,640,480))
    blur = cv2.GaussianBlur(img,(5,7),1)
    edge = cv2.Canny(blur,100,120)
    
    if not paused:
        forward()
        #if check(edge,FWControlPoint,"n") < 20:
            #releaseAll()
        if check(edge,FWControlPoint,"n") < 30 or gameLoop % 15 == 0:
            releaseAll()
            backward()
        if check(edge,RLControlPoint,"e") < 50:
            left()
            print("Went Left")
        if check(edge,RLControlPoint,"w") < 50:
            right()
            print("Went Right")
        
        gameLoop += 1

    cv2.imshow("Edges",edge)

    keys = key_check()
    if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()