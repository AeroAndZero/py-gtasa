import cv2
import numpy as np
import math

targetIcon = cv2.imread('mapIcon.png',cv2.IMREAD_GRAYSCALE)
th,tw = 10,10
templateThreshold = 0.8

#Center of target Icon = (4,5)

def findTarget(image,targetIcon = targetIcon,threshold = 0.8,draw = False):
    targetPoint = (0,0)
    image_org = image
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(image,targetIcon,cv2.TM_CCOEFF_NORMED)
    ch,cw = 5,4
    loc = np.where( res >= threshold)
    
    for pt in zip(*loc[::-1]):
        targetPoint = (pt[0] + cw, pt[1] + ch)
        if draw:
            cv2.circle(image_org, (pt[0] + cw, pt[1] + ch), 1,(0,255,255), 1)
    
    return image_org,targetPoint

def getMap(screenshot):
    #Extracting Map
    gameMap = screenshot[360:459,32:142]
    return gameMap

def getDistance(point1,point2):
    dist = math.sqrt( ( (point1[0] - point2[0])*(point1[0] - point2[0]) ) 
                    + ( (point1[1] - point2[1])*(point1[1] - point2[1]) ) )
    return dist

def getClosestPoint(points = [],refPoint = ()):
    minDist = getDistance(points[0],refPoint)
    minDistIndex = 0
    for index,point in enumerate(points):
        if minDist > getDistance(point,refPoint):
            minDist = getDistance(point,refPoint)
            minDistIndex = index

    return points[minDistIndex]

def findPath(image,startPoint = (0,0),endPoint = (0,0),threshold = 10,drawOn = False):
    currentPoint = startPoint
    lastPoint = (0,0)
    path = []
    h,w = image.shape[0],image.shape[1]
    gridRange = 1

    while 0 <= currentPoint[0] < w and 0 <= currentPoint[1] < h and currentPoint != endPoint:
        points = [(0,0)]
        lastPoint = currentPoint

        for dy in range(-1,2,1):
            for dx in range(-1,2,1):
                if dy == 0 and dx == 0:
                    pass
                if image[currentPoint[1]+dy,currentPoint[0]+dx][0] <= threshold and image[currentPoint[1]+dy,currentPoint[0]+dx][1] <= threshold and image[currentPoint[1]+dy,currentPoint[0]+dx][2] <= threshold:
                    points.append((currentPoint[0]+dx,currentPoint[1]+dy))
                    print("New Point attached")
                
        currentPoint = getClosestPoint(points,endPoint)
        path.append(currentPoint)
        
    for point in path:
        print(point)
        cv2.circle(image,point,1,(255,0,255),2)

    return image
        
if __name__ == "__main__":
    image = cv2.imread('pathFindingTest.png')
    newImage = findPath(image,startPoint= (99,41),endPoint = (108,67),drawOn=True)
    cv2.imshow("what",newImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()