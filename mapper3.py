import cv2
import numpy as np
import math

image = 0

#returns distance between two points
def getDistance(point1,point2):
    dist = math.sqrt( ( (point1[0] - point2[0])*(point1[0] - point2[0]) ) 
                    + ( (point1[1] - point2[1])*(point1[1] - point2[1]) ) )
    return dist

#Scans the pixels around a specific pixel
def getSurroundingNodes(image,node):
    threshold=[50,50,50]
    SurroundingNodes = []
    pixel = node.position
    for dy in range(-1,2,1):
        for dx in range(-1,2,1):
            if (dy == 0 and dx == 0) or (not((0 < pixel[0]+dx < image.shape[1]) and (0 < pixel[1]+dy < image.shape[0]))):
                continue
            if (image[pixel[1]+dy,pixel[0]+dx][0] <= threshold[0]) and (image[pixel[1]+dy,pixel[0]+dx][1] <= threshold[1]) and (image[pixel[1]+dy,pixel[0]+dx][2] <= threshold[2]):
                SurroundingNodes.append(Node(parent=node,position=[pixel[0]+dx,pixel[1]+dy]))

    return SurroundingNodes

class Node():
    def __init__(self, parent=None, position=None):
        global image
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 99999
        self.f = 0
        self.child = []

    def changeParent(self, other):
        return self.parent == other.parent

def sortNodes(nodes):
    for i in range(len(nodes)):
        minFCost = nodes[i].f
        minFCostIndex = i

        for j in range(i,len(nodes)):
            if (nodes[j].f) < minFCost:
                minFCost = nodes[j].f
                minFCostIndex = j
        tempNode = nodes[i]
        nodes[i] = nodes[minFCostIndex]
        nodes[minFCostIndex] = tempNode

    return nodes

def getMinF(nodes):
    tempF = []
    for node in nodes:
        tempF.append(node.f)

    minF = min(tempF)
    minFIndex = tempF.index(minF)
    return nodes[minFIndex]

def findPath(image,startPoint=(0,0),endPoint=(0,0)):
    pathFound = False

    startNode = Node(parent=None,position=startPoint)
    startNode.g = 0
    startNode.h = getDistance(startNode.position,endPoint)
    startNode.f = startNode.g + startNode.h

    openList = [startNode]
    closedList = []

    while len(openList) != 0 and not pathFound:
        q = getMinF(openList)
        
        '''
        image[q.position[1],q.position[0]][0] = 0
        image[q.position[1],q.position[0]][1] = 255
        image[q.position[1],q.position[0]][2] = 0
        '''

        q.child = getSurroundingNodes(image,q)

        for successor in q.child:
            skipThis = False
            if successor.position[0] == endPoint[0] and successor.position[1] == endPoint[1]:
                #print("Path Found!")
                pathFound = True
                lastSuccessor = successor
            else:
                pass
                #print("Path Not Found, Researching")

            successor.g = q.g + getDistance(successor.position,q.position)
            successor.h = getDistance(successor.position,endPoint)
            successor.f = successor.g + successor.h

            if successor in openList:
                if openList[openList.index(successor)].f < successor.f:
                    continue
            
            if successor in closedList:
                if closedList[closedList.index(successor)].f < successor.f:
                    continue
            
            openList.append(successor)

        openList.remove(q)
        closedList.append(q)

    if pathFound:
        while lastSuccessor.parent != None:
            image[lastSuccessor.position[1],lastSuccessor.position[0]][0] = 255
            image[lastSuccessor.position[1],lastSuccessor.position[0]][1] = 0
            image[lastSuccessor.position[1],lastSuccessor.position[0]][2] = 255
            lastSuccessor = lastSuccessor.parent

    return image

if __name__ == '__main__':
    image = cv2.imread('pathFindingTest.png')
    image = cv2.resize(image,(100,100))
    #startPoint = (23,56)
    #endPoint = (78,74)
    startPoint = (43,33)
    endPoint = (51,58)
    image = findPath(image,startPoint,endPoint)
    image = cv2.resize(image,(500,500))
    cv2.imshow("Mapper 3",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()