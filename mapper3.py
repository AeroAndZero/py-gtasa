import cv2
import numpy as np
import math

#returns distance between two points
def getDistance(point1,point2):
    dist = math.sqrt( ( (point1[0] - point2[0])*(point1[0] - point2[0]) ) 
                    + ( (point1[1] - point2[1])*(point1[1] - point2[1]) ) )
    return dist

#Scans the pixels around a specific pixel
def getPathPoints(image,pixel,threshold=[10,10,10]):
    pathPoints = []
    for dy in range(-1,2,1):
        for dx in range(-1,2,1):
            if (dy == 0 and dx == 0) or (not((0 < pixel[0]+dx < image.shape[1]) and (0 < pixel[1]+dy < image.shape[0]))):
                continue
            if (image[pixel[1]+dy,pixel[0]+dx][0] <= threshold[0]) and (image[pixel[1]+dy,pixel[0]+dx][1] <= threshold[1]) and (image[pixel[1]+dy,pixel[0]+dx][2] <= threshold[2]):
                pathPoints.append([pixel[0]+dx,pixel[1]+dy])

    return pathPoints

class Node():
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0

    def changeParent(self, other):
        return self.parent == other.parent

def sortNodes(nodes):
    for i in range(len(nodes)):
        minFCost = nodes[i].g + nodes[i].h
        minFCostIndex = i

        for j in range(i,len(nodes)):
            if (nodes[j].g + nodes[j].h) < minFCost:
                minFCost = nodes[j].g + nodes[j].h
                minFCostIndex = j
        tempNode = nodes[i]
        nodes[i] = nodes[minFCostIndex]
        nodes[minFCostIndex] = tempNode

    return nodes

def findPath(image,currentNode=Node(parent=None,position=(0,0)),endPoint=(0,0)):
    #Setup
    currentNode.g = getDistance(currentNode.position,currentNode.parent.position)
    currentNode.h = getDistance(currentNode.position,endPoint)
    children = []

    if currentNode.position == endPoint:
        print("Path Found")
        return 0
    else:
        print("Path Not Found yet. Searching again")

    pathPoints = getPathPoints(image,currentNode.position)

    for i in range(len(pathPoints)):
        children.append(Node(parent = currentNode,position=pathPoints[i]))
        children[i].g = getDistance(currentNode.position,children[i].position) + currentNode.g
        children[i].h = getDistance(children[i].position,endPoint)

    if len(children) > 0:
        children = sortNodes(children)
        findPath(image,currentNode=children[i],endPoint=endPoint)
    else:
        print("No Childrens")
        return


if __name__ == '__main__':
    '''
    nodes = []
    for i in range(10,0,-1):
        nodes.append(Node(parent=None,position=None))
        nodes[10-i].g = i
        nodes[10-i].h = i

    newNodes = sortNodes(nodes)
    for node in newNodes:
        print(node.g + node.h)
    '''
    image = cv2.imread('pathFindingTest.png')
    startNode = Node(parent=None,position=(103,40))
    endPoint = (103,41)
    newImage = findPath(image,currentNode=Node(parent=startNode,position=startNode.position),endPoint = endPoint)
    '''
    newImage = cv2.resize(newImage,(500,500))
    cv2.imshow("Mapper 2",newImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''