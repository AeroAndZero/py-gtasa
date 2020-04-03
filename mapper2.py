import cv2
import numpy as np
import math

def getDistance(point1,point2):
    dist = math.sqrt( ( (point1[0] - point2[0])*(point1[0] - point2[0]) ) 
                    + ( (point1[1] - point2[1])*(point1[1] - point2[1]) ) )
    return dist

def findPath(image,startPoint = (0,0),endPoint = (0,0),threshold = 10,drawOn = False):
	tempPoint = [startPoint[0],startPoint[1]]
	oldPoint = [0,0]
	pathPoints = []
	pathPointCosts = []

	#Path Finding
	while oldPoint != tempPoint:
		#init
		pathPoints = []
		oldPoint = tempPoint
		pathPointCosts = []

		#Finding Path Points
		for dy in range(-1,2,1):
			for dx in range(-1,2,1):
				if dy == 0 and dx == 0:
					continue

				if (0 < tempPoint[0] + dx < image.shape[1]) and (0 < tempPoint[1] + dy < image.shape[0]):
					if (image[tempPoint[1]+dy,tempPoint[0]+dx][0] <= threshold) and (image[tempPoint[1]+dy,tempPoint[0]+dx][1] <= threshold) and (image[tempPoint[1]+dy,tempPoint[0]+dx][2] <= threshold):
						pathPoints.append([tempPoint[0]+dx,tempPoint[1]+dy])

		#Getting The Route
		if len(pathPoints) != 0:
			for i in range(len(pathPoints)):
				#Actual A* Path Finding Algo
				GCost = getDistance(pathPoints[i],endPoint)
				HCost = getDistance(pathPoints[i],startPoint)
				pathPointCosts.append(GCost + HCost)

			#Assinging/Moving to next Pixel
			minCost = min(pathPointCosts)
			minCostIndex = pathPointCosts.index(minCost)
			tempPoint = pathPoints[minCostIndex]

		#Drawing the Path
		image[tempPoint[1],tempPoint[0]][0] = 255
		image[tempPoint[1],tempPoint[0]][1] = 0
		image[tempPoint[1],tempPoint[0]][2] = 255

	return image

if __name__ == '__main__':
	image = cv2.imread('pathFindingTest.png')
	newImage = findPath(image,startPoint= (150,230),endPoint = (204,180),drawOn=True)
	newImage = cv2.resize(newImage,(500,500))
	cv2.imshow("Mapper 2",newImage)
	cv2.waitKey(0)
	cv2.destroyAllWindows()