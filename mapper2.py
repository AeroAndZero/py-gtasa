import cv2
import numpy as np
import math

def getDistance(point1,point2):
	dist = math.sqrt( ( (point1[0] - point2[0])*(point1[0] - point2[0]) ) 
					+ ( (point1[1] - point2[1])*(point1[1] - point2[1]) ) )
	return dist

def findPath(image,startPoint = (0,0),endPoint = (0,0),threshold = 100,drawOn = False):
	tempPoint = [startPoint[0],startPoint[1]]
	destination = [endPoint[0],endPoint[1]]
	oldPoint = [0,0]
	pathPoints = []
	pathPointCosts = []
	history = []

	#Path Finding
	while tempPoint != destination:
		#init
		pathPoints = []
		history.append(tempPoint)
		pathPointFCosts = []

		#Finding Path Points
		for dy in range(-1,2,1):
			for dx in range(-1,2,1):
				if dy == 0 and dx == 0:
					continue

				if (0 < tempPoint[0] + dx < image.shape[1]) and (0 < tempPoint[1] + dy < image.shape[0]):
					if (image[tempPoint[1]+dy,tempPoint[0]+dx][0] <= threshold) and (image[tempPoint[1]+dy,tempPoint[0]+dx][1] <= threshold) and (image[tempPoint[1]+dy,tempPoint[0]+dx][2] <= threshold):
						pathPoints.append([tempPoint[0]+dx,tempPoint[1]+dy])

		for point in pathPoints:
			if point in history:
				pathPoints.remove(point)
				print("Something Removed")

		#Getting The Route
		if len(pathPoints) != 0:
			for i in range(len(pathPoints)):
				#Actual A* Path Finding Algo
				GCost = getDistance(pathPoints[i],tempPoint)
				HCost = getDistance(pathPoints[i],endPoint)
				pathPointFCosts.append(HCost + GCost)

			#Getting Minimum FCost And Moving on that pixel
			minCost = min(pathPointFCosts)
			minCostIndex = pathPointFCosts.index(minCost)
			tempPoint = pathPoints[minCostIndex]
		else:
			break

		#Drawing the Path
		image[tempPoint[1],tempPoint[0]][0] = 255
		image[tempPoint[1],tempPoint[0]][1] = 0
		image[tempPoint[1],tempPoint[0]][2] = 255

	return image

if __name__ == '__main__':
	image = cv2.imread('pathFindingTest.png')
	#image = cv2.resize(image,(100,100))
	#startPoint = (23,56)
	#endPoint = (78,74)
	startPoint = (190,216)
	endPoint = (291,279)
	image = findPath(image,startPoint,endPoint)
	image = cv2.resize(image,(500,500))
	cv2.imshow("Mapper 3",image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()