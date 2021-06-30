import cv2
import numpy as np
import math

img = cv2.imread('500px_trangle.jpg')

radii = [250, 500]
base = 440
#865
height = 370
#375

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
# cv2.imwrite('inverted.jpg', thresh)

#Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt0 = contours[0]
    
# Create all-zero arrays for storing drawn contours
#img_contours = np.zeros(thresh.shape)
cv2.drawContours(img, contours, -1, (0,255,0), 2)
cv2.imwrite('contours.jpg', img)
#Find OpenCV counted Area
countedArea = cv2.contourArea(cnt0)

#Compute areas of all shapes
# = radii[1]*radii[1]*math.pi 
#radii[1]*radii[1]*math.pi
#squareComputedArea = side[0]*side[0]
#=side[0]*side[0]
computedArea = base*height/2
#side[1]*side[1]*math.sqrt(3)/4

diff = abs(countedArea-computedArea)
perDiff = 100*diff/computedArea
print(countedArea)
print(computedArea)
print(diff)
print(perDiff)