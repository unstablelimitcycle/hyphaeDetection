import cv2
import numpy as np
import math
import pandas as pd
from os import listdir
from os.path import isfile, join
from hyphaeDetectionFunctions import load_images_from_folder


#Designate folder to open
folder_to_open = '/Users/lisarogers/PythonStuff/testImages/Images/'

#Designate path to write files to 
path = '/Users/lisarogers/PythonStuff/testImages/'

#Get file name to use as string in CSV columns
filenames = listdir(folder_to_open)
filenames.sort()
stringsforCSV = [f for f in filenames if f.endswith('.jpg')]

#Load Images from designated folder
Images = load_images_from_folder(folder_to_open)
Images.sort()
imagesLen = len(Images)

radii = [250, 500]
side = [500, 1000]

#Initialize arrays for storing counted and computed areas
countedArea = np.empty(imagesLen)
difference = np.empty(imagesLen)

for n in range(0,imagesLen):

    img_gray = cv2.cvtColor(Images[n], cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
    # cv2.imwrite('inverted.jpg', thresh)

    #Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt0 = contours[0]
    
    # Create all-zero arrays for storing drawn contours
    img_contours = np.zeros(thresh.shape)

    # Draw Contours
    cv2.drawContours(img_contours, contours, 0, (0,255,0), 1)
    #cv2.imwrite('contours.jpg', img_contours)

    #Find OpenCV counted Area
    countedArea[n] = cv2.contourArea(cnt0)

#Compute areas of all shapes
circleComputedArea = [radii[0]*radii[0]*math.pi, radii[1]*radii[1]*math.pi]
squareComputedArea = [side[0]*side[0], side[1]*side[1]]
triangleComputedArea = [side[0]*side[0]*math.sqrt(3)/4, side[1]*side[1]*math.sqrt(3)/4]
computedArea = [circleComputedArea[1], squareComputedArea[1], triangleComputedArea[1], circleComputedArea[0], squareComputedArea[0], triangleComputedArea[0]]

for n in range(0,imagesLen):
    difference[n] = abs(computedArea[n]-countedArea[n])

#Create DataFrame for csv files
data = {'Image Name': stringsforCSV,'Counted Area': countedArea, 'Computed Area': computedArea,'Difference': difference }

#Read data into a CSV 
df = pd.DataFrame(data, columns = ['Image Name','Counted Area', 'Computed Area', 'Difference'])
df.to_csv('OpenCV Area Computing Accuracy Test.csv')
