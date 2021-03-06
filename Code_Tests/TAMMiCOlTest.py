import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib import pyplot as plt
from hyphaeDetectionFunctions import load_images_from_folder
import pandas as pd

#Designate folder to open
folder_to_open = '/Users/lisarogers/PythonStuff/TAMMiCol/TAMMiCol Test Data/'

#Designate path to write files to 
path = '/Users/lisarogers/PythonStuff/TAMMiCol/'

# #Get file name to use as string in CSV columns
filenames = listdir(folder_to_open)
stringsforCSV = [f for f in filenames if f.endswith('.tif')]

#Load Images from designated folder
Images = load_images_from_folder(folder_to_open)

#Initalize Array: Contour Areas and Growth Index 
area1 = np.empty(len(Images), dtype=object)
hyphae = np.empty(len(Images), dtype=object)
growthIndex = np.empty(len(Images), dtype=object)

#Find interior contour based on colony at 73hrs
intContImage = cv2.cvtColor(Images[0], cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(intContImage,(5,5), 0)
_, thresh_img = cv2.threshold(img_blur, 125, 255, cv2.THRESH_BINARY)
kernal = np.ones((5, 5), np.uint8)
closed_image = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernal)
closed_image_file = 'initialColony.jpg'
cv2.imwrite(join(path, closed_image_file), closed_image)

cont, hier = cv2.findContours(closed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contNum = len(cont)
contLength = np.empty(contNum)

for i in range(0,contNum):
    contLength[i] = len(cont[i])

max2Contour = np.amax(contLength)
posMax2Contour = np.argmax(contLength)
cnt2 = cont[posMax2Contour]
area2 = cv2.contourArea(cnt2)

for n in range(0, len(Images)):
   
    # Convert all images to greyscale
    images = cv2.cvtColor(Images[n], cv2.COLOR_BGR2GRAY)
   
    # Blur all to eliminate noise
    blur_images = cv2.GaussianBlur(images,(5,5), 0)

    # Threshhold images and save images
    #Inverted binary thresh for TAMMiCol images
    _, thresh_img = cv2.threshold(blur_images, 125, 255, cv2.THRESH_BINARY)
    thresh_images = f'threshold_images{n}.jpg'
    cv2.imwrite(join(path,thresh_images), thresh_img)

    #Close holes with Rectangular 5x5 Kernel 
    kernal = np.ones((5, 5), np.uint8)
    closed_images = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernal)
    closed_image = f'closedcolony{n}.jpg'
    cv2.imwrite(join(path, closed_image), closed_images)

    # Find contours
    contours, hierarchy = cv2.findContours(closed_images, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
    #Determine number of contours
    contNumber = len(contours)

    #Find the lengths/sizes of each contour, store in an array
    contLengths = np.empty(contNumber)
    for i in range(0,contNumber):
        contLengths[i] = len(contours[i])

    #Determine the longest/largest contour and its position in contours array
    max1Contour = np.amax(contLengths)
    posMax1Contour = np.argmax(contLengths)

    #Designate contours
    cnt1 = contours[posMax1Contour]
    
    #Compute and store Growth Index
    area1[n] = cv2.contourArea(cnt1)
    #area2[n] = cv2.contourArea(cnt2)
    hyphae[n] = np.abs(area1[n] - area2)
    growthIndex[n] = hyphae[n]/np.minimum(area1[n],area2)


#Create DataFrame for csv files
data = {'Image Name': stringsforCSV,'Area 1': area1, 'Area 2': area2,'Growth Index': growthIndex}

#Read data into a CSV 
df = pd.DataFrame(data, columns = ['Image Name', 'Area 1', 'Area 2', 'Growth Index'])
df.to_csv('Other TammiCol Data.csv')