import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from hyphaeDetectionFunctions import load_images_from_folder
from PIL import Image


#Add pixel count to this file so that the pictures are in the correct order
folder = '/Users/lisarogers/Dropbox/Pictures for Lisa/Binary Pictures for Pixel Counting/WLP013, 100mM/'
path = '/Users/lisarogers/Dropbox/Pictures for Lisa/Binary Pictures for Pixel Counting/WLP013, 100mM/Updated Centers Filled/'

filenames = listdir(folder)
stringsforCSV = [f for f in filenames if f.endswith('.jpg')]

images = load_images_from_folder(folder)

#Initialize arrays for storing pixel count data
#First the ring pixel count
ringBlackPixelCount = np.empty(len(images), dtype=object)
ringWhitePixelCount = np.empty(len(images), dtype=object)

#Then center pixel count
centerBlackPixelCount = np.empty(len(images), dtype=object)
centerWhitePixelCount = np.empty(len(images), dtype=object)

#Initialize Area vectors for checking against pixel count
area1 = np.empty(len(images), dtype=object)
area2 = np.empty(len(images), dtype=object)
hyphae = np.empty(len(images), dtype=object)
minArea = np.empty(len(images), dtype=object)
ringDiff = np.empty(len(images), dtype=object)
centerDiff = np.empty(len(images), dtype=object)

for n in range(0,len(images)):
 
    CV32images = cv2.cvtColor(images[n], cv2.COLOR_BGR2GRAY)

    #Try without closing the images
    kernal = np.ones((5, 5), np.uint8)
    #kernal = kernal/sum(kernal)
    closed_images = cv2.morphologyEx(CV32images, cv2.MORPH_CLOSE, kernal)

    #Convert image to real binary to get pixel count
    im = Image.open(folder+f'binarycolony{n}.jpg')
    binaryimg = im.convert("1")

    #Initialize Pixel Count
    blackRingPixels=0
    whiteRingPixels=0

    for pixel in binaryimg.getdata():
        if pixel ==0:
            blackRingPixels +=1
        elif pixel == 255:
            whiteRingPixels += 1
    #store pixel counts for image        
    ringBlackPixelCount[n] = blackRingPixels
    ringWhitePixelCount[n] = whiteRingPixels

    # Find contours so we can fill the center of our image
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

    #Determine the second longest/largest contour and its position in contours array
    #Replace current maximum of ContLengths with 0.
    contLengths[:] = [0 if x==max1Contour else x for x in contLengths]

    #Recalculate maximum
    max2Contour =np.amax(contLengths)
    posMax2Contour = np.argmax(contLengths)

    #Designate contours
    cnt1 = contours[posMax1Contour]
    cnt2 = contours[posMax2Contour] 
    cnts = [cnt1, cnt2] 

    # Draw and Store Contours
    blankCanvas = np.zeros(CV32images.shape) 
    cv2.drawContours(blankCanvas, [cnt2], -1, (255,255,255), thickness=cv2.FILLED)
    image_contours = f'CenterFilled{n}.jpg'
    cv2.imwrite(join(path, image_contours), blankCanvas)

    area1[n] = cv2.contourArea(cnt1)
    area2[n] = cv2.contourArea(cnt2)
    hyphae[n] = np.abs(area1[n] - area2[n])
    minArea[n] = min(area1[n],area2[n])

    #Convert center filled images to real binary to get pixel count
    imC = Image.open(path+f'CenterFilled{n}.jpg')
    binaryimgC = imC.convert("1")

    #Initialize Pixel Count
    blackCenterPixels=0
    whiteCenterPixels=0

    for pixel in binaryimgC.getdata():
        if pixel ==0:
            blackCenterPixels +=1
        elif pixel == 255:
            whiteCenterPixels += 1
    #store pixel counts for image        
    centerBlackPixelCount[n] = blackCenterPixels
    centerWhitePixelCount[n] = whiteCenterPixels

    ringDiff[n] = 100*abs(ringWhitePixelCount[n]- hyphae[n])/ringWhitePixelCount[n]
    centerDiff[n] = 100*abs(centerWhitePixelCount[n]- minArea[n])/centerWhitePixelCount[n]

#Construct Data Frames to store all info and put in csv
data = {'Image Name': stringsforCSV,'Ring White Pixel Count': ringWhitePixelCount,'Hyphael Ring Area, OpenCV': hyphae, 'Ring % Difference': ringDiff,'Center White Pixel Count': centerWhitePixelCount,'Inner Contour Area, OpenCV': minArea, 'Center % Difference':centerDiff}
df = pd.DataFrame(data, columns = ['Image Name', 'Ring White Pixel Count', 'Hyphael Ring Area, OpenCV', 'Ring % Difference','Center White Pixel Count','Inner Contour Area, OpenCV', 'Center % Difference'])
df.to_csv('Pixel Counts for Strain WLP013 at 100uM.csv')