import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from hyphaeDetectionFunctions import load_images_from_folder
from PIL import Image


#Add pixel count to this file so that the pictures are in the correct order
folder = '/Users/lisarogers/Dropbox/Pictures for Lisa/Binary Pictures for Pixel Counting/WLP013, 0mM/'
path = '/Users/lisarogers/Dropbox/Pictures for Lisa/Binary Pictures for Pixel Counting/WLP013, 0mM/Centers Filled/'

images = load_images_from_folder(folder)

for n in range(0,len(images)):
 
    CV32images = cv2.cvtColor(images[n], cv2.COLOR_BGR2GRAY)

    kernal = np.ones((5, 5), np.uint8)
    #kernal = kernal/sum(kernal)
    closed_images = cv2.morphologyEx(CV32images, cv2.MORPH_CLOSE, kernal)


    #Convert image to real binary to get pixel count
    im = Image.open(path+'')
    binaryimg = im.convert("1")
    unknownPixels=0
    blackPixels=0
    whitePixels=0
    for pixel in binaryimg.getdata():
        

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
    #img_contours = np.zeros(CV32images.shape)
    cv2.drawContours(closed_images, [cnt2], -1, (255,255,255), thickness=cv2.FILLED)
    #cv2.imshow()
    #cv2.drawContours(images, contours, -1, (0,255,0), 2)
    image_contours = f'allContours{n}.jpg'
    cv2.imwrite(join(path, image_contours), closed_images)
