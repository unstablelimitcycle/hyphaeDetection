import cv2
from os import listdir
from os.path import isfile, join
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from hyphaeDetectionFunctions import load_images_from_folder

def hyphaedelity(folder_to_open, filetype, written_path):
    #Get file name to use as string in CSV columns
    filenames = listdir(folder_to_open)
    if filetype.endswith('.jpg')==True: 
        stringsforCSV = [f for f in filenames if f.endswith('.jpg')]
    #elif filetype.endwith('.tiff')==True:
    #    stringsforCSV = [f for f in filenames if f.endswith('.tiff')]

    path = written_path
    # assert os.path.isfile(path)
    # with open(path,"r") as f:
    #     pass
 
    #Load Images from designated folder
    Images = load_images_from_folder(folder_to_open)

    #Initalize Array: Contour Areas and Growth Index 
    area1 = np.empty(len(Images), dtype=object)
    area2 = np.empty(len(Images), dtype=object)
    hyphae = np.empty(len(Images), dtype=object)
    growthIndex = np.empty(len(Images), dtype=object)

    #Need to now store the image data for ML purposes
    for n in range(0, len(Images)):
   
        # Convert all images to greyscale
        images = cv2.cvtColor(Images[n], cv2.COLOR_BGR2GRAY)
   
        # Blur all to eliminate noise
        blur_images = cv2.GaussianBlur(images,(5,5), 0)

        # Threshhold images and save images
        _, thresh_images = cv2.threshold(blur_images,125,255,cv2.THRESH_BINARY)
        thresh_image = f'binarycolony{n}.jpg'
        cv2.imwrite(join(path, thresh_image), thresh_images)
  
        #Close holes with Rectangular 5x5 Kernel 
        kernal = np.ones((5, 5), np.uint8)
        #kernal = kernal/sum(kernal)
        closed_images = cv2.morphologyEx(thresh_images, cv2.MORPH_CLOSE, kernal)
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

        #Determine the second longest/largest contour and its position in contours array
        #Replace current maximum of ContLengths with 0.
        contLengths[:] = [0 if x==max1Contour else x for x in contLengths]

        #Recalculate maximum
        max2Contour =np.amax(contLengths)
        posMax2Contour = np.argmax(contLengths)

        #Designate contours
        cnt1 = contours[posMax1Contour]
        cnt2 = contours[posMax2Contour]  


        # Draw and Store Contours
        img_contours = np.zeros(images.shape)
        cv2.drawContours(img_contours, [cnt1, cnt2], -1, (255,255,255), 2)
        image_contours = f'allContours{n}.jpg'
        cv2.imwrite(join(path, image_contours), img_contours)
    

        #Compute and store Growth Index
        area1[n] = cv2.contourArea(cnt1)
        area2[n] = cv2.contourArea(cnt2)
        hyphae[n] = np.abs(area1[n] - area2[n])
        growthIndex[n] = hyphae[n]/np.minimum(area1[n],area2[n])
    
    #Statistics for each colony
    meanIndex = np.mean(growthIndex)
    medianIndex =  np.median(growthIndex)
    stdIndex = np.std(growthIndex)
    medianArea  = np.median(area2)
    varIndex = np.var(growthIndex)

    #Create DataFrame for csv files
    data = {'Image Name': stringsforCSV,'Area 1': area1, 'Area 2': area2,'Growth Index': growthIndex}
    data2 = {'Mean Index': [meanIndex], 'Median Index': [medianIndex], 'Std Dev Index': [stdIndex], 'Variance Index': [varIndex]}

    #Read data into a CSV 
    df = pd.DataFrame(data, columns = ['Image Name', 'Area 1', 'Area 2', 'Growth Index'])
    df2  = pd.DataFrame(data2, columns = ['Mean Index', 'Median Index', 'Std Dev Index', 'Variance Index'])
    newdf = df.append(df2)
    newdf.to_csv(path+'/Test Data Output.csv')

    return 