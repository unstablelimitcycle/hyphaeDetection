import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from hyphaeDetectionFunctions import load_images_from_folder

#Designate folder to open: For GUI, prompt user for input folder
folder_to_open1 = '/Users/lisarogers/Dropbox/Pictures for Lisa/02152021/0 uM/WLP566/'
folder_to_open2 = '/Users/lisarogers/Dropbox/Pictures for Lisa/02152021/100 uM/WLP566/'
folder_to_open3 = '/Users/lisarogers/Dropbox/Pictures for Lisa/02152021/200 uM/WLP566/'

#Designate path to write files to 
#path = '/Users/lisarogers/PythonStuff/writtenImagesNew/200uM/WLP566/'

#Get file name to use as string in CSV columns
filenames1 = listdir(folder_to_open1)
filenames2 = listdir(folder_to_open2)
filenames3 = listdir(folder_to_open3)

stringsforCSV1 = [f for f in filenames1 if f.endswith('.jpg')]
stringsforCSV2 = [f for f in filenames2 if f.endswith('.jpg')]
stringsforCSV3 = [f for f in filenames3 if f.endswith('.jpg')]

#Compare to see if strings match, then can use for cross comparison

#Load Images from designated folder
Images0uM = load_images_from_folder(folder_to_open1)
Images100uM = load_images_from_folder(folder_to_open2)
Images200uM = load_images_from_folder(folder_to_open3)

#Initalize Array: Contour Areas and Growth Index 
area1_0uM = np.empty(len(Images0uM), dtype=object)
area1_100uM = np.empty(len(Images100uM), dtype=object)
area1_200uM = np.empty(len(Images200uM), dtype=object)

area2_0uM = np.empty(len(Images0uM), dtype=object)
area2_100uM = np.empty(len(Images100uM), dtype=object)
area2_200uM = np.empty(len(Images200uM), dtype=object)

hyphae0uM = np.empty(len(Images0uM), dtype=object)
hyphae100uM = np.empty(len(Images100uM), dtype=object)
hyphae200uM = np.empty(len(Images200uM), dtype=object)

growthIndex0uM = np.empty(len(Images0uM), dtype=object)
growthIndex100uM = np.empty(len(Images100uM), dtype=object)
growthIndex200uM = np.empty(len(Images200uM), dtype=object)

#Need to now store the image data for ML purposes
for n in range(0, len(Images0uM)): #just use 0uM since 100uM and 200uM are the same length
   
    # Convert all images to greyscale
    images0uM = cv2.cvtColor(Images0uM[n], cv2.COLOR_BGR2GRAY)
    images100uM = cv2.cvtColor(Images100uM[n], cv2.COLOR_BGR2GRAY)
    images200uM = cv2.cvtColor(Images200uM[n], cv2.COLOR_BGR2GRAY)
   
    # Blur all to eliminate noise
    blur_images0uM = cv2.GaussianBlur(images0uM,(5,5), 0)
    blur_images100uM = cv2.GaussianBlur(images100uM,(5,5), 0)
    blur_images200uM = cv2.GaussianBlur(images200uM,(5,5), 0)

    # Threshhold images and save images
    _, thresh_images0uM = cv2.threshold(blur_images0uM,125,255,cv2.THRESH_BINARY)
    _, thresh_images100uM = cv2.threshold(blur_images100uM,125,255,cv2.THRESH_BINARY)
    _, thresh_images200uM = cv2.threshold(blur_images200uM,125,255,cv2.THRESH_BINARY)
  
    #Close holes with Rectangular 5x5 Kernel 
    kernal = np.ones((5, 5), np.uint8)
    closed_images0uM = cv2.morphologyEx(thresh_images0uM, cv2.MORPH_CLOSE, kernal)
    closed_images100uM = cv2.morphologyEx(thresh_images100uM, cv2.MORPH_CLOSE, kernal)
    closed_images200uM = cv2.morphologyEx(thresh_images200uM, cv2.MORPH_CLOSE, kernal)
   
    # Find contours
    contours0uM, hierarchy0uM = cv2.findContours(closed_images0uM, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours100uM, hierarchy100uM = cv2.findContours(closed_images100uM, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours200uM, hierarchy200uM = cv2.findContours(closed_images200uM, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Determine number of contours
    contNumber0uM = len(contours0uM)
    contNumber100uM = len(contours100uM)
    contNumber200uM = len(contours200uM)

    #Find the lengths/sizes of each contour, store in an array
    contLengths0uM = np.empty(contNumber0uM)
    contLengths100uM = np.empty(contNumber100uM)
    contLengths200uM = np.empty(contNumber200uM)

    for i in range(0,contNumber0uM):
        contLengths0uM[i] = len(contours0uM[i])

    for i in range(0,contNumber100uM):
        contLengths100uM[i] = len(contours100uM[i])

    for i in range(0,contNumber200uM):
        contLengths200uM[i] = len(contours200uM[i])        

    #Determine the longest/largest contour and its position in contours array
    max1Contour0uM = np.amax(contLengths0uM)
    posMax1Contour0uM = np.argmax(contLengths0uM)

    max1Contour100uM = np.amax(contLengths100uM)
    posMax1Contour100uM = np.argmax(contLengths100uM)

    max1Contour200uM = np.amax(contLengths200uM)
    posMax1Contour200uM = np.argmax(contLengths200uM)

    #Determine the second longest/largest contour and its position in contours array
    #Replace current maximum of ContLengths with 0.
    contLengths0uM[:] = [0 if x==max1Contour0uM else x for x in contLengths0uM]
    contLengths100uM[:] = [0 if x==max1Contour100uM else x for x in contLengths100uM]
    contLengths200uM[:] = [0 if x==max1Contour200uM else x for x in contLengths200uM]

    #Recalculate maximum
    max2Contour0uM =np.amax(contLengths0uM)
    posMax2Contour0uM = np.argmax(contLengths0uM)

    max2Contour100uM =np.amax(contLengths100uM)
    posMax2Contour100uM = np.argmax(contLengths100uM)

    max2Contour200uM =np.amax(contLengths200uM)
    posMax2Contour200uM = np.argmax(contLengths200uM)

    #Designate contours
    cnt1_0uM= contours0uM[posMax1Contour0uM]
    cnt2_0uM = contours0uM[posMax2Contour0uM]  

    cnt1_100uM= contours100uM[posMax1Contour100uM]
    cnt2_100uM = contours100uM[posMax2Contour100uM] 

    cnt1_200uM= contours200uM[posMax1Contour200uM]
    cnt2_200uM = contours200uM[posMax2Contour200uM] 

    # # Draw and Store Contours
    # img_contours = np.zeros(images.shape)
    # cv2.drawContours(img_contours, [cnt1, cnt2], -1, (0,255,0), 2)
    # #cv2.drawContours(images, contours, -1, (0,255,0), 2)
    
    #Compute and store Growth Index
    area1_0uM[n] = cv2.contourArea(cnt1_0uM)
    area2_0uM[n] = cv2.contourArea(cnt2_0uM)
    hyphae0uM[n] = np.abs(area1_0uM[n] - area2_0uM[n])
    growthIndex0uM[n] = hyphae0uM[n]/np.minimum(area1_0uM[n],area2_0uM[n])

    area1_100uM[n] = cv2.contourArea(cnt1_100uM)
    area2_100uM[n] = cv2.contourArea(cnt2_100uM)
    hyphae100uM[n] = np.abs(area1_100uM[n] - area2_100uM[n])
    growthIndex100uM[n] = hyphae100uM[n]/np.minimum(area1_100uM[n],area2_100uM[n])

    area1_200uM[n] = cv2.contourArea(cnt1_200uM)
    area2_200uM[n] = cv2.contourArea(cnt2_200uM)
    hyphae200uM[n] = np.abs(area1_200uM[n] - area2_200uM[n])
    growthIndex200uM[n] = hyphae200uM[n]/np.minimum(area1_200uM[n],area2_200uM[n])
    
#Statistics for each colony
meanIndex0uM = np.mean(growthIndex0uM)
medianIndex0uM =  np.median(growthIndex0uM)
stdIndex0uM = np.std(growthIndex0uM)
varIndex0uM = np.var(growthIndex0uM)

meanIndex100uM = np.mean(growthIndex100uM)
medianIndex100uM =  np.median(growthIndex100uM)
stdIndex100uM = np.std(growthIndex100uM)
varIndex100uM = np.var(growthIndex100uM)

meanIndex200uM = np.mean(growthIndex200uM)
medianIndex200uM =  np.median(growthIndex200uM)
stdIndex200uM = np.std(growthIndex200uM)
varIndex200uM = np.var(growthIndex200uM)

#How do I want to output csvs for the new statistics?

#Create DataFrame for csv files
data = {'Image Name': stringsforCSV,'Area 1': area1, 'Area 2': area2,'Growth Index': growthIndex}
data2 = {'Mean Index': [meanIndex], 'Median Index': [medianIndex], 'Std Dev Index': [stdIndex], 'Variance Index': [varIndex]}

#Read data into a CSV 
df = pd.DataFrame(data, columns = ['Image Name', 'Area 1', 'Area 2', 'Growth Index'])
df2  = pd.DataFrame(data2, columns = ['Mean Index', 'Median Index', 'Std Dev Index', 'Variance Index'])
newdf = df.append(df2)
newdf.to_csv('Data 20032021 WLP566 200uM.csv')
#df.to_csv('Analyzed Image Data')