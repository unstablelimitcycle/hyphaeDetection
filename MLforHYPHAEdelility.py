from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import cv2
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from hyphaeDetectionFunctions import load_images_from_folder

####Start with Image Processing####
#Designate folder to open
folder_to_open = '/Users/lisarogers/PythonStuff/allImages/CompleteData/'

#Designate path to write files to 
path = '/Users/lisarogers/PythonStuff/allImages/WrittenImagesforML/'
path2 = '/Users/lisarogers/PythonStuff/allImages/MLCleanData/'
path3 = '/Users/lisarogers/PythonStuff/allImages/MLOutliers/'

#Get file name to use as string in CSV columns
filenames = listdir(folder_to_open)
stringsforCSV = [f for f in filenames if f.endswith('.jpg')]

#Load Images from designated folder
Images = load_images_from_folder(folder_to_open)

#Initalize Array: Contour Areas and Growth Index 
area1 = np.empty(len(Images), dtype=object)
area2 = np.empty(len(Images), dtype=object)
hyphae = np.empty(len(Images), dtype=object)
growthIndex = np.empty(len(Images), dtype=object)
closed_images = []
outliers = []
usableImages = []

for n in range(0, len(Images)):
   
    # Convert all images to greyscale
    images = cv2.cvtColor(Images[n], cv2.COLOR_BGR2GRAY)
   
    # Blur all to eliminate noise
    blur_images = cv2.GaussianBlur(images,(5,5), 0)

    # Threshhold images and save images
    _, thresh_images = cv2.threshold(blur_images,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  
    #Close holes with Rectangular 5x5 Kernel 
    kernal = np.ones((5, 5), np.uint8)
    cImage = cv2.morphologyEx(thresh_images, cv2.MORPH_CLOSE, kernal)
    closed_images.append(cImage)
    #Output and save binarized, blurred, and closed images of colonies
    cImageFilename = f'BWcolony{n}.jpg'
 
    
    # Find contours
    contours, hierarchy = cv2.findContours(cImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
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

    #Compute and store Growth Index
    area1[n] = cv2.contourArea(cnt1)
    area2[n] = cv2.contourArea(cnt2)
    hyphae[n] = np.abs(area1[n] - area2[n])
    growthIndex[n] = hyphae[n]/np.minimum(area1[n],area2[n])
 
    #Removing outliers
    if growthIndex[n] > 5:
        #If growth index is above threshhold, put the image in the outliers list
        outliers.append(cImage)
        #Remove the entry from the area1, area2, hyphae, and growthIndex arrays
        #area1.pop(n)
        np.delete(area1,n)
        #area2.pop(n)
        np.delete(area2,n)
        #hyphae.pop(n)
        np.delete(hyphae,n)
        #growthIndex.pop(n)
        np.delete(growthIndex,n)
        #Save the Image File to the MLOutliers folder, path3
        cv2.imwrite(join(path3, cImageFilename), cImage)

    else:
        #If the growth index is less than the threshold, put the image in the clean/usable data list
        usableImages.append(cImage)
        #Put image file in MlCLeanData folder, path2
        cv2.imwrite(join(path2, cImageFilename), cImage)
        #Keep the area1, area2, hyphae, and growthIndex entries (no command needed)
    
    #Now all of the usable images are in one list/array, usableImages. 


#Create DataFrames for csv files - one with outliers, one without. AS long as each list is the same length, shouldn't have a problem
allData = {'Image Name': stringsforCSV,'Area 1': area1, 'Area 2': area2,'Growth Index': growthIndex}
usableData =  {'Image Name': stringsforCSV,'Area 1': area1, 'Area 2': area2,'Growth Index': growthIndex}

#Read data into CSV 
df = pd.DataFrame(allData, columns = ['Image Name', 'Area 1', 'Area 2', 'Growth Index'])
df.to_csv('Processed Yeast Image Data Complete.csv')

df2 = pd.Dataframe(usableData, columns = ['Image Name', 'Area1', 'Area2', 'Growth Index'])
df2.to_csv('Processed Yeast Image Data Outliers Removed.csv')

##### Start Machine Learning #####