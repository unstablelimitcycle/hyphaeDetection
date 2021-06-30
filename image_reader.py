import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib import pyplot as plt
from hyphaeDetectionFunctions import load_images_from_folder

#Designate folder to open
folder_to_open = '/Users/lisarogers/PythonStuff/allImages/CompleteData/'

#Designate path to write files to 
path = '/Users/lisarogers/PythonStuff/allImages/WrittenImagesforML/'

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

#Need to now store the image data for ML purposes
for n in range(0, len(Images)):
   
    # Convert all images to greyscale
    images = cv2.cvtColor(Images[n], cv2.COLOR_BGR2GRAY)
   
    # Blur all to eliminate noise
    blur_images = cv2.GaussianBlur(images,(5,5), 0)

    # Threshhold images and save images
    _, thresh_images = cv2.threshold(blur_images,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # thresh_image = f'threshholdcolony{n}.jpg'
    # cv2.imwrite(join(path, thresh_image), thresh_images)
  
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

    # # Approximate shape of contour 1 to draw
    # arc1 = cv2.arcLength(cnt1, True)
    # epsilon1 = 0.0001*arc1
    # approx1 = cv2.approxPolyDP(cnt1, epsilon1, True)

    # # Approximate shape of contour 2 to draw
    # arc2 = cv2.arcLength(cnt2, True)
    # epsilon2 = 0.001*arc2
    # approx2 = cv2.approxPolyDP(cnt2, epsilon2, True)

    # Draw and Store Contours
    
    cv2.drawContours(images, contours, -1, (0,255,0), 2)
    #image_contours = f'allContours{n}.jpg'
    #cv2.imwrite(join(path, image_contours), images)
    

    #Compute and store Growth Index
    area1[n] = cv2.contourArea(cnt1)
    area2[n] = cv2.contourArea(cnt2)
    hyphae[n] = np.abs(area1[n] - area2[n])
    growthIndex[n] = hyphae[n]/np.minimum(area1[n],area2[n])
    
    # #Introduce threshold for growthIndex
    # if growthIndex[n] > 5:
    #     outliers = [] #Set this outside of loop?
    #     growthIndex[n] =  outliers.append()
    #     else:
    #         return 
    
#Statistics for each colony
meanIndex = np.mean(growthIndex)
stdIndex = np.std(growthIndex)
varIndex = np.var(growthIndex)
print(meanIndex)
print(stdIndex)
print(varIndex)
fig = plt.hist(growthIndex, bins='auto')
#plt.show()
#plt.savefig('/Users/lisarogers/PythonStuff/Written Images 02152021/histogram.jpg')

#Create DataFrame for csv files
data = {'Image Name': stringsforCSV,'Area 1': area1, 'Area 2': area2,'Growth Index': growthIndex}

#Read data into a CSV 
df = pd.DataFrame(data, columns = ['Image Name', 'Area 1', 'Area 2', 'Growth Index'])
df.to_csv('Processed Yeast Image Data Complete.csv')


