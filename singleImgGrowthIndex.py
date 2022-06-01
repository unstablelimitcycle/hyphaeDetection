import cv2
import numpy as np

# Same stuff as image_reader.py, but with just one image

img = cv2.imread('62_29.jpg')

img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('img_grey.jpg', img_grey)

img_blur = cv2.GaussianBlur(img_grey,(5,5), 0)
cv2.imwrite('img_blur.jpg', img_blur)

# Global Threshold
_, thresh_img = cv2.threshold(img_blur, 125, 255, cv2.THRESH_BINARY)
cv2.imwrite('thresh_img.jpg', thresh_img)

# Otsu's Threshold
_, otsu_thresh = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('otsu_thresh.jpg', otsu_thresh)

# Perform Opening then Closing with Rectangular Kernals
#kernal_open = np.ones((10, 10), np.uint8)
#kernal_close = np.ones((35, 35), np.uint8)
#Elliptical Kernals
kernal_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernal_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
#opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernal_open)
closing = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernal_close)
#cv2.imwrite('opening.jpg',opening)
cv2.imwrite('closing.jpg',closing)

#Trying Gaussian Kernel
# ksize = 6
# sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 

# kernal_gauss = cv2.getGaussianKernel(ksize, sigma) #Try Gaussian kernal

# openKernelGauss = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernal_gauss)
# closeKernelGauss = cv2.morphologyEx(openKernelGauss, cv2.MORPH_CLOSE, kernal_gauss)

# cv2.imwrite('openKernelGauss.jpg',openKernelGauss)
# cv2.imwrite('closeKernelGauss.jpg',closeKernelGauss)

#Find contours
contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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
cnt0 = contours[posMax1Contour]
cnt1 = contours[posMax2Contour]

# Approximate shape of contour 1 -> Not really needed now!
arc1 = cv2.arcLength(cnt0, True)
epsilon1 = 0.0001*arc1
approx1 = cv2.approxPolyDP(cnt0, epsilon1, True)

# Approximate shape of contour 2 -> Not really needed now!
arc2 = cv2.arcLength(cnt1, True)
epsilon2 = 0.001*arc2
approx2 = cv2.approxPolyDP(cnt1, epsilon2, True)

# Create all-zero arrays for storing drawn contours
img_contours = np.zeros(img.shape)

# Draw Contours
#img_contours = cv2.drawContours(img_grey, [cnt0], 0, (0,255,0), 1)
#img_contours = cv2.drawContours(img_grey, [approx1, approx2],-1,(0,254,0),2)
#cv2.drawContours(img_contours, [approx1, approx2],-1,(0,254,0),2)
#cv2.imwrite('approx_contours.jpg', img_contours)

#Try instead drawing contours on top of BW pic

cv2.drawContours(img_contours, [cnt0, cnt1], -1,(0,255,0),2)
cv2.imwrite('contours.jpg', img_contours)
#Find area within contour, compute growthIndex
area1 = cv2.contourArea(cnt0) # replace approx1 with cnt0
area2 = cv2.contourArea(cnt1) # replace approx2 with cnt2
hyphae = area1-area2
growthIndex = hyphae/area2
