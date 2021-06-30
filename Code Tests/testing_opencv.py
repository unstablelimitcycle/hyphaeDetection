import cv2
import numpy as np

# Testing same stuff as image_reader.py, but with just one image

img = cv2.imread('samplepic.jpg')

img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('img_grey.jpg',img_grey)

img_blur = cv2.GaussianBlur(img_grey,(5,5), 0)
cv2.imwrite('img_blur.jpg', img_blur)


_, thresh_img = cv2.threshold(img_blur, 125, 255, cv2.THRESH_BINARY)
cv2.imwrite('thresh_img.jpg', thresh_img)
# _, thresh_img2 = cv2.threshold(img_blur, 130, 255, cv2.THRESH_BINARY)
# cv2.imwrite('thresh_img2.jpg', thresh_img2)

# Perform Closing with Rectangular 5x5 Kernel 
kernal = np.ones((5, 5), np.uint8)
#kernal = kernal/sum(kernal)
closing = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernal)
cv2.imwrite('closing.jpg',closing)

# Try Canny - very precise outlines of inner and outer ring
# canny_output = cv2.Canny(img, 125, 255)
# closed_canny_output = cv2.Canny(closing, 125, 255)
# cv2.imwrite('canny.jpg',canny_output)
# cv2.imwrite('canny_closing.jpg',closed_canny_output)

#Find contours
# contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# canny_contours, canny_hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# closed_canny_contours, closed_canny_hierarchy = cv2.findContours(closed_canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt0 = contours[0]
cnt1 = contours[1]

# Approximate shape of contour 1
arc1 = cv2.arcLength(cnt0, True)
epsilon1 = 0.001*arc1
approx1 = cv2.approxPolyDP(cnt0, epsilon1, True)

# Approximate shape of contour 2
arc2 = cv2.arcLength(cnt1, True)
epsilon2 = 0.001*arc2
approx2 = cv2.approxPolyDP(cnt1, epsilon2, True)


# Create all-zero arrays for storing drawn contours
img_contours = np.zeros(img.shape)
closed_contour1 = np.zeros(img.shape)
closed_contour2 = np.zeros(img.shape)
# canny_img_contours = np.zeros(img.shape)
# closed_canny_img_contours = np.zeros(closing.shape)

# Draw Contours
cv2.drawContours(img_contours, contours, -1, (0,255,0), 1)
cv2.drawContours(closed_contour1, [approx1],-1,(0,255,0),1)
cv2.drawContours(closed_contour2, [approx2],-1,(0,255,0),1)
cv2.imwrite('curve1.jpg', closed_contour1)
cv2.imwrite('curve2.jpg', closed_contour2)
# newcurve = cv2.imread('curve.jpg')

# cv2.drawContours(canny_img_contours, canny_contours, -1, (0,255,0), 1)
# cv2.drawContours(closed_canny_img_contours, canny_contours, -1, (0,255,0), 1)

#Find area within contour
area1 = cv2.contourArea(approx1)
area2 = cv2.contourArea(approx2)
hyphae = area1-area2
growthIndex = 100*hyphae/area1
print(area1)
print(area2)
print(growthIndex)

cv2.imwrite('contours.jpg',img_contours)
# cv2.imwrite('canny_contours.jpg', canny_img_contours)
# cv2.imwrite('closed_canny_contours.jpg', closed_canny_img_contours)

quit()

# Create kernels
# Rectangular with LPF
kernal = np.ones((5, 5), np.uint8)
kernal = kernal/sum(kernal)

# Rectangular with HPF
kernal_hp = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
kernal_hp = kernal_hp/(np.sum(kernal_hp) if np.sum(kernal_hp) != 0 else 1)

# Elliptical
ell_kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Blur
blur_kernal = cv2.blur(img, (5, 5))

# Apply Low-Pass Filter to Image
img_lpf = cv2.filter2D(img, -1, kernal)
# Print Image after Low-Pass Filter
cv2.imwrite('lowpassresult.jpg', img_lpf)

# dilate
dilation = cv2.dilate(img, kernal, iterations=1)

# Apply High-Pass Filter to Image
img_hpf = cv2.filter2D(img, -1, kernal_hp)
dil_hpf = cv2.filter2D(dilation, -1, kernal_hp)
# Print image after High-Pass filter
cv2.imwrite('highpassresult.jpg', img_hpf)
cv2.imwrite('highpassresult2.jpg', dil_hpf)

# Perform Morphological Operations and print
# Erosion
erosion = cv2.erode(img, kernal, iterations=1)
ero2 = cv2.erode(img, ell_kernal, iterations=1)
ero3 = cv2.erode(img, blur_kernal, iterations=1)
cv2.imwrite('erosion.png', erosion)
cv2.imwrite('erosion2.png', ero2)
cv2.imwrite('erosion3.png', ero3)

# Dilation
dilation = cv2.dilate(img, kernal, iterations=1)
dil2 = cv2.dilate(img, ell_kernal, iterations=1)
dil3 = cv2.dilate(img, blur_kernal, iterations=1)
cv2.imwrite('dilation.png', dilation)
cv2.imwrite('dilation2.png', dil2)
cv2.imwrite('dilation3.png', dil3)

# Opening
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernal)
ope2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, ell_kernal)
ope3 = cv2.morphologyEx(img, cv2.MORPH_OPEN, blur_kernal)
cv2.imwrite('opening.png', opening)
cv2.imwrite('opening2.png', ope2)
cv2.imwrite('opening3.png', ope3)

# CLosing
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernal)
clo2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ell_kernal)
clo3 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, blur_kernal)
cv2.imwrite('closing.png', closing)
cv2.imwrite('closing2.png', clo2)
cv2.imwrite('closing3.png', clo3)

# Gradient
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernal)
grad2 = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, ell_kernal)
grad3 = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, blur_kernal)
cv2.imwrite('gradient.png', gradient)
cv2.imwrite('gradient2.png', grad2)
cv2.imwrite('gradient3.png', grad3)
