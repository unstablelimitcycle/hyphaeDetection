# hyphaeDetection
Python project for detection of hyphae in yeast colonies

 - Python -v3.8.5
 - OpenCV -v4.5.1
 - Numpy -v1.19.2
 - Pandas - v1.2.3
 - matplotlib -v3.3.2

For bulk analysis, use image_reader.py
Input path of folder of images for analysis into folder_to_open = 'your_path_here'.
Set path for written images to be stored to in path = 'your_path_here'. 
Will then output csv with data, written images in folder designated.

For single images, use singleImgGrowthIndex.py
Simply add the name of the image to be analyzed to img = cv2.imread('Your Filename Here')

GUI coming ASAP!
