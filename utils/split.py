'''
python3 utils/split.py
'''

import cv2
import sys
import os

if not os.path.exists('patches'):
    os.makedirs('patches')

nRows = 10 #int(sys.argv[2])
# Number of columns
mCols = 10 #int(sys.argv[3])

# Reading image
img = cv2.imread('dataset/test/crack/1639043382760_image_detect_device_0_side_1_1_crop.jpg')
#print img

#cv2.imshow('image',img)

# Dimensions of the image
sizeX = img.shape[1]
sizeY = img.shape[0]

print(img.shape)


for i in range(0,nRows):
    for j in range(0, mCols):
        roi = img[i*sizeY/nRows:i*sizeY/nRows + sizeY/nRows ,j*sizeX/mCols:j*sizeX/mCols + sizeX/mCols]
        cv2.imshow('rois'+str(i)+str(j), roi)
        cv2.imwrite('patches/patch_'+str(i)+str(j)+".jpg", roi)


cv2.waitKey()