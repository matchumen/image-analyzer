import numpy as nm
import cv2
import matplotlib

#0 grayscale, 1 colored
img = cv2.imread('img/kitten.jpg',1)
cv2.imshow('cat',img)
height, width, channel = img.shape; #delete channel if grayscale

for x in range(width):
    for y in range(height):
        #Do manual grayscale
        img[y,x]=10;


cv2.waitKey(0)
cv2.destroyAllWindows()