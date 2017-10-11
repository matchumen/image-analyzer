import numpy as nm
import cv2
import matplotlib

#0 grayscale, 1 colored
img = cv2.imread('img/kitten.jpg',1)
cv2.imshow('cat',img)
height, width, channel = img.shape; #delete channel if grayscale

for i in range(width):
    for j in range(height):
        #Do manual grayscale
        #img[y,x]=155;
        img[j,i] = img[j,i,0] * 0.3 + img[j,i,1] * 0.59 + img[j,i,2] * 0.11
        #print(img[y,x])
        ddsadas
cv2.imshow('cat2',img)
cv2.waitKey(0)
cv2.destroyAllWindows()