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
        #img[y,x]=155;
        r=img[y,x][0]*0.30;
        g=img[y,x][1]*0.59;
        b=img[y,x][2]*0.11;
        img[y,x][0]=r;
        img[y,x][1]=r;
        img[y,x][2]=r;
        #print(img[y,x])

cv2.imshow('cat2',img)
cv2.waitKey(0)
cv2.destroyAllWindows()