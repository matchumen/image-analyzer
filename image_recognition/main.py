import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogram(grayscale_image):
    """Custom histogram"""
    hist = np.zeros(256)
    height, width, channel = grayscale_image.shape
    for i in range(width):
        for j in range(height):
            intensity = grayscale_image[j,i,0]
            hist[intensity] += 1
    plt.title("Custom Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Amount")
    plt.plot(hist)
    plt.show()

def custom_grayscale(img):
    """Custom grayscale"""
    height, width, channel = img.shape; #delete channel if grayscale
    for i in range(width):
        for j in range(height):
            #img[y,x]=155;
            img[j,i] = img[j,i,0] * 0.3 + img[j,i,1] * 0.59 + img[j,i,2] * 0.11

    return img

def calcHistExample(grayscale_image):
    """opencv example""" 
    hist = cv2.calcHist([grayscale_image],[0],None,[256],[0,256])
    plt.plot(hist)
    plt.show()

def colorSpace_YCbCv(img):
    """custom implementation of CVTCOLOR"""
    height, width, channel = img.shape
    img_YCRCB = img.copy();
    for i in range(width):
        for j in range(height):
            img_YCRCB[j,i,0] = ((img_YCRCB[j,i,0]*0.257)+16) + ((img_YCRCB[j,i,1]*0.504) + 128) + ((img_YCRCB[j,i,2]*0.098)+128)
            img_YCRCB[j,i,1] = ((img_YCRCB[j,i,0]*-0.148)+16) + ((img_YCRCB[j,i,1]*-0.291) + 128) + ((img_YCRCB[j,i,2]*0.439)+128)
            img_YCRCB[j,i,2] = ((img_YCRCB[j,i,0]*0.439)+16) + ((img_YCRCB[j,i,1]*-0.368) + 128) + ((img_YCRCB[j,i,2]*-0.071)+128)
    return img_YCRCB

def showChannel(img, channel):
    x = img.copy()
    if(channel==0):
        x[:, :, 1] = 0
        x[:, :, 2] = 0
    if(channel==1):
        x[:, :, 0] = 0
        x[:, :, 2] = 0
    if(channel==2):
        x[:, :, 0] = 0
        x[:, :, 1] = 0
    return x


def main():
    obrazek = 'img/kostky.png'
    img = cv2.imread(obrazek,1)#0 grayscale, 1 colored

    """
    cv2.imshow('cat',img)
    img_grayscale = custom_grayscale(img)
    cv2.imshow('cat2',img_grayscale)
    histogram(img_grayscale)
    calcHistExample(img_grayscale)5
    """
    img_YCbCv = colorSpace_YCbCv(img)
    img_2 = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    cv2.imshow('custom', showChannel(img_YCbCv, 2))
    cv2.imshow('inbuilt', showChannel(img_2,2))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()