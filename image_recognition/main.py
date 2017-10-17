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

def main():
    img = cv2.imread('img/kitten.jpg',1)#0 grayscale, 1 colored
    #cv2.imshow('cat',img)
    img_grayscale = custom_grayscale(img)
    #cv2.imshow('cat2',img_grayscale)
    #histogram(img_grayscale)
    #calcHistExample(img_grayscale)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()