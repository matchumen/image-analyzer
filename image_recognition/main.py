import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

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
            img[j,i] = img[j,i,2] * 0.3 + img[j,i,1] * 0.59 + img[j,i,0] * 0.11

    return img

def calcHistExample(grayscale_image):
    """opencv example""" 
    hist = cv2.calcHist([grayscale_image],[0],None,[256],[0,256])
    plt.plot(hist)
    plt.show()

def colorSpace_YCbCr(img):
    """custom implementation of CVTCOLOR"""
    height, width, channel = img.shape
    _img = img.copy()
    for i in range(width):
        for j in range(height):
            _img[j,i,0] = ((_img[j,i,0]*0.257) + (_img[j,i,1]*0.504) + (_img[j,i,2]*0.098)) + 16 #Y
            _img[j,i,1] = ((_img[j,i,0]*-0.148) + (_img[j,i,1]*-0.291) + (_img[j,i,2]*0.439)) + 128 #Cb
            _img[j,i,2] = ((_img[j,i,0]*0.439) + (_img[j,i,1]*-0.368) + (_img[j,i,2]*-0.071))+ 128 #Cr
    return _img

def colorSpace_HSI(img):
    height, width, channel = img.shape
    HSI = np.zeros((height,width,channel))
    for i in range(width):
        for j in range(height):
            #0-255 na 0-1 interval
            r = img[j,i,0]/255
            g = img[j,i,1]/255
            b = img[j,i,2]/255
            citatel = (1/2)*((r-g)+(r-b))
            jmenovatel = ((r-g)**2+(r-b)*(g-b))**(1/2)
            H = np.arccos(citatel/jmenovatel+0.000001)
            S = 1-(3/(r+g+b)+0.000001)*min([r,g,b])
            I = (r+g+b)/3
            HSI[j,i,0] = H
            HSI[j,i,1] = S
            HSI[j,i,2] = I
    return HSI


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

def imShowColorBar(img):
    imgplot = plt.imshow(img)
    plt.colorbar()
    plt.show()


def main():
    obrazek = 'img/kostky.png'
    img = cv2.imread(obrazek,1)#0 grayscale, 1 colored BGR format
    img_HSI = colorSpace_HSI(img)
    """
    #sezeni2 vlastní histogramy a grayscale
    cv2.imshow('cat',img)
    img_grayscale = custom_grayscale(img)
    cv2.imshow('cat2',img_grayscale)
    histogram(img_grayscale)5
    calcHistExample(img_grayscale)
    """
 
    """
    #sezeni 3 vlastni Ycbcr
    img_YCbCr = colorSpace_YCbCr(img)
    img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    cv2.imshow("def", img)
    cv2.imshow('custom', img_YCbCr)
    cv2.imshow('inbuilt', img_2)
    """
    """
    #sezeni 4 Segmentace obrazu na základě histogramu, zobrazit colorbar
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    imShowColorBar(img_ycbcr)
    img_ycbcr = custom_grayscale(showChannel(img_ycbcr,1))img_HSI = colorSpace_HSI(img)
    cv2.imshow("Puvodni", img)
    cv2.imshow('Seda 1/2 slozka', img_ycbcr)
    calcHistExample(img_ycbcr)
    hranice = 120 #podle histogramu 120 pro cr, 50 cb
    height, width, channel  = img_ycbcr.shape
    for i in range(width):
        for j in range(height):
            if(img_ycbcr[j,i,0] < hranice):
                img[j,i,0] = img[j,i,1] = img[j,i,2] = 0
    cv2.imshow("Segmentace", img)
    """

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()