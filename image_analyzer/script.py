import cv2
import numpy as np
import matplotlib.pyplot as plt
import math as m


#==1.ukol===============================================================================================================

def custom_grayscale(image):
    for i in range(1,len(image)):
        for j in range(1,len(image[1])):
            image[i,j] = image[i,j,0] * 0.3 + image[i,j,1] * 0.59 + image[i,j,2] * 0.11

    return image

#==2.ukol===============================================================================================================

def custom_histogram(greyscale_image):
    hist = np.zeros(256)
    for i in range(1,len(greyscale_image)):
        for j in range(1,len(greyscale_image[1])):
            hist[greyscale_image[i,j]] += 1
    plt.title("Custom Histogram")
    plt.xlabel("Greyscale")
    plt.ylabel("Amount")
    plt.plot(hist)
    plt.show()

#==3.ukol==RGB>>YCbCr===================================================================================================
def rgb_to_ycbcr(image):
    height, width = image.shape[:2]
    ycbcr = image.copy()
    for i in range(0, height):
        for j in range(0, width):
            ycbcr[i, j, 0] = (image[i, j, 0] * 0.299) + (image[i, j, 1] * 0.587) + (image[i, j, 2] * 0.114) + 16
            ycbcr[i, j, 1] = (image[i, j, 0] * -0.169) + (image[i, j, 1] * -0.331) + (image[i, j, 2] * 0.500) + 128
            ycbcr[i, j, 2] = (image[i, j, 0] * 0.500) + (image[i, j, 1] * -0.419) + (image[i, j, 2] * -0.081) + 128

    return ycbcr

def rgb_to_hsi(image):
    height, width = image.shape[:2]
    hsi = image.copy()
    for i in range(height):
        for j in range(width):
            b = image[i,j,0]/255
            g = image[i,j,1]/255
            r = image[i,j,2]/255
            minimal = min([r,g,b])
            citatel = (1/2)*((r-g)+(r-b))
            jmenovatel = (r-g)**2+((r-b)*(g-b))**(1/2)
            hsi[i,j,0]=m.acos(citatel/jmenovatel)
            #hsi[i,j,1]=1-((3/(r+g+b))*minimal)
            hsi[i,j,2]=(r+g+b)/3

    return hsi

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

def isolate_channel(image, threshold, channel):
    ycbcr = image.copy()
    ycbcr = rgb_to_ycbcr(image)
    height, width = image.shape[:2]
    for i in range(height):
        for j in range(width):
            if ycbcr[i, j, channel] < threshold:
                image[i, j, :] = 0
    return image

#==LADENI===============================================================================================================
flowers = cv2.imread("kostky.png")
#cv2.imshow("Flowers",rgb_to_ycbcr(showChannel(flowers,1)))
#cv2.imshow("Flowers2", isolate_channel(flowers, 190, 1))
#custom_histogram(rgb_to_ycbcr(showChannel(flowers,2)))
#flowers2 = cv2.cvtColor(flowers, cv2.COLOR_BGR2YCrCb)
#cv2.imshow("Flowers2", flowers2)
#cv2.imshow("Flowers3", rgb_to_ycbcr(flowers));
cv2.imshow("Bricks", rgb_to_hsi(flowers))


cv2.waitKey(0)
cv2.destroyAllWindows()

'''
ukol 4 počítačové vidění 2017 přihlásit se jako host heslo: pvi
    přednáška č. 2 cca 
    0-min 1-max místo 255 nebo 65535
    prostor YCbCr 
    hue saturation 
    cvtcolor zkusit
    zprogramovat hue ycbcr, zobrazit jednu barevnou složku, jednu barvu roznásovit v matici prní sloupec r druhý sloupec g třetí sloupec b plus offset
'''