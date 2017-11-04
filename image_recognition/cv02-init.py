import numpy as np
import matplotlib.pyplot as plt
import cv2
plt.ion()
#clear = lambda: os.system('cls')
#clear()
plt.close('all')

cap = cv2.VideoCapture('video/cv02_hrnecek.mp4')
object = cv2.imread('img/cv02_vzor_hrnecek.bmp')
object_hsv = cv2.cvtColor(object, cv2.COLOR_RGB2HSV)
#cv2.imshow("Hue", object_hsv[:,:,0])
hist_object, b = np.histogram(object_hsv[:,:,0], 256, (0, 256))
indexMaxHodnoty = np.argmax(hist_object, axis = 0)
pocet = hist_object[indexMaxHodnoty]
hist_object = hist_object.astype(dtype="double")
for i in range(len(hist_object)):
    hist_object[i] = round((hist_object[i]/pocet),3);
plt.plot(hist_object) #110 až 140

while True:
    ret, frame = cap.read()
    if not ret:
        break 
    height, width, channel = frame.shape
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    maska = frame_hsv
    hist, b = np.histogram(frame_hsv[:,:,0], 256, (0, 256))
    for i in range(width):
        for j in range(height):
            if(frame_hsv[j,i,0]<110 or frame_hsv[j,i,0] > 140):
                maska[j,i,0] = 0   
                maska[j,i,1] = 0
                maska[j,i,2] = 0 
            #Xt= sum(sum())
            #Yt= sum(sum())
    frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2RGB)
    x1 = 100
    y1 = 100
    x2 = 200
    y2 = 200
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
    cv2.imshow('Image', frame)
    key = 0xFF & cv2.waitKey(30)
    if key == 27:
        break
    
cv2.destroyAllWindows()