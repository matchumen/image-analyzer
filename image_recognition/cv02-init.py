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
#plt.plot(hist_object) 
prah_dolni = 115;
prah_horni = 135;
init = True;

while True:
    ret, frame = cap.read()
    if not ret:
        break 
    height, width, channel = frame.shape
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hist, b = np.histogram(frame_hsv[:,:,0], 256, (0, 256))
    frame_hsv[(frame_hsv[:,:, 0] < prah_dolni) | (frame_hsv[:,:, 0] > prah_horni)] = 0
    frame_hsv[(frame_hsv[..., 0] >= prah_dolni) & (frame_hsv[..., 0] <= prah_horni)] = 255
    x_sum = 0
    y_sum = 0
    x_times = 0
    y_times = 0
    if (init):
        for i in range(width):
            for j in range(height):
                if (frame_hsv[j, i, 0] == 255):
                    x_sum += i
                    y_sum += j
                    x_times += 1
                    y_times += 1
                    init = False;
    else:
        for i in range(x1, x2-1):
            for j in range(y1, y2-1):
                if (frame_hsv[j, i, 0] == 255):
                    x_sum += i
                    y_sum += j
                    x_times += 1
                    y_times += 1
    x1 = int(x_sum/x_times)-50
    y1 = int(y_sum/y_times)-70
    x2 = int(x_sum/x_times)+50
    y2 = int(y_sum/y_times)+70
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
    cv2.imshow('Image', frame)
    key = 0xFF & cv2.waitKey(30)
    if key == 27:
        break
    
cv2.destroyAllWindows()