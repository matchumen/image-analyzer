import cv2

image = cv2.imread('small.png',1)

print("Hello World!")

for i in range(1,len(image)):
    for j in range(1,len(image[1])):
        image[i,j] = image[i,j,0] * 0.3 + image[i,j,1] * 0.59 + image[i,j,2] * 0.11

cv2.imshow("Display window",image)
cv2.waitKey(0)
cv2.destroyAllWindows()