import numpy as np
import matplotlib.pyplot as plt
import cv2

img=cv2.imread("C:/Users/valef/Downloads/vision1/Archivos del curso/formas.png",0)
plt.figure()
plt.imshow(img,cmap='gray')
copy=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

#Binarizar
ret, thres=cv2.threshold(img,127,255,cv2.THRESH_BINARY)

contour,jerarquia= cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contour)):
    if jerarquia[0][i][3]==-1:
        copy=cv2.drawContours(copy,contour,i,(0,255,0),3)
    else:
        copy=cv2.drawContours(copy,contour,i,(0,0,255),3)
plt.imshow(copy)
print(len(contour))
print('\n')
print(jerarquia)
print(contour)