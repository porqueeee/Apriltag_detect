import numpy as np
import matplotlib.pyplot as plt
import cv2

#Dirección de la imagen
img=cv2.imread("C:/Users/valef/Downloads/vision1/Archivos del curso/soldadura5.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
copy=img.copy()
plt.figure(figsize=(10,10))
f, grid = plt.subplots(2,2) 
grid[0,0].axis('off')
grid[0,1].axis('off')
grid[1,0].axis('off')
grid[1,1].axis('off')

grid[0,0].set_title("Blur + grayscale")
grid[0,1].set_title("Threshold")
grid[1,0].set_title("Morphology")
grid[1,1].set_title("Find Contours")

kernel=np.ones((5,5),np.uint8)

gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#grid[0,0].imshow(gray,cmap="gray")

blur=cv2.blur(gray,(3,3))
grid[0,0].imshow(blur,cmap="gray")

_, thresh=cv2.threshold(blur,60,180,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
grid[0,1].imshow(thresh,cmap="gray")

open=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
open=cv2.erode(open,kernel)

grid[1,0].imshow(open,cmap="gray")

#canny=cv2.Canny(thresh,30,100)
#grid[1,0].imshow(canny)

contour,jerarquia= cv2.findContours(open,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
copy=cv2.drawContours(copy,contour,-1,(0,255,0),-1)
grid[1,1].imshow(copy)
print(len(contour))