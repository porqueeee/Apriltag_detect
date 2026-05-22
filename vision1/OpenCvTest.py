#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2

img=cv2.imread("C:/Users/valef/Downloads/vision1/Archivos del curso/perro.jpg")
plt.imshow(img)
img_RGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img_RGB)
red=img_RGB[:,:,0]
green=img_RGB[:,:,1]
blue=img_RGB[:,:,2]
plt.imshow(red)
print(green.shape)
