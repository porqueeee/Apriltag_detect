import numpy as np
import matplotlib.pyplot as plt
import cv2

def display_img(img):
    fig=plt.figure(figsize=(10,10))
    ax= fig.add_subplot(111)
    ax.imshow(img,cmap='gray')

img = cv2.imread('C:/Users/valef/Downloads/vision1/Archivos del curso/crucigrama.jpg',0) #El 0 es para poner en blanco y negro
display_img(img)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,5)
display_img(sobelx)

sobely = cv2.Sobel(img,cv2.CV_64F,0,1,5)
display_img(sobely)

add=sobely+sobelx
display_img(add)

#sobelxy=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
#display_img(sobelxy)