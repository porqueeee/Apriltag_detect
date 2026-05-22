import numpy as np
import matplotlib.pyplot as plt
import cv2

def show(img, title="Imagen", cmap =None, size =5):
    plt.figure(figsize=(size,size))
    if len(img.shape) == 2:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

img=cv2.imread("C:/Users/valef/Downloads/vision1/Archivos del curso/crucigrama.jpg",0)

corte, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)

show(thresh,"Crucigrama",cmap='gray')