import numpy as np
import matplotlib.pyplot as plt
import cv2

tablero = cv2.imread('C:/Users/valef/Downloads/vision1/Archivos del curso/tablero.jpg',0)

val,th1=cv2.threshold(tablero,160,255,cv2.THRESH_BINARY)

plt.figure()
plt.imshow(th1,'gray')

found,esquinas=cv2.findChessboardCorners(th1,(7,7))
cv2.drawChessboardCorners(th1,(7,7),esquinas,found)
print(found)
plt.figure()
plt.imshow(th1)

#-----------------------
puntos = cv2.imread('C:/Users/valef/Downloads/vision1/Archivos del curso/puntos.jpg',0)
