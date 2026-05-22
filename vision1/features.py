import numpy as np
import matplotlib.pyplot as plt
import cv2

tablero = cv2.imread('C:/Users/valef/Downloads/vision1/Archivos del curso/ajedrez_real.jpg')
tablero=cv2.cvtColor(tablero, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(tablero)

tablero_gris=cv2.cvtColor(tablero,cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(tablero_gris,cmap='gray')

esquinas=cv2.goodFeaturesToTrack(tablero_gris,211,0.01,10)
esquinas=np.intp(esquinas)

for i in esquinas:
    x,y=i.ravel()
    cv2.circle(tablero,(x,y),5,[255,0,0],-1)

plt.figure()
plt.imshow(tablero)