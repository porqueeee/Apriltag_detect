import numpy as np
import matplotlib.pyplot as plt
import cv2

tablero = cv2.imread('C:/Users/valef/Downloads/vision1/Archivos del curso/tablero.jpg')
tablero=cv2.cvtColor(tablero, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(tablero)

tablero_gris=cv2.cvtColor(tablero,cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(tablero_gris,cmap='gray')

ajedrez = cv2.imread('C:/Users/valef/Downloads/vision1/Archivos del curso/ajedrez_real.jpg')
ajedrez=cv2.cvtColor(ajedrez, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(ajedrez)

ajedrez_gris=cv2.cvtColor(ajedrez,cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(ajedrez_gris,cmap='gray')

tablero_gris_float=np.float32(tablero_gris)
ajedrez_gris_float=np.float32(ajedrez_gris)

#Detección de esquinas

destino=cv2.cornerHarris(tablero_gris_float,2,3,0.04)
destino=cv2.dilate(destino,None)
plt.figure()
plt.imshow(destino)

#colorea de rojo en la imagen original el 1% de medidas más 
# "brillantes" del resultado de corner Harris
tablero[destino>0.1*destino.max()]=[255,0,0]
plt.figure()
plt.imshow(tablero)

#--------------------------------------

destino2=cv2.cornerHarris(ajedrez_gris_float,2,3,0.04)
destino2=cv2.dilate(destino2,None)


ajedrez[destino2>0.1*destino2.max()]=[255,0,0]
plt.figure()
plt.imshow(ajedrez)