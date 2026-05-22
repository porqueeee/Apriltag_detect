import numpy as np
import matplotlib.pyplot as plt
import cv2

tolerancia=20

tablero=cv2.imread("apt4.jpeg")
tablero=cv2.cvtColor(tablero, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(tablero)

tablero_gris=cv2.cvtColor(tablero,cv2.COLOR_BGR2GRAY)
tablero_gris_float=np.float32(tablero_gris)

#Detección de esquinas
_, thresh = cv2.threshold(tablero_gris, 0, 255, cv2.THRESH_OTSU)
plt.figure()
plt.imshow(thresh, cmap="gray")
print(thresh)

destino=cv2.cornerHarris(tablero_gris,2,5,0.04)
destino=cv2.dilate(destino,(3,3))

ret, destino = cv2.threshold(destino,0.01*destino.max(),255,0)
destino = np.uint8(destino)
plt.figure()
plt.imshow(destino)
#Encuentra los centroides de los puntos que detectó el corner Harris
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(destino)
#ni idea jaja, así venía en el ejemplo
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(tablero_gris_float,np.float32(centroids),(5,5),(-1,-1),criteria)

#Ordeno la lista de las coordenadas de las esquinas según la primera columna
#print(len(corners))
corners=corners[corners[:, 0].argsort()]
#print(corners)

#Marca las esquinas
for i in range(len(corners)):
    cv2.drawMarker(tablero, ((int(corners[i][0])), (int(corners[i][1])) ), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

plt.figure()
plt.imshow(tablero)