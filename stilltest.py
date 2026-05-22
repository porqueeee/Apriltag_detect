
import numpy as np
import matplotlib.pyplot as plt
import cv2
from djitellopy import Tello


ap0=np.array([0, 0], [0, 7], [1, 1], [1, 2], [1, 3], [1, 6], [2, 1])
ap1=np.array([0, 0], [0, 7], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5] )
ap2=np.array([0, 0], [0, 7], [1, 1], [1, 3], [1, 4], [2, 1], [2, 3])
ap3=np.array([0, 0], [0, 7], [1, 1], [1, 4], [2, 2], [2, 4], [2, 6])
ap4=np.array([0, 0], [0, 7], [1, 1], [1, 3], [2, 1], [2, 2], [2, 5])
ap5=np.array([0, 0], [0, 7], [1, 4], [1, 5], [1, 6], [2, 2], [2, 3])
ap6=np.array([0, 0], [0, 7], [1, 3], [1, 5], [2, 2], [2, 3], [2, 4])
ap7=np.array([0, 0], [0, 7], [1, 1], [1, 2], [1, 3], [1, 4], [2, 1])
ap8=np.array([0, 0], [0, 7], [1, 1], [1, 3], [1, 4], [1, 5], [2, 2])
ap9=np.array([0, 0], [0, 7], [1, 2], [1, 3], [1, 4], [1, 5], [2, 1])
ap10=np.array([0, 0], [0, 7], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3])
ap11=np.array([0, 0], [0, 7], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5])


def apriltag(tablero):
    #máxima distancia entre puntos para que sean considerados el mismo punto
    tolerancia=10
    #tablero = cv2.imread('tableapriltags.jpg')
    #tablero=tablero[300:700, 100:550]
    #tablero=cv2.imread("apt3.jpeg")
    tablero=cv2.cvtColor(tablero, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(tablero)

    tablero_gris=cv2.cvtColor(tablero,cv2.COLOR_BGR2GRAY)
    #plt.figure()
    #plt.imshow(tablero_gris,cmap='gray')

    tablero_gris_float=np.float32(tablero_gris)

    #Detección de esquinas

    destino=cv2.cornerHarris(tablero_gris_float,2,17,0.04)
    destino=cv2.dilate(destino,None)


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

    #Determina y dibuja el controno del apriltag
    length=len(corners)-1
    if abs(corners[0][1]-corners[length][1])>tolerancia:
        edge=length
    else:
        edge=length-1

    #Longitud del lado del cuadro
    side= int(abs(corners[0][1]-corners[edge][1]))
    #Longitud de cuadro de la cuadrícula
    fraction= int(side/8)

    plt.figure()
    plt.imshow(tablero)

    gridlist=[]
    organizer=[corners[0]]

    for i in range(1,len(corners)):
        
        #Si el punto está en la misma columna lo agrega a la lista

        if abs(corners[i][0]-corners[i-1][0])<=tolerancia and abs(corners[i][0]-corners[i-1][0])!=0:
            organizer.append(corners[i])
        #Si no, entonces ordena la lista según la segunda columna
        #Agrega esa lista ya ordenada a una lista y reinicia el organizer
        elif abs(corners[i][0]-corners[i-1][0])!=0:
            organizer=np.array(organizer)
            organizer=organizer[organizer[:, 1].argsort()]
            organizer=organizer.tolist()
            gridlist= gridlist + organizer
            organizer=[corners[i]]

    organizer=np.array(organizer)
    organizer=organizer[organizer[:, 1].argsort()]
    organizer=organizer.tolist()
    gridlist= gridlist + organizer
    organizer=[corners[i]]

    #print(len(gridlist))
    #print(gridlist)

    gridlist=np.array(gridlist)
    map=gridlist/fraction

    for i in range(len(map)):
        map[i][0]=int(map[i][0])
        map[i][1]=int(map[i][1])



    print(map)
    if len(map)==ap1:
        return True, tablero
    
    else:
        return False, tablero 

img=cv2.imread("apriltagtest.png")
id, frame= apriltag(img)
cv2.imshow(str(id), frame)
cv2.waitKey(0)