#Lee apriltags, les da un id basado en los vértices de sus secciones blancas
#También marca el centro de la imagen, espero no sea tanto problema


import numpy as np
import matplotlib.pyplot as plt
import cv2

def apriltag(tablero):
    #máxima distancia entre puntos para que sean considerados el mismo punto
    tolerancia=10

    #Consigue imagen
    #tablero = cv2.imread('tableapriltags.jpg')
    #tablero=tablero[300:700, 100:550]
    #tablero=cv2.imread("apt3.jpeg")
    tablero=cv2.cvtColor(tablero, cv2.COLOR_BGR2RGB)

    tablero_gris=cv2.cvtColor(tablero,cv2.COLOR_BGR2GRAY)
    #plt.figure()
    #plt.imshow(tablero_gris,cmap='gray')

    tablero_gris_float=np.float32(tablero_gris)

    #Detección de esquinas

    destino=cv2.cornerHarris(tablero_gris_float,2,7,0.04)
    destino=cv2.dilate(destino,None)
    #plt.figure()
    #plt.imshow(destino)

    ret, destino = cv2.threshold(destino,0.01*destino.max(),255,0)
    destino = np.uint8(destino)
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

    gridlist=[]
    organizer=[corners[0]]
    centro=[tablero.shape[0]/2, tablero.shape[1]/2]

    for i in range(1,len(corners)):
        #Si el punto está en la misma columna lo agrega a la lista
        if abs(corners[i][0]-corners[i-1][0])<=tolerancia and corners[i][0]!=corners[i-1][0]:
            organizer.append(corners[i])
        #Si no, entonces ordena la lista según la segunda columna
        #Agrega esa lista ya ordenada a una lista y reinicia el organizer
        elif corners[i][0]!=corners[i-1][0]:
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


    #print(gridlist)
    #Determina y dibuja el controno del apriltag
    length=len(gridlist)-1
    #Longitud del lado del cuadro
    side= int(abs(gridlist[0][1]-gridlist[-1][1]))
    #Longitud de cuadro de la cuadrícula
    fraction= int(side/7)
    if fraction==0:
        fraction=1

    gridlist=np.array(gridlist)
    map=gridlist/fraction

    id=[ [int(map[0][0]), int(map[0][1])]  ]
    for i in range(1,len(map)):
        if int(map[i][0])==int(map[i-1][0]) and int(map[i][1])==int(map[i-1][1]):
            print(":(")
        else:
            id.append( [int(map[i][0]),int(map[i][1])])

    print(id)
    #print(len(id))

    cv2.rectangle(tablero, ((int(gridlist[0][0])), (int(gridlist[0][1]))),  ((int(gridlist[0][0])+side), (int(gridlist[0][1])+side)),  (0, 0, 255))
    cv2.rectangle(tablero, ((int(gridlist[0][0])+fraction), (int(gridlist[0][1])+fraction)),  ((int(gridlist[-1][0])-fraction), (int(gridlist[-1][1]))-fraction ),  (255, 0, 0))

    return tablero


tablero = cv2.imread('ap4.jpeg')
tablero=apriltag(tablero)

tablero = cv2.imread('ap4.jpeg')
tablero= cv2.rotate(tablero,cv2.ROTATE_90_CLOCKWISE)
tablero=apriltag(tablero)

tablero = cv2.imread('ap4.jpeg')
tablero= cv2.rotate(tablero,cv2.ROTATE_180)
tablero=apriltag(tablero)
plt.figure()
plt.imshow(tablero)

tablero = cv2.imread('ap4.jpeg')
tablero= cv2.rotate(tablero,cv2.ROTATE_90_COUNTERCLOCKWISE)

tablero=apriltag(tablero)
#plt.figure()
#plt.imshow(tablero)

