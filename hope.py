##No vuela, prende la cámara y analiza el código

import numpy as np
import matplotlib.pyplot as plt
import cv2
from djitellopy import Tello


apt42 = np.array([
    [1., 1.], [1., 9.], [2., 2.], [2., 4.], [2., 5.], [2., 6.], [2., 7.], [2., 8.],
    [3., 3.], [3., 4.], [3., 5.], [3., 6.], [3., 7.], [4., 2.], [4., 3.], [4., 4.],
    [4., 5.], [4., 6.], [5., 2.], [5., 3.], [5., 4.], [5., 5.], [5., 6.],
    [5., 7.], [5., 8.], [6., 2.], [6., 3.], [6., 4.], [6., 6.], [6., 7.], [6., 8.],
    [7., 3.], [7., 4.], [7., 6.], [7., 7.], [8., 3.], [8., 4.], [8., 6.], [8., 8.],
    [9., 1.], [9., 9.]
])
vertex=len(apt42)

def apriltag(tablero):
    tolerancia=10
    tablero=cv2.cvtColor(tablero, cv2.COLOR_BGR2RGB)
    #Recortar feed para que solo vea el centro con el apriltag
    tablero=tablero[70:870, 100:550, :] 
    tablero_gris=cv2.cvtColor(tablero,cv2.COLOR_BGR2GRAY)
    tablero_gris_float=np.float32(tablero_gris)

    #Detección de esquinas
    #Calibrar valor de kernelsize
    destino=cv2.cornerHarris(tablero_gris_float,2,15,0.04)
    destino=cv2.dilate(destino,None)

    ret, destino = cv2.threshold(destino,0.01*destino.max(),255,0)
    destino = np.uint8(destino)

    #Encuentra los centroides de los puntos que detectó el corner Harris
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(destino)
    #ni idea jaja, así venía en el ejemplo
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(tablero_gris_float,np.float32(centroids),(5,5),(-1,-1),criteria)

    #Ordeno la lista de las coordenadas de las esquinas según la primera columna
    print(len(corners))
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
    if len(map)==vertex:
        return True, tablero
    
    else:
        return False, tablero 


    


# Inicializar el dron Tello
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

# Iniciar la transmisión de video
tello.streamon()

try:
    while True:
        frame = tello.get_frame_read().frame  # Capturar un frame de la cámara del dron
        frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        id, frame= apriltag(frame)
        print(frame.shape)
        cv2.imshow("Tello Camera", frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrumpido manualmente.")

finally:
    # Detener el dron y cerrar todo
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()
