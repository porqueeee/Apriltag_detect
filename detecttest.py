import numpy as np
import matplotlib.pyplot as plt
import cv2
from djitellopy import Tello

ap0=np.array([[0, 0], [0, 7], [1, 1], [1, 2], [1, 3], [1, 6], [2, 1]])
ap1=np.array([[0, 0], [0, 7], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5] ])
ap2=np.array([[0, 0], [0, 7], [1, 1], [1, 3], [1, 4], [2, 1], [2, 3]])
ap3=np.array([[0, 0], [0, 7], [1, 1], [1, 4], [2, 2], [2, 4], [2, 6]])
ap4=np.array([[0, 0], [0, 7], [1, 1], [1, 3], [2, 1], [2, 2], [2, 5]])
ap5=np.array([[0, 0], [0, 7], [1, 4], [1, 5], [1, 6], [2, 2], [2, 3]])
ap6=np.array([[0, 0], [0, 7], [1, 3], [1, 5], [2, 2], [2, 3], [2, 4]])
ap7=np.array([[0, 0], [0, 7], [1, 1], [1, 2], [1, 3], [1, 4], [2, 1]])
ap8=np.array([[0, 0], [0, 7], [1, 1], [1, 3], [1, 4], [1, 5], [2, 2]])
ap9=np.array([[0, 0], [0, 7], [1, 2], [1, 3], [1, 4], [1, 5], [2, 1]])
ap10=np.array([[0, 0], [0, 7], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3]])
ap11=np.array([[0, 0], [0, 7], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]])


# Inicializar el dron Tello
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
# Iniciar la transmisión de video
tello.streamon()

#máxima distancia entre puntos para que sean considerados el mismo punto
tolerancia=10

def apriltag(tablero):

    tablero=cv2.cvtColor(tablero, cv2.COLOR_BGR2RGB)
    tablero_gris=cv2.cvtColor(tablero,cv2.COLOR_BGR2GRAY)
    tablero_gris_float=np.float32(tablero_gris)

    #Detección de esquinas
    destino=cv2.cornerHarris(tablero_gris_float,2,7,0.04)
    destino=cv2.dilate(destino,None)

    ret, destino = cv2.threshold(destino,0.01*destino.max(),255,0)
    destino = np.uint8(destino)
    #Encuentra los centroides de los puntos que detectó el corner Harris
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(destino)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(tablero_gris_float,np.float32(centroids),(5,5),(-1,-1),criteria)

    #Ordeno la lista de las coordenadas de las esquinas según la primera columna
    corners=corners[corners[:, 0].argsort()]

    #Marca las esquinas
    for i in range(len(corners)):
        cv2.drawMarker(tablero, ((int(corners[i][0])), (int(corners[i][1])) ), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

    gridlist=[]
    organizer=[corners[0]]
    centro=[tablero.shape[0]/2, tablero.shape[1]/2]

    #Ordeno la lista según la segunda columna, respetando la primera
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


    print(gridlist)
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
    print(len(id))

    cv2.rectangle(tablero, ((int(gridlist[0][0])), (int(gridlist[0][1]))),  ((int(gridlist[0][0])+side), (int(gridlist[0][1])+side)),  (0, 0, 255))
    cv2.rectangle(tablero, ((int(gridlist[0][0])+fraction), (int(gridlist[0][1])+fraction)),  ((int(gridlist[-1][0])-fraction), (int(gridlist[-1][1]))-fraction ),  (255, 0, 0))

    print(tablero.shape)
    
    return tablero


try:
    while True:
        frame = tello.get_frame_read().frame  # Capturar un frame de la cámara del dron
        frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Mostrar las figuras y colores detectados en la ventana
        detect=apriltag(frame)
        cv2.imshow("Tello Camera", detect)

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
