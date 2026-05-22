import numpy as np
import matplotlib.pyplot as plt
import cv2

img=cv2.imread("C:/Users/valef/Downloads/vision1/Archivos del curso/chiikawa.jpg")
#img=cv2.imread("Archivos del curso/mulan.jpeg")
#Recorre los índices de los colores para que quede como RGB en vez de BGR
img=img[:,:,::-1]
plt.figure()

#Por default el tipo de dato de una imagen es uint8, limita los valores de 0 a 255
#Tenerlos como float permite realizar operaciones sin ese límite
red=img[:,:,0].astype(float)
green=img[:,:,1].astype(float)
blue=img[:,:,2].astype(float)

#Sirve para poder mostrar múltiples imágenes a la vez
f, grid = plt.subplots(2,2)


#iguala el canal verde [:,:,1] al promedio de los otros dos
img2=img.copy()
img3=img.copy()
img4=img.copy()
img5=img.copy()

#Después de realizar la operación regresamos al tipo de valor uint8
img2[:,:,1]=((red + blue)/2).astype(np.uint8)
grid[0,0].imshow(img2)


#Voltea esquina superior izquierda
img3[:576,:1024,:]=cv2.flip(img3[:576,:1024,:],1)
grid[1,0].imshow(img3)

#Calcula la luminancia y la aumenta si es menor a 100
Lum=np.mean((0.299*red,0.587*green,0.114*blue)).astype(np.uint8)
print("Luminancia=", Lum)
if Lum<100:
    #np.clip es necesario para mantener el rango de 0 a 255
    img5=np.clip((img5).astype(np.uint16)+30,0,255)
    grid[0,1].imshow(img5)


#Crea un efecto espejo
img4[:,1024:,:]=cv2.flip(img4[:,:1024,:],1)
grid[1,1].imshow(img4)

#Escala de grises :)
#img3[:,:,0]=((red + blue)/2).astype(np.uint8)
#img3[:,:,2]=img3[:,:,0]
#img3[:,:,1]=img3[:,:,0]
#grid[2].imshow(img3)
