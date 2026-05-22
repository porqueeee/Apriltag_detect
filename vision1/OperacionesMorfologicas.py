import numpy as np
import matplotlib.pyplot as plt
import cv2

canvas=cv2.imread("C:/Users/valef/Downloads/vision1/Archivos del curso/Noise2.png").astype(np.uint8)

#para poder visualizar todos los pasos en una sola imagen
f, grid = plt.subplots(2,2,figsize=(10,10))

#diferentes pasos requirieron diferentes tamaños de kernel
kernel=np.ones((5,5),np.uint8)
kernel2=np.ones((3,3),np.uint8)
kernel3=np.ones((7,7),np.uint8)

#Si no lo agrego, las operaciones erode y dialate no 
# detectan bien los bordes de la figura
blur=cv2.blur(canvas,(3,3))
grid[0,0].imshow(blur)

#Open reduce el ruido del fondo
open= cv2.morphologyEx(blur,cv2.MORPH_OPEN,kernel2)
grid[0,1].imshow(open)


#Probé múltiples configuraciones de open, close, erode y dilate 
# a ver que se veía mejor
dilatar=cv2.erode(open,kernel3)
dilatar=cv2.dilate(dilatar,kernel)
blur=cv2.blur(dilatar,(7,7))
grid[1,0].imshow(dilatar)

#Aunque se eliminó el ruido del fondo, el de la parte blanca no lo pude 
#eliminar, agregué un thresh para que se vea bonito, aunque no es parte del ejercicio.
_, thresh = cv2.threshold(dilatar,80,255,cv2.THRESH_BINARY)
grid[1,1].imshow(thresh)

#Configuración del display de las imágenes
grid[0,0].axis('off')
grid[0,1].axis('off')
grid[1,0].axis('off')
grid[1,1].axis('off')

grid[0,0].set_title("blur")
grid[0,1].set_title("open")
grid[1,0].set_title("combinación")
grid[1,1].set_title("thresh")

