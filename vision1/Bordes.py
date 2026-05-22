import numpy as np
import matplotlib.pyplot as plt
import cv2
perro = cv2.imread('C:/Users/valef/Downloads/vision1/Archivos del curso/perro_cara.jpg')
plt.figure()
plt.imshow(perro)

ba=cv2.Canny(perro,127,127)
bb=cv2.Canny(perro,0,255)
bc=cv2.Canny(perro,255,0)
bd=cv2.Canny(perro,0,127)
be=cv2.Canny(perro,127,0)
bf=cv2.Canny(perro,100,160)

f, grid = plt.subplots(2,3)

grid[0,0].imshow(ba)
grid[0,1].imshow(bb)
grid[0,2].imshow(bc)
grid[1,0].imshow(bd)
grid[1,1].imshow(be)
grid[1,2].imshow(bf)

grid[0,0].axis('off')
grid[0,1].axis('off')
grid[0,2].axis('off')
grid[1,0].axis('off')
grid[1,1].axis('off')
grid[1,2].axis('off')

mediana=np.median(perro)
lower=int(max(0,0.7*mediana))
upper=int(min(255,1.3*mediana))

blur=cv2.blur(perro,(3,3))
bordes=cv2.Canny(blur,lower,upper)
plt.figure()
plt.imshow(bordes)
