#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Abrir imagen
img= Image.open("C:/Users/valef/Downloads/vision1/Archivos del curso/perro.jpg")
print(type(img))
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Convertir imagen a un arreglo
img_array = np.asarray(img)
print(type(img_array))
print(img_array.shape)

#Imprimir imagen
plt.imshow(img_array)

#Hacer una copia
img_rojo = img_array.copy()
print(img_rojo.shape)

#Imprimir cada matriz de cada canal
print('Canal R')
print(img_rojo[:,:,0])
print('Canal G')
print(img_rojo[:,:,1])
print('Canal B')
print(img_rojo[:,:,2])

plt.imshow(img_rojo[:,:,0], cmap = 'gray')
plt.imshow(img_rojo[:,:,1], cmap = 'gray')
plt.imshow(img_rojo[:,:,2], cmap = 'gray')

#Hagamos 0´s las componentes G y B
img_rojo[:,:,1] = 0
img_rojo[:,:,2] = 0
print(img_rojo[:,:,1])
print(img_rojo[:,:,2])
plt.imshow(img_rojo)

#Hagamos otra copia y 0's el canal R y B
img_verde = img_array.copy()
img_verde[:,:,0] = 0
img_verde[:,:,2] = 0
print(img_verde[:,:,0])
print(img_verde[:,:,2])
plt.imshow(img_verde)

img_azul = img_array.copy()
img_azul[:,:,0] = 0
img_azul[:,:,1] = 0
print(img_azul[:,:,0])
print(img_azul[:,:,1])
plt.imshow(img_azul)


#Suma de canales así la volvemos a la original
img_new = img_rojo + img_verde + img_azul
plt.imshow(img_new)

# %%
