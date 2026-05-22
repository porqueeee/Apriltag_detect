#Template matching

import numpy as np
import matplotlib.pyplot as plt
import cv2

plantilla = cv2.imread('C:/Users/valef/Downloads/vision1/Archivos del curso/perro.jpg')
plantilla=cv2.cvtColor(plantilla,cv2.COLOR_BGR2RGB)
print(plantilla.shape)

cara = cv2.imread('C:/Users/valef/Downloads/vision1/Archivos del curso/perro_cara.jpg')
cara=cv2.cvtColor(cara,cv2.COLOR_BGR2RGB)
print(cara.shape)

metodos=['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']


for m in metodos:
    copia=plantilla.copy()
    mi_metodo=eval(m)
    res=cv2.matchTemplate(plantilla,cara,mi_metodo)
    min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
    
    if m in ['cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']:
        esquina=(min_loc[0]+cara.shape[0],min_loc[1]+cara.shape[1])
        cv2.rectangle(copia, min_loc,esquina, (0,0,255), 3)
    
    else:
        esquina=(max_loc[0]+cara.shape[0],max_loc[1]+cara.shape[1])
        cv2.rectangle(copia, max_loc,esquina, (0,0,255), 3)
    plt.figure(figsize=(10,10))

    plt.subplot(121)
    plt.imshow(res)
    plt.title('Resultado')

    plt.subplot(122)
    plt.imshow(copia)
    plt.title('Detección')