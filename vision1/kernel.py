import numpy as np
import matplotlib.pyplot as plt
import cv2
def load_img():
    blank_img=np.zeros((600,600))
    font=cv2.FONT_ITALIC
    cv2.putText(blank_img,"ABCDE",(50,300),font,5,(255,255,255),25)
    return blank_img

def display_img(img):
    fig=plt.figure(figsize=(12,10))
    ax= fig.add_subplot(111)
    ax.imshow(img,'gray')

canvas=load_img()
display_img(canvas)

#Erosion= erosiona los bordes de la imagen

kernel=np.ones((5,5),np.uint8)
erosion= cv2.erode(canvas,kernel,iterations=2)
display_img(erosion)



#Ruido blanco

white_noise=(np.random.randint(0,2,(600,600)))*255
black_noise=(np.random.randint(0,2,(600,600)))*(-255)
noisy= canvas+white_noise
blacknoisy= black_noise+canvas
blacknoisy[blacknoisy==-255]= 0
display_img(noisy)

#Open: ayuda a eliminar ruido de fondo
opening=cv2.morphologyEx(noisy,cv2.MORPH_OPEN,kernel)
display_img(opening)

#Close: ayuda a eliminar ruido de enfrente
display_img(blacknoisy)
closing=cv2.morphologyEx(blacknoisy,cv2.MORPH_CLOSE,kernel)
display_img(closing)

#Gradiente
gradiente=cv2.morphologyEx(canvas,cv2.MORPH_GRADIENT,kernel)
display_img(gradiente)

dilatar=cv2.dilate(canvas,kernel)
display_img(dilatar)